#!/usr/bin/env python3
"""lmms-eval plugin for base and generic-retrofitted HF VLMs."""

from __future__ import annotations

import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple

import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from accelerate.utils import InitProcessGroupKwargs
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from generic_vlm_attnres_retrofit import load_generic_retrofit
from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models import MODEL_REGISTRY_V2
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.registry_v2 import ModelManifest
from lmms_eval.protocol import ChatMessages


@register_model("generic_vlm_attnres")
class GenericVLMHf(lmms):
    """A minimal image-text generation adapter shared by both test families."""

    is_simple = False

    def __init__(
        self,
        pretrained: str,
        retrofit_state_path: str | None = None,
        revision: str = "main",
        device: str = "cuda",
        device_map: str = "auto",
        batch_size: int = 1,
        trust_remote_code: bool = False,
        fix_mistral_regex: bool = False,
        low_cpu_mem_usage: bool = False,
        attn_implementation: str | None = None,
        use_cache: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs
        batch_size = int(batch_size)
        if batch_size != 1:
            raise ValueError("generic_vlm_attnres currently requires batch_size=1")
        self.batch_size_per_gpu = batch_size
        self.path = pretrained
        self.use_cache = bool(use_cache)

        accelerator = Accelerator(
            kwargs_handlers=[
                InitProcessGroupKwargs(timeout=timedelta(weeks=52))
            ]
        )
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            resolved_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            resolved_map = device_map if device_map else device

        load_kwargs = {
            "revision": revision,
            "dtype": torch.bfloat16,
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "trust_remote_code": trust_remote_code,
            "device_map": resolved_map,
        }
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.path,
            **load_kwargs,
        ).eval()
        self._retrofit = None
        if retrofit_state_path:
            self._retrofit = load_generic_retrofit(
                self._model,
                retrofit_state_path,
            )
            self._model.eval()
            print(
                f"[generic-vlm-lmms] loaded retrofit {retrofit_state_path}",
                flush=True,
            )
        self._config = self._model.config
        self.processor = AutoProcessor.from_pretrained(
            self.path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            fix_mistral_regex=fix_mistral_regex,
        )
        self._tokenizer = self.processor.tokenizer
        self._max_length = int(
            getattr(self._config.text_config, "max_position_embeddings", 8192)
        )

        if accelerator.num_processes > 1:
            if accelerator.distributed_type not in {
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            }:
                raise ValueError(
                    f"Unsupported distributed type {accelerator.distributed_type}"
                )
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                config = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": (
                        self.batch_size_per_gpu * accelerator.num_processes
                    ),
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(
                    must_match=True,
                    **config,
                )
            if accelerator.distributed_type in {
                DistributedType.FSDP,
                DistributedType.DEEPSPEED,
            }:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(
                    self.model,
                    evaluation_mode=True,
                )
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self.accelerator.unwrap_model(self._model)

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self):
        return (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @staticmethod
    def _flatten(items):
        return [value for group in items for value in group]

    def generate_until(
        self,
        requests: List[Instance],
    ) -> List[GenerationResult]:
        results: List[GenerationResult] = []

        def collate(item):
            return item[2], item[2]

        reordered = utils.Collator(
            [request.args for request in requests],
            collate,
            group_fn=lambda item: item[2],
            grouping=True,
        )
        chunks = reordered.get_batched(n=self.batch_size, batch_fn=None)
        iterations = (
            len(requests) // self.batch_size
            + int(len(requests) % self.batch_size != 0)
        )
        progress = tqdm(
            total=iterations,
            disable=self.rank != 0,
            desc="Model Responding",
        )
        total_elapsed = 0.0
        total_tokens = 0

        for chunk in chunks:
            ctx, doc_to_messages, raw_gen_kwargs, doc_ids, tasks, splits = zip(
                *chunk
            )
            del ctx
            task = tasks[0]
            split = splits[0]
            chat_messages = [
                ChatMessages(
                    **{
                        "messages": doc_to_messages[0](
                            self.task_dict[task][split][doc_id]
                        )
                    }
                )
                for doc_id in doc_ids
            ]
            visuals = []
            videos = []
            for messages in chat_messages:
                visual, video, _ = messages.extract_media()
                visuals.append(visual)
                videos.append(video)
            visuals = self._flatten(visuals) or None
            videos = self._flatten(videos) or None

            messages = chat_messages[0].model_dump()["messages"]
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                images=visuals,
                videos=videos,
                text=text,
                return_tensors="pt",
            ).to(self.device, self.model.dtype)

            generation = dict(raw_gen_kwargs[0])
            generation.pop("until", None)
            generation.setdefault("max_new_tokens", 1024)
            generation.setdefault("temperature", 0)
            generation.setdefault("num_beams", 1)
            do_sample = float(generation.get("temperature") or 0) > 0
            generation["do_sample"] = do_sample
            if not do_sample:
                generation["temperature"] = None

            start = time.time()
            generated = self.model.generate(
                **inputs,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eot_token_id,
                use_cache=self.use_cache,
                **generation,
            )
            elapsed = time.time() - start
            trimmed = [
                output[len(input_ids) :]
                for input_ids, output in zip(inputs.input_ids, generated)
            ]
            answers = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            total_elapsed += elapsed
            total_tokens += sum(len(tokens) for tokens in trimmed)

            for answer, tokens in zip(answers, trimmed):
                results.append(
                    GenerationResult(
                        text=answer,
                        token_counts=TokenCounts(output_tokens=len(tokens)),
                    )
                )
                self.cache_hook.add_partial(
                    "generate_until",
                    (text, generation),
                    answer,
                )
            progress.update(1)

        progress.close()
        log_metrics(
            total_gen_tokens=total_tokens,
            total_elapsed_time=total_elapsed,
            avg_speed=(
                total_tokens / total_elapsed if total_elapsed > 0 else 0.0
            ),
            additional_metrics={"rank": self.rank},
        )
        return reordered.get_original(results)

    def loglikelihood(
        self,
        requests: List[Instance],
    ) -> List[Tuple[float, bool]]:
        del requests
        raise NotImplementedError

    def generate_until_multi_round(self, requests):
        del requests
        raise NotImplementedError


MODEL_REGISTRY_V2.register_manifest(
    ModelManifest(
        model_id="generic_vlm_attnres",
        chat_class_path=(
            "lmms_eval_generic_attnres.GenericVLMHf"
        ),
    ),
    overwrite=True,
)


if __name__ == "__main__":
    import lmms_eval.__main__ as main_module

    main_module.cli_evaluate()
