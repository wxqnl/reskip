## Discussion
AR-RETROFIT installs an intrinsic depth-routing signal into a frozen pretrained transformer through a short fine-tune;
the resulting routing weights drive RESKIP as a natural application. At the canonical L=4 v3 recipe, the retrofit improves
5/6 VLM benchmarks on Qwen3-VL-2B/4B at iso-cost compiled inference, and lifts LAMBADA by +3.3/+8.7pp
with the largest gain concentrated on MMStar math (consistent with a router that lifts deliberate-reasoning paths). The
installed signal is usable: at 340M from-scratch, RESKIP reaches 1.19× wall-clock at zero benchmark drop, and a
rate-matched comparison at ∼12% skip rate shows the input-dependent rule strictly beats static and random schedules.
A LIBERO VLA warm-started from the retrofit improves the matched pure-OFT baseline at both 2B and 4B; RESKIP
on the action stream, with thresholds re-calibrated on the action distribution, further improves the trained policy at the
9

conservative operating point.
Scope of claims.
The result should be read as evidence that pretrained transformers can acquire a usable depth-
routing pathway through lightweight retrofitting, not that the resulting RESKIP policy is universally optimal or that
low ATTNRES weight alone certifies safe layer removal. In all large-model and VLA experiments, skip eligibility is
calibrated by per-block ablation or action-drift, and thresholds are calibrated on the decoded modality. We separate
compiled full-depth inference from eager dynamic skipping, since current CUDA graph capture cannot accommodate the
dynamic branch; the iso-cost result requires the former and the dynamic-skip savings require the latter, and combining
them is open systems work. The retrofit is demonstrated on one VLM family at two scales; transfer to other backbones
is open.
## References
[1] Shuai Bai et al. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025.
[2] Andre M. Bastos, W. Martin Usrey, Rick A. Adams, George R. Mangun, Pascal Fries, and Karl J. Friston.
Canonical microcircuits for predictive coding. Neuron, 76(4):695–711, 2012. doi: 10.1016/j.neuron.2012.10.038.
[3] Kevin Black, Noah Brown, Danny Driess, et al. π0: A vision-language-action flow model for general robot control.
arXiv:2410.24164, 2024.
[4] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, et al. RT-2:
Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818,
2023.
[5] Timothy J. Buschman and Earl K. Miller. Top-down versus bottom-up control of attention in the prefrontal and
posterior parietal cortices. Science, 315(5820):1860–1862, 2007. doi: 10.1126/science.1138071.
[6] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao,
Dahua Lin, and Feng Zhao. Are we on the right way for evaluating large vision-language models? In Neural
Information Processing Systems (NeurIPS), 2024. MMStar benchmark; arXiv:2403.20330.
[7] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord.
Think you have solved question answering? Try ARC, the ai2 reasoning challenge. arXiv:1803.05457, 2018.
[8] DeepSeek-AI. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning, 2025.
[9] Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. Universal transformers.
In ICLR, 2019.
[10] James J. DiCarlo, Davide Zoccolan, and Nicole C. Rust. How does the brain solve visual object recognition?
Neuron, 73(3):415–434, 2012. doi: 10.1016/j.neuron.2012.01.010.
[11] Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen
Zhou. Enhancing chat language models by scaling high-quality instructional conversations. arXiv preprint
arXiv:2305.14233, 2023.
[12] Maha Elbayad, Jiatao Gu, Edouard Grave, and Michael Auli. Depth-adaptive transformer. In ICLR, 2020.
[13] Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai, Anas
Mahmoud, Bilge Acun, Saurabh Agarwal, Ahmed Roman, Ahmed Aly, Beidi Chen, and Carole-Jean Wu.
LayerSkip: Enabling early exit inference and self-speculative decoding. In Proceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12622–12642. Association
for Computational Linguistics, 2024. doi: 10.18653/v1/2024.acl-long.681. URL https://aclanthology.
org/2024.acl-long.681/.
10

[14] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, et al. A framework for few-shot language model