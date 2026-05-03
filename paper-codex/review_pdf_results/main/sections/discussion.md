## Discussion
AR-RETROFIT is the main claim: a standard pretrained transformer can be given an intrinsic depth-routing pathway
through a short, identity-preserving fine-tune rather than from-scratch pretraining. At the canonical L=4 v3 recipe it
improves 5/6 VLM benchmarks at both Qwen3-VL scales, lifts LAMBADA by +3.3/+8.7pp, and concentrates the
largest gain on MMStar math. RESKIP and LIBERO then test the usefulness of the installed signal: from-scratch
ATTNRES reaches 1.19× wall-clock at zero benchmark drop, the retrofitted VLM supports conservative text/action
skipping after calibration, and the VLA warm-start improves the matched OFT operating point. The boundary is
calibration. We do not claim that the current RESKIP policy is universally optimal, nor that depth attention alone
certifies safe removal; eligible sets must be selected by ablation or action-drift and thresholds must be calibrated on the
decoded modality. We also separate compiled full-depth inference from eager dynamic skipping, since current CUDA
graph capture does not support the dynamic branch. The central result is that a pretrained model already capable of
token-axis test-time scaling can acquire a second, depth-axis allocation mechanism without retraining from scratch.
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
9

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
[14] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, et al. A framework for few-shot language model