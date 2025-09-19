# All About Transformers

A curated, research-oriented collection of influential Transformer papers, implementations, and surveys across language, vision, multimodal learning, speech, and decision making. The goal is to track seminal ideas alongside the fast-moving landscape of foundation models and supporting infrastructure.

## Table of Contents
- [Surveys & Foundational Works](#surveys--foundational-works)
- [Efficient & Scalable Architectures](#efficient--scalable-architectures)
- [Large Language Models](#large-language-models)
- [Vision Transformers](#vision-transformers)
  - [Hybrid CNN-Transformer Architectures](#hybrid-cnn-transformer-architectures)
  - [Classification & Representation Learning](#classification--representation-learning)
  - [Detection, Segmentation & 3D Perception](#detection-segmentation--3d-perception)
- [Multimodal & Vision-Language](#multimodal--vision-language)
  - [General-Purpose Foundation Models](#general-purpose-foundation-models)
  - [Retrieval, VQA & Document Understanding](#retrieval-vqa--document-understanding)
- [Speech & Audio](#speech--audio)
- [Reinforcement Learning & Agents](#reinforcement-learning--agents)
- [Libraries & Tooling](#libraries--tooling)

## Surveys & Foundational Works
- **Attention Is All You Need**, NeurIPS 2017 — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin. [[paper]](https://arxiv.org/abs/1706.03762) [[official code]](https://github.com/tensorflow/tensor2tensor) [[PyTorch]](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
- **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**, NAACL 2019 — Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. [[paper]](https://arxiv.org/abs/1810.04805) [[official code]](https://github.com/google-research/bert) [[Transformers]](https://github.com/huggingface/transformers)
- **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)**, JMLR 2020 — Colin Raffel, Noam Shazeer, Adam Roberts, et al. [[paper]](https://jmlr.org/papers/v21/20-074.html) [[official code]](https://github.com/google-research/text-to-text-transfer-transformer)
- **Efficient Transformers: A Survey**, ACM Comput. Surv. 2022 — Yi Tay, Mostafa Dehghani, Dara Bahri, Donald Metzler. [[paper]](https://arxiv.org/abs/2009.06732)
- **A Survey on Vision Transformer Methods**, ACM Comput. Surv. 2023 — Salman Khan, Muzammal Naseer, et al. [[paper]](https://arxiv.org/abs/2101.01169)
- **Transformers in Reinforcement Learning: A Survey**, arXiv 2023 — Haochen Chen, Yihan Du, et al. [[paper]](https://arxiv.org/abs/2301.03044)
- **Transformers in Vision: A Survey**, ACM Comput. Surv. 2024 — Ali Hassani, Steven Walton, et al. [[paper]](https://arxiv.org/abs/2010.08252)

## Efficient & Scalable Architectures
- **Reformer: The Efficient Transformer**, ICLR 2020 — Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya. [[paper]](https://arxiv.org/abs/2001.04451) [[official code]](https://github.com/google/trax/tree/master/trax/models/reformer)
- **Linformer: Self-Attention with Linear Complexity**, NeurIPS 2020 — Sinong Wang, Belinda Li, Madian Khabsa, Han Fang, Hao Ma. [[paper]](https://arxiv.org/abs/2006.04768) [[code]](https://github.com/lucidrains/linformer)
- **Longformer: The Long-Document Transformer**, arXiv 2020 — Iz Beltagy, Matthew E. Peters, Arman Cohan. [[paper]](https://arxiv.org/abs/2004.05150) [[official code]](https://github.com/allenai/longformer)
- **Sparse Transformers**, NeurIPS 2019 — Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever. [[paper]](https://arxiv.org/abs/1904.10509) [[code]](https://github.com/openai/sparse_attention)
- **BigBird: Transformers for Longer Sequences**, NeurIPS 2020 — Manzil Zaheer, Guru Guruganesh, et al. [[paper]](https://arxiv.org/abs/2007.14062) [[official code]](https://github.com/google-research/bigbird)
- **Performer: Transformer with Linear Attention**, ICLR 2021 — Krzysztof Choromanski, Valerii Likhosherstov, et al. [[paper]](https://arxiv.org/abs/2009.14794) [[official code]](https://github.com/google-research/google-research/tree/master/performer)
- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**, NeurIPS 2022 — Tri Dao, Daniel Y. Fu, et al. [[paper]](https://arxiv.org/abs/2205.14135) [[official code]](https://github.com/Dao-AILab/flash-attention)
- **LongNet: Scaling Transformers to 1,000,000,000 Tokens**, ICML 2024 — Shuming Ding, Zihan Liu, et al. [[paper]](https://arxiv.org/abs/2307.02486) [[official code]](https://github.com/microsoft/LongNet)

## Large Language Models
- **Language Models are Few-Shot Learners (GPT-3)**, NeurIPS 2020 — Tom B. Brown, Benjamin Mann, et al. [[paper]](https://arxiv.org/abs/2005.14165)
- **Training Compute-Optimal Large Language Models (Chinchilla)**, arXiv 2022 — Jordan Hoffmann, Sebastian Borgeaud, et al. [[paper]](https://arxiv.org/abs/2203.15556)
- **PaLM: Scaling Language Modeling with Pathways**, arXiv 2022 — Aakanksha Chowdhery, Sharan Narang, et al. [[paper]](https://arxiv.org/abs/2204.02311)
- **PaLM 2 Technical Report**, arXiv 2023 — Rohan Anil, Andrew M. Dai, et al. [[paper]](https://arxiv.org/abs/2305.10403)
- **GPT-4 Technical Report**, 2023 — OpenAI. [[paper]](https://cdn.openai.com/papers/gpt-4.pdf)
- **Scaling Language Models: Methods, Analysis & Insights from Training Gopher**, arXiv 2021 — Jack W. Rae, Sebastian Borgeaud, et al. [[paper]](https://arxiv.org/abs/2112.11446)
- **OPT: Open Pre-trained Transformer Language Models**, arXiv 2022 — Susan Zhang, Stephen Roller, et al. [[paper]](https://arxiv.org/abs/2205.01068) [[official code]](https://github.com/facebookresearch/metaseq)
- **BLOOM: A 176B-Parameter Open-Access Multilingual Language Model**, arXiv 2022 — BigScience Workshop. [[paper]](https://arxiv.org/abs/2211.05100) [[official code]](https://github.com/bigscience-workshop/bloom)
- **LLaMA: Open and Efficient Foundation Language Models**, arXiv 2023 — Hugo Touvron, Thibaut Lavril, et al. [[paper]](https://arxiv.org/abs/2302.13971) [[code]](https://github.com/facebookresearch/llama)
- **Llama 2: Open Foundation and Fine-Tuned Chat Models**, arXiv 2023 — Hugo Touvron, Louis Martin, et al. [[paper]](https://arxiv.org/abs/2307.09288) [[code]](https://github.com/facebookresearch/llama)
- **Mistral 7B**, arXiv 2023 — Albert Q. Jiang, Alexandre Sablayrolles, et al. [[paper]](https://arxiv.org/abs/2310.06825) [[official code]](https://github.com/mistralai/mistral-src)
- **Mixtral of Experts (Mixtral 8x7B)**, arXiv 2024 — Albert Q. Jiang, Alexandre Sablayrolles, et al. [[paper]](https://arxiv.org/abs/2401.04088) [[official code]](https://github.com/mistralai/mistral-src)
- **Gemma: Open Models Based on Gemini Research and Technology**, arXiv 2024 — Alexandre Carlier, Simran Arora, et al. [[paper]](https://arxiv.org/abs/2403.08295) [[code]](https://github.com/google/gemma_pytorch)
- **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model**, arXiv 2024 — DeepSeek-AI. [[paper]](https://arxiv.org/abs/2405.04434) [[official code]](https://github.com/deepseek-ai/DeepSeek-V2)
- **Phi-2: The Surprising Power of Small Language Models**, arXiv 2023 — Xiaodong Liu, Yelong Shen, et al. [[paper]](https://arxiv.org/abs/2309.05461) [[official code]](https://github.com/microsoft/Phi-2)
- **Phi-3 Technical Report**, arXiv 2024 — F. A. Chowdhury, Harkirat Behl, et al. [[paper]](https://arxiv.org/abs/2404.14219) [[official code]](https://github.com/microsoft/Phi-3)
- **Qwen2 Technical Report**, arXiv 2024 — Jian Yang, Ying Wen, et al. [[paper]](https://arxiv.org/abs/2407.10671) [[official code]](https://github.com/QwenLM/Qwen2)

## Vision Transformers

### Hybrid CNN-Transformer Architectures
- **Attention Augmented Convolutional Networks**, ICCV 2019 — Irwan Bello, Barret Zoph, Ashish Vaswani, Jonathon Shlens, Quoc V. Le. [[paper]](https://arxiv.org/abs/1904.09925) [[PyTorch]](https://github.com/leaderj1001/Attention-Augmented-Conv2d)
- **Self-Attention Generative Adversarial Networks (SAGAN)**, ICML 2019 — Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena. [[paper]](https://arxiv.org/abs/1805.08318) [[official code]](https://github.com/brain-research/self-attention-gan)
- **VideoBERT: A Joint Model for Video and Language Representation Learning**, ICCV 2019 — Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, Cordelia Schmid. [[paper]](https://arxiv.org/abs/1904.01766)
- **End-to-End Lane Shape Prediction with Transformers (LSTR)**, arXiv 2020 — Ruijin Liu, Zejian Yuan, Tie Liu, Zhiliang Xiong. [[paper]](https://arxiv.org/abs/2011.04233) [[official code]](https://github.com/liuruijin17/LSTR)

### Classification & Representation Learning
- **Image Transformer**, ICML 2018 — Niki Parmar, Ashish Vaswani, et al. [[paper]](https://arxiv.org/abs/1802.05751) [[official code]](https://github.com/tensorflow/tensor2tensor)
- **Stand-Alone Self-Attention in Vision Models**, NeurIPS 2019 — Prajit Ramachandran, Niki Parmar, et al. [[paper]](https://arxiv.org/abs/1906.05909) [[code]](https://github.com/google-research/google-research/tree/master/standalone_self_attention_in_vision_models)
- **An Image is Worth 16x16 Words: Vision Transformer (ViT)**, ICLR 2021 — Alexey Dosovitskiy, Lucas Beyer, et al. [[paper]](https://arxiv.org/abs/2010.11929) [[PyTorch]](https://github.com/lucidrains/vit-pytorch)
- **Training Data-Efficient Image Transformers (DeiT)**, arXiv 2020 — Hugo Touvron, Matthieu Cord, et al. [[paper]](https://arxiv.org/abs/2012.12877) [[official code]](https://github.com/facebookresearch/deit)
- **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**, ICCV 2021 — Ze Liu, Yutong Lin, et al. [[paper]](https://arxiv.org/abs/2103.14030) [[official code]](https://github.com/microsoft/Swin-Transformer)
- **Masked Autoencoders Are Scalable Vision Learners (MAE)**, CVPR 2022 — Kaiming He, Xinlei Chen, et al. [[paper]](https://arxiv.org/abs/2111.06377) [[official code]](https://github.com/facebookresearch/mae)
- **BEiT: BERT Pre-Training of Image Transformers**, ICLR 2022 — Hangbo Bao, Li Dong, Furu Wei. [[paper]](https://arxiv.org/abs/2106.08254) [[official code]](https://github.com/microsoft/unilm/tree/master/beit)
- **CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification**, ICCV 2021 — Chun-Fu Richard Chen, Quanfu Fan, Rameswar Panda. [[paper]](https://arxiv.org/abs/2103.14899) [[official code]](https://github.com/IBM/CrossViT)
- **DINO: Emerging Properties in Self-Supervised Vision Transformers**, ICCV 2021 — Mathilde Caron, Hugo Touvron, et al. [[paper]](https://arxiv.org/abs/2104.14294) [[official code]](https://github.com/facebookresearch/dino)
- **EVA-CLIP: Improved Training Techniques for CLIP at Scale**, arXiv 2023 — Yunzhu Li, Xiuyu Sun, et al. [[paper]](https://arxiv.org/abs/2303.15389) [[official code]](https://github.com/baaivision/EVA/tree/master/EVA-CLIP)
- **EVA-02: A Visual Representation for Any-scale Recognition**, arXiv 2023 — Kunchang Li, Yutong Lin, et al. [[paper]](https://arxiv.org/abs/2308.01390) [[official code]](https://github.com/baaivision/EVA)

### Detection, Segmentation & 3D Perception
- **DETR: End-to-End Object Detection with Transformers**, ECCV 2020 — Nicolas Carion, Francisco Massa, et al. [[paper]](https://arxiv.org/abs/2005.12872) [[official code]](https://github.com/facebookresearch/detr) [[Detectron2]](https://github.com/poodarchu/DETR.detectron2)
- **Deformable DETR**, ICLR 2021 — Xizhou Zhu, Weijie Su, et al. [[paper]](https://arxiv.org/abs/2010.04159) [[official code]](https://github.com/fundamentalvision/Deformable-DETR)
- **UP-DETR: Unsupervised Pre-training for Object Detection with Transformers**, CVPR 2021 — Zhigang Dai, Bolun Cai, et al. [[paper]](https://arxiv.org/abs/2011.09094)
- **Mask2Former: Masked-Attention Mask Transformer for Universal Image Segmentation**, CVPR 2022 — Bowen Cheng, Ishan Misra, Alexander Kirillov, et al. [[paper]](https://arxiv.org/abs/2112.01527) [[official code]](https://github.com/facebookresearch/Mask2Former)
- **Segment Anything**, ICCV 2023 — Alexander Kirillov, Eric Mintun, et al. [[paper]](https://arxiv.org/abs/2304.02643) [[official code]](https://github.com/facebookresearch/segment-anything)
- **DINOv2: Learning Robust Visual Features without Supervision**, arXiv 2023 — Mathieu Caron, Hugo Touvron, et al. [[paper]](https://arxiv.org/abs/2304.07193) [[official code]](https://github.com/facebookresearch/dinov2)
- **Point Transformer**, ICCV 2021 — Hengshuang Zhao, Li Jiang, et al. [[paper]](https://arxiv.org/abs/2011.00931)

## Multimodal & Vision-Language

### General-Purpose Foundation Models
- **ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations**, NeurIPS 2019 — Jiasen Lu, Dhruv Batra, Devi Parikh, Stefan Lee. [[paper]](https://arxiv.org/abs/1908.02265) [[official code]](https://github.com/facebookresearch/vilbert-multi-task)
- **LXMERT: Learning Cross-Modality Encoder Representations**, EMNLP 2019 — Hao Tan, Mohit Bansal. [[paper]](https://arxiv.org/abs/1908.07490) [[official code]](https://github.com/airsplay/lxmert)
- **VisualBERT: A Simple and Performant Baseline for Vision and Language**, arXiv 2019 — Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, Kai-Wei Chang. [[paper]](https://arxiv.org/abs/1908.03557) [[official code]](https://github.com/uclanlp/visualbert)
- **CLIP: Learning Transferable Visual Models From Natural Language Supervision**, ICML 2021 — Alec Radford, Jong Wook Kim, et al. [[paper]](https://arxiv.org/abs/2103.00020) [[official code]](https://github.com/openai/CLIP)
- **Flamingo: A Visual Language Model for Few-Shot Learning**, NeurIPS 2022 — Jean-Baptiste Alayrac, Jeff Donahue, et al. [[paper]](https://arxiv.org/abs/2204.14198)
- **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models**, ICML 2023 — Junnan Li, Dongxu Li, et al. [[paper]](https://arxiv.org/abs/2301.12597) [[official code]](https://github.com/salesforce/LAVIS)
- **LLaVA: Large Language and Vision Assistant**, ICCV 2023 — Haotian Liu, Chunyuan Li, et al. [[paper]](https://arxiv.org/abs/2304.08485) [[official code]](https://github.com/haotian-liu/LLaVA)
- **Kosmos-1: Multimodal Large Language Models**, arXiv 2023 — Pengchuan Zhang, Xiujun Li, et al. [[paper]](https://arxiv.org/abs/2302.14045)
- **MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models**, arXiv 2023 — Deyao Zhu, Jun Chen, et al. [[paper]](https://arxiv.org/abs/2304.10592) [[official code]](https://github.com/Vision-CAIR/MiniGPT-4)
- **IdeFICS: Open Foundation Models for Multimodal Vision-Language Understanding**, arXiv 2023 — Hugging Face M4 Team. [[paper]](https://arxiv.org/abs/2306.08195) [[official code]](https://github.com/huggingface/idefics)

### Retrieval, VQA & Document Understanding
- **UNITER: Universal Image-Text Representation Learning**, ECCV 2020 — Yen-Chun Chen, Linjie Li, et al. [[paper]](https://arxiv.org/abs/1909.11740) [[official code]](https://github.com/ChenRocks/UNITER)
- **Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks**, ECCV 2020 — Xiujun Li, Xi Yin, et al. [[paper]](https://arxiv.org/abs/2004.06165) [[official code]](https://github.com/microsoft/Oscar)
- **LayoutLM: Pre-training of Text and Layout for Document Image Understanding**, KDD 2020 — Yiheng Xu, Minghao Li, et al. [[paper]](https://arxiv.org/abs/1912.13318) [[official code]](https://github.com/microsoft/unilm/tree/master/layoutlm)
- **ImageBERT: Cross-modal Pre-training with Large-scale Weak-supervised Image-Text Data**, arXiv 2020 — Di Qi, Lin Su, et al. [[paper]](https://arxiv.org/abs/2001.07966)
- **12-in-1: Multi-Task Vision and Language Representation Learning**, CVPR 2020 — Jiasen Lu, Vedanuj Goswami, Marcus Rohrbach, Devi Parikh, Stefan Lee. [[paper]](https://arxiv.org/abs/1912.02315) [[official code]](https://github.com/facebookresearch/vilbert-multi-task)
- **Kosmos-2: Grounding Multimodal LLMs to the World**, arXiv 2023 — Yujing Wang, Chenguang Zhu, et al. [[paper]](https://arxiv.org/abs/2306.14824)
- **Donut: Document Understanding Transformer without OCR**, ECCV 2022 — Geewook Kim, Teakgyu Hong, et al. [[paper]](https://arxiv.org/abs/2111.15664) [[official code]](https://github.com/clovaai/donut)
- **Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond**, arXiv 2023 — Junjie Bai, Chang Zhou, et al. [[paper]](https://arxiv.org/abs/2308.12966) [[official code]](https://github.com/QwenLM/Qwen-VL)

## Speech & Audio
- **wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations**, NeurIPS 2020 — Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, Michael Auli. [[paper]](https://arxiv.org/abs/2006.11477) [[official code]](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec)
- **HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction**, ICASSP 2021 — Wei-Ning Hsu, Benjamin Bolte, et al. [[paper]](https://arxiv.org/abs/2106.07447) [[official code]](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert)
- **Whisper: Robust Speech Recognition via Large-Scale Weak Supervision**, arXiv 2022 — Alec Radford, Jong Wook Kim, et al. [[paper]](https://arxiv.org/abs/2212.04356) [[official code]](https://github.com/openai/whisper)
- **AudioLM: A Language Modeling Approach to Audio Generation**, NeurIPS 2022 — Zalán Borsos, Raphaël Marion, et al. [[paper]](https://arxiv.org/abs/2209.03143)
- **MusicLM: Generating Music From Text**, arXiv 2023 — Andrea Agostinelli, Timo I. Denk, et al. [[paper]](https://arxiv.org/abs/2301.11325) [[official code]](https://github.com/google-research/google-research/tree/master/musiclm)
- **SpeechT5: Unified-Modal Encoder-Decoder Pre-training for Spoken Language Processing**, ACL 2022 — Xu Tan, Xu Li, et al. [[paper]](https://arxiv.org/abs/2110.07205) [[official code]](https://github.com/microsoft/SpeechT5)
- **SeamlessM4T: Massively Multilingual & Multimodal Machine Translation**, arXiv 2023 — Juan Pino, Changhan Wang, et al. [[paper]](https://arxiv.org/abs/2308.11596) [[official code]](https://github.com/facebookresearch/seamless_communication)

## Reinforcement Learning & Agents
- **The Transformer Reinforcement Learning Framework (Decision Transformer)**, NeurIPS 2021 — Lili Chen, Kevin Lu, et al. [[paper]](https://arxiv.org/abs/2106.01345) [[official code]](https://github.com/kzl/decision-transformer)
- **Trajectory Transformer: Off-Policy Reinforcement Learning with Sequence Modeling**, NeurIPS 2021 — Michael Janner, Qiyang Li, Sergey Levine. [[paper]](https://arxiv.org/abs/2106.02039) [[official code]](https://github.com/JannerM/trajectory-transformer)
- **Generalist Agent (Gato)**, Science 2022 — Scott Reed, Konrad Zolna, et al. [[paper]](https://arxiv.org/abs/2205.06175)
- **RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control**, arXiv 2023 — Anthony Brohan, Noah Brown, et al. [[paper]](https://arxiv.org/abs/2307.15818)
- **OpenVLA: An Open-Source Vision-Language-Action Model**, arXiv 2024 — Kevin Zakka, Dieter Fox, et al. [[paper]](https://arxiv.org/abs/2406.09246) [[official code]](https://github.com/openvla/openvla)
- **Transformer-based World Models Are Happy with 100k Interactions**, NeurIPS 2023 — Yan Duan, Murtaza Dalal, et al. [[paper]](https://arxiv.org/abs/2306.10612)
- **SayCan: Grounding Large Language Models for Robotics**, arXiv 2022 — Michael Ahn, Anthony Brohan, et al. [[paper]](https://arxiv.org/abs/2204.01691)
- **Toolformer: Language Models Can Teach Themselves to Use Tools**, NeurIPS 2023 — Timo Schick, Jane Dwivedi-Yu, et al. [[paper]](https://arxiv.org/abs/2302.04761)

## Libraries & Tooling
- **Hugging Face Transformers**, actively maintained — Thomas Wolf, Lysandre Debut, et al. [[code]](https://github.com/huggingface/transformers) [[docs]](https://huggingface.co/docs/transformers/index)
- **xFormers: A modular and hackable Transformer library**, Meta AI 2022. [[code]](https://github.com/facebookresearch/xformers)
- **Open-source FlashAttention implementations**, 2022 — Dao-AI Lab. [[code]](https://github.com/Dao-AILab/flash-attention)
- **NVIDIA TensorRT-LLM**, 2023 — NVIDIA. [[code]](https://github.com/NVIDIA/TensorRT-LLM)
- **TransformerLens: Interpretability tools for Transformer language models**, 2023 — Neel Nanda, et al. [[code]](https://github.com/NeelNanda-IO/TransformerLens)
- **vLLM: Easy, Fast, and Cheap LLM Serving**, 2023 — Tianqi Chen, Yiming Wang, et al. [[paper]](https://arxiv.org/abs/2307.10928) [[official code]](https://github.com/vllm-project/vllm)
- **lm-evaluation-harness: A Framework for Evaluating Language Models**, 2022 — EleutherAI. [[code]](https://github.com/EleutherAI/lm-evaluation-harness)
- **LightLLM: A High-Performance Distributed LLM Serving Framework**, 2024 — ModelScope Contributors. [[code]](https://github.com/ModelTC/lightllm)
