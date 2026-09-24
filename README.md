# 1.5B-MLLM
基于 **Qwen3-0.6B** 构建的轻量级多模态大模型（总参数量约 **1.5B**）。本项目从零实现了图像、音频、视频到文本的特征对齐，配套完整的训练脚本、KV Cache 加速推理，以及 **FastAPI + WebUI** 的全栈部署方案。

> ⚠️ **重要提示：本仓库不包含训练好的投影层权重。**
> 必须先按 [快速开始](#-快速开始quick-start) 的第 4 步训练，否则推理会加载随机初始化的投影层、输出无意义文本（这是预期行为，不是 bug）。

---

## ✨ 项目亮点（Project Highlights）

- 🎯 **极简高效的投影层设计**：摒弃复杂的 Q-Former 架构，采用 `2层MLP + GELU + LayerNorm (Pre-Norm)` 的投影层。理论依据来源于 LLaVA-1.5 及相关多模态对齐研究，能有效对齐不同模态的数值尺度（Norm Discrepancy），在小规模模型上实现极高的训练效率。

- 🧠 **四模态统一融合**：原生支持文本、图像、音频、视频四种模态。通过占位符替换（Placeholder Replacement）技术，将多模态特征无缝嵌入到 Qwen3 的语言空间。

- 🎬 **轻量视频处理方案**：无需额外加载重型视频编码器。复用 SigLIP 逐帧编码 + 时间维度 Mean Pooling，在显著降低显存占用的同时，保持与单张图像一致的 Token 数量（**729 tokens**），极大简化了序列拼接逻辑。

- ⚡ **冻结训练策略**：训练时冻结全部编码器（SigLIP / Whisper）与 LLM（Qwen3），**仅训练投影层**（可训练参数仅 **3.81 M，占总量 0.25%**）。整个训练可在消费级显卡上完成（作者在 **RTX 4060 Laptop 8GB** 上实测跑通）。

- 🚀 **支持 KV Cache 的增量解码**：推理阶段手写生成循环，首步处理完整序列与多模态特征，后续步仅传入单个 Token 并复用 `past_key_values`，避免重复计算历史 Token。

- 💻 **开箱即用的全栈部署**：包含 FastAPI 后端服务与现代化前端界面（多模态文件上传、对话置顶、本地历史记录、深色主题）。

---

🏗️ 系统架构（Architecture）

```text
                                  ┌──────────────────────┐
                                  │      Qwen3-0.6B      │
                                  │      (Frozen)        │
                                  └───────────▲──────────┘
                                              │
                       Text Embeddings + Multimodal Embeddings
                                              │
        ┌──────────────────────┬──────────────┴──────────────┐
        │                      │                             │
┌───────┴────────┐    ┌────────┴─────────┐          ┌────────┴────────┐
│  Image Input   │    │  Video Input     │          │   Audio Input   │
│ SigLIP so400m  │    │ SigLIP + Mean    │          │  Whisper-base   │
│  27×27 = 729   │    │ Pooling → 729    │          │  → 1500 frames  │
└───────┬────────┘    └────────┬─────────┘          └────────┬────────┘
        │                      │                             │
┌───────▼──────────────────────▼─────────┐       ┌───────────▼─────────┐
│        Vision Projector                │       │   Audio Projector   │
│  LayerNorm + Linear + GELU + Linear    │       │ (同结构，输入 512)   │
│        1152 → 1024                     │       │      512 → 1024     │
└───────────────────┬────────────────────┘       └───────────┬─────────┘
                    │                                        │
                    └──────────────┬─────────────────────────┘
                                   │
                    Replace <image> / <video> / <audio>
                        placeholder embeddings
                                   │
                                   ▼
                        送入 Qwen3 计算 logits
                       
关键设计：视频与图像共用同一个 VisionProjector，不引入额外参数；三种模态的占位符 token 全部取自 Qwen3 词表自带的特殊 token，不需要扩展词表（无需 resize_token_embeddings）。

```text 

📂 项目结构（Project Structure）

```text
1B/
├── mllm/                          # 核心代码包
│   ├── __init__.py
│   ├── projectors.py              # 投影层（Vision / Audio）
│   ├── encoders.py                # 编码器封装（SigLIP / Whisper / Video）
│   ├── modeling_mllm.py           # 总模型（占位符替换 + Qwen3 前向）
│   ├── dataset.py                 # 多模态数据集 + collate_fn
│   ├── utils.py                   # 占位符分配、断点保存/加载、断点续训状态
│   ├── video_io.py                # 视频解码兼容层（torchvision → PyAV）
│   └── audio_io.py                # 音频解码兼容层（librosa → soundfile → wave）
├── models/                        # 本地模型权重（需自行下载，见下）
│   ├── Qwen3-0.6B/
│   ├── siglip-so400m-patch14-384/
│   └── whisper-base/
├── scripts/
│   ├── train.py                   # 投影层训练（AMP + 梯度累积 + 断点续训）
│   └── inference.py               # 命令行推理（KV Cache，支持交互模式）
├── tools/
│   ├── prepare_data.py            # 数据集下载转换 → JSONL
│   └── check_data.py              # 训练前数据体检
├── data/
│   ├── train_full.jsonl           # 训练索引（合并后的）
│   ├── images/  audios/  videos/  # 媒体文件
│   └── train_coco.jsonl / train_audiocaps.jsonl / train_msrvtt.jsonl
├── checkpoints/                   # 训练好的投影层权重（*.pt，约 44 MB/个）
├── main.py                        # 架构预检脚本
├── server.py                      # FastAPI 后端
├── index.html                     # 前端聊天界面
├── requirements.txt
└── .gitignore
```text 

🛠️ 快速开始（Quick Start）
1. 环境准备
bash
复制
git clone https://github.com/your-username/1.5B-MLLM.git
cd 1.5B-MLLM
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
注意：torch 必须安装 CUDA 版本，否则无法使用 GPU：

bash
复制
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

2. 准备模型权重
请手动下载以下三个模型，分别放入 models/ 下的同名目录：
模型	说明	大小
Qwen3-0.6B	语言模型底座（冻结）	~1.4 GB
SigLIP-so400m-patch14-384	视觉编码器（仅用视觉塔，冻结）	~3.3 GB
Whisper-base	音频编码器（仅用 encoder，冻结）	~0.3 GB
国内网络可用镜像下载：

bash
复制
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1     # 国内访问 Xet 存储会 401，务必加上

hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B
hf download google/siglip-so400m-patch14-384 --local-dir models/siglip-so400m-patch14-384
hf download openai/whisper-base --local-dir models/whisper-base
下载完成后先跑一次预检，确认三个塔都能加载：

bash
复制
python main.py

3. 准备数据集
 bash
复制
# 转换公开数据集为项目所需的 JSONL 格式
python tools/prepare_data.py coco        # COCO 2014   → data/train_coco.jsonl
python tools/prepare_data.py audiocaps   # AudioCaps   → data/train_audiocaps.jsonl
python tools/prepare_data.py msrvtt      # MSR-VTT     → data/train_msrvtt.jsonl
python tools/prepare_data.py merge       # 合并三者     → data/train_full.jsonl

# 训练前体检（检查媒体文件是否齐全、占位符数量是否匹配）
python tools/check_data.py data/train_full.jsonl  
数据格式说明（JSONL，每行一个 JSON 对象）：

jsonl
复制
{"text": "<image> A cat sitting on a windowsill", "image_path": "000000123456.jpg"}
{"text": "<audio> A dog barking loudly", "audio_path": "audiocaps_100011.wav"}
{"text": "<video> A person is running in the park", "video_path": "msrvtt_video9770.mp4"}
{"text": "用一句话介绍杭州。"}
字段	必填	说明
text	✅	文本内容，用 <image> / <audio> / <video> 标记媒体插入位置
image_path	可选	相对 --image_root 的路径
audio_path	可选	相对 --audio_root 的路径
video_path	可选	相对 --video_root 的路径

4. 训练投影层
   bash
复制
# 8GB 显存（如 RTX 4060 Laptop）推荐配置
python scripts/train.py \
    --train_data_path data/train_full.jsonl \
    --batch_size 1 --grad_accum_steps 4 \
    --num_epochs 3 --max_length 2048

# 16GB 显存（如 RTX 4080）可以适当加大
python scripts/train.py --batch_size 2 --grad_accum_steps 8 --num_epochs 3
训练过程中只有 VisionProjector 和 AudioProjector 的参数会被更新，其余全部冻结。日志会打印可训练参数量供核对：

text
复制
Trainable parameters: 3,808,514 (0.3460%)
[AMP] autocast dtype=torch.bfloat16 | GradScaler=关闭(bf16 不需要)
[Epoch 1] step=100 loss=3.2286 lr=1.000000e-03 elapsed=00:05:25
断点续训（优化器、调度器、学习率进度都会恢复）：

bash
复制
python scripts/train.py --resume_from checkpoints/projector_step_1500.pt --num_epochs 6
   

5. 命令行推理测试
bash
复制
# 图像推理
python scripts/inference.py --prompt "描述这张图片" --image ./data/images/000000123456.jpg

# 音频推理
python scripts/inference.py --prompt "转写这段音频" --audio ./data/audios/audiocaps_100011.wav

# 视频推理
python scripts/inference.py --prompt "描述这个视频" --video ./data/videos/msrvtt_video9770.mp4

# 交互模式（支持 /image /audio /video /clear 命令）
python scripts/inference.py --interactive
 
6. 启动 Web 应用
bash
复制

# 启动 FastAPI 后端（默认端口 8000）
python server.py
浏览器访问 http://127.0.0.1:8000，即可体验支持多模态文件上传的聊天界面。

其他端点：

GET /api/health — 服务与模型状态
POST /api/chat — 对话接口（multipart/form-data，支持 prompt / image / audio / video / history）


# 🔬 技术细节（Technical Details）

占位符替换机制
文本中的 <image> / <audio> / <video> 会在 tokenize 之前被替换为 Qwen3 词表自带的特殊 token：

模态	占位符 Token	Token ID	展开后数量
图像	<|image_pad|>	151655	729
音频	<|vision_pad|>	151654	1500
视频	<|video_pad|>	151656	729
在 forward 首步，模型提取这些 Token 的位置索引，将其 Embedding 直接替换为经过投影层映射后的多模态特征：

python
复制
vision_feats  = self.vision_encoder(pixel_values)      # [B, 729, 1152]
vision_embeds = self.vision_projector(vision_feats)    # [B, 729, 1024]
image_mask    = (input_ids == self.config.image_token_id)
inputs_embeds[image_mask] = vision_embeds.reshape(-1, 1024)
使用词表自带的特殊 token 而非新增 token，意味着无需扩展词表、无需训练新的 embedding。

LayerNorm 的必要性
投影层前加入 LayerNorm，是因为预训练编码器（如 SigLIP）输出的特征范数往往远高于 LLM 的词嵌入。对齐数值尺度能有效防止模型产生"表征惯性"，提升多模态融合效果。

实测数据：投影输出标准差 0.026，与 Qwen3 词嵌入的 0.029 基本一致。

视频时间池化
视频帧经 SigLIP 编码后形状为 [B, N_frames, 729, H]，通过 mean(dim=1) 压缩为 [B, 729, H]。这使视频特征可以直接复用视觉投影层，且占位符数量与单张图像完全一致（729 个），极大简化了序列拼接逻辑。

精度与显存
默认使用 bfloat16（比 fp16 动态范围更大，AMP 下梯度不易溢出）；仅当显卡不支持 bf16 时回退 fp16。
训练投影层必须保持 fp32：torch.amp.GradScaler 不允许 unscale fp16 梯度。
编码器、投影层、LLM 三者之间的 dtype/device 会在 forward 中自动对齐。

# 📊 实测数据（Tested on RTX 4060 Laptop 8GB）

项目	实测值
总参数量	~1.5 B（Qwen3 596M + SigLIP 视觉塔 878M + Whisper 21M + 投影层 3.81M）
可训练参数量	3,808,514（0.25%）
投影层 checkpoint	44.6 MB
推理峰值显存	2.06 GB（三塔同时加载）
训练显存	batch_size=1 / max_length=2048 可用，余量约 2 GB
训练速度	~1.3 秒/样本（bf16, batch 1, 2048 上下文）
训练环境	Python 3.14.6 / torch 2.14.0+cu126 / transformers 5.17.0

# ⚠️ 已知限制（Known Limitations）

本项目经过完整的端到端验证（训练 + 断点续训 + 四模态推理 + Web 服务），但仍有以下限制，使用前请务必了解：

不提供预训练权重。投影层效果完全取决于你的训练数据量与步数。小数据集（几百条）只能看到 loss 下降，无法获得可用效果。
底座能力上限：LLM 仅 0.6B，知识与推理深度受底座限制，不适合复杂推理任务。
上下文长度 2048：单张图像占 729 token、30 秒音频占 1500 token，因此图像 + 音频同时出现会超出上限（代码会自动丢弃被截断的模态）。如需同时输入，请把 --max_length 提到 4096。
仅支持单图：一条样本最多一张图像、一段音频、一个视频（图像与视频互斥，二者都走视觉塔）。
音频上限约 30 秒：Whisper 会截断更长音频。
未做 prompt/response 分段 mask：当前 loss 覆盖整段文本（仅 mask 占位符）。若需要标准的指令微调，需额外实现。
Unsupported：图像生成、语音合成。本模型只做"多模态理解 → 输出文本"，没有任何生成模块（无 VAE / diffusion / TTS）。
无 SFT / RLHF：模型只是"续写"，指令遵循能力不可靠。

# 📄 开源协议
本项目代码采用 MIT License 开源。

⚠️ 请注意：仓库不包含任何预训练模型权重。Qwen3、SigLIP、Whisper 以及使用的公开数据集（COCO、AudioCaps、MSR-VTT 等）均遵循各自原有的许可协议，使用时请自行确认合规性。


# 🙏 致谢
本项目站在以下开源工作的肩膀上：

LLaVA — 投影层对齐范式
Qwen3 — 语言模型底座
SigLIP — 视觉编码器
Whisper — 音频编码器

如果这个项目对你有帮助，欢迎点一个 ⭐ Star！


