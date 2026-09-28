# 🚀 1.5B-MLLM：一个能跑在 8GB 显卡上的轻量多模态大模型

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![UI](https://img.shields.io/badge/UI-tkinter%20native-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Params](https://img.shields.io/badge/Total%20Params-1.5B-orange)
![Trainable](https://img.shields.io/badge/Trainable-3.81M%20(0.25%25)-brightgreen)
![VRAM](https://img.shields.io/badge/VRAM-8GB%20(consumer)-success)

基于 **Qwen3-0.6B** 构建的多模态大模型，支持**图像 / 音频 / 视频 → 文本**。
从零实现了特征对齐、投影层训练、KV Cache 加速推理，并附带一个**开箱即用的原生桌面程序**
（tkinter UI，不内嵌浏览器、不启动 HTTP 服务、不占用端口）。

> **已在 RTX 4060 Laptop (8GB) 上完成完整训练**：
> 31,450 条多模态样本 × 3 epochs × 38 小时，仅训练 **3,808,514** 个参数（占总量 0.25%）。

<p align="center">
  <img src="loss_curve.png" alt="训练 loss 曲线" width="720">
</p>

---

## 📋 目录

- [它能做什么 / 不能做什么](#它能做什么--不能做什么)
- [快速开始](#快速开始)
- [实测效果](#实测效果)
- [系统架构](#系统架构)
- [项目结构](#项目结构)
- [完整训练流程](#完整训练流程)
- [技术细节](#技术细节)
- [已知问题与修复记录](#已知问题与修复记录)
- [已知限制](#已知限制)
- [常见问题](#常见问题)
- [未来计划](#未来计划)

---

## 它能做什么 / 不能做什么

🎯

**先说清楚，避免误解：** 这是一个**多模态理解**模型，不是一个通用聊天助手。

| ✅ 做得好的 | ❌ 做不好的 |
|---|---|
| 图像描述（物体、颜色、材质、场景） | 中文纯文本问答（底座只有 0.6B，且训练数据无中文） |
| 音频事件识别与描述 | 复杂推理、多步计算、代码生成 |
| 视频内容概括 | 图像生成 / 语音合成（本模型只输出文本） |
| 英文模态描述 | 中文多模态（训练数据为英文） |

> 💡 **评估建议：用图片提问来评估本模型。**
> 图像描述是它的强项；中文纯文本问答受底座 0.6B 限制，不要用它判断程序是否正常。

---

## 快速开始

⚡

### 1. 环境准备

```bash
git clone https://github.com/2793324341/1.5B-MLLM.git
cd 1.5B-MLLM

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements-min.txt
```

> ⚠️ **torch 必须装 CUDA 版**，否则会退回 CPU 推理（单张图可能几十秒）：
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
> ```

### 2. 下载预训练权重（约 5.2 GB）

```bash
python tools/download_models.py
```

脚本会下载到 `models/` 下的同名目录，默认走 `hf-mirror.com` 镜像（国内可用），
失败时自动回退 ModelScope。也可手动下载放进 `models/<同名目录>/`。

| 模型 | 用途 | 大小 |
|---|---|---|
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 语言模型底座（**冻结**） | ~1.4 GB |
| [SigLIP-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) | 视觉编码器（**冻结**） | ~3.3 GB |
| [Whisper-base](https://huggingface.co/openai/whisper-base) | 音频编码器（**冻结**） | ~0.3 GB |

架构预检：

```bash
python main.py
```

### 3. 准备投影层权重

**仓库不含训练好的投影层权重**（43.6 MB/个）。二选一：

**方式 A：自己训练（推荐，见[完整训练流程](#完整训练流程)）**

```bash
# 零依赖演示数据，60 秒跑通「数据 → 训练 → 推理」
python tools/make_demo_data.py
python scripts/train.py --train_data_path data/demo/train.jsonl --num_epochs 1
```

**方式 B：从 Releases 下载作者训练好的权重**（如果提供了的话）

放到 `checkpoints/projector_epoch_3.pt` 即可。

> ⚠️ **不训练的话推理输出会是无意义文本。** 投影层是随机初始化的，必须训练。

### 4. 启动桌面程序

**Windows：双击 `start_desktop.bat`**（或中文名的 `启动桌面版.bat`）

**命令行（跨平台）：**

```bash
python desktop_app.py
```

程序会在**自己的进程里**加载模型（约 12~15 秒），界面顶部状态栏从"正在加载模型…"
变成"就绪"后即可提问。

**怎么用：**

| 操作 | 说明 |
|---|---|
| 输入文字 → `Enter` | 发送（`Shift+Enter` 换行） |
| 先点「＋图片」再提问 | 图文问答，图片会显示成缩略图 |
| 「＋音频」/「＋视频」 | 音频理解 / 视频理解（用法同上） |
| 「停止生成」 | 生成过程中随时中断 |
| 「文件 → 新对话」 | 清空上下文（`Ctrl+N`） |

> 不需要手写 `<image>` 这类占位符，程序会按你附加的文件自动补上。

**命令行推理（无需图形界面）：**

```bash
# 图像
python scripts/inference.py --prompt "Describe this image" --image ./data/images/xxx.jpg

# 音频
python scripts/inference.py --prompt "What sound is this" --audio ./data/audios/xxx.wav

# 视频
python scripts/inference.py --prompt "Describe this video" --video ./data/videos/xxx.mp4

# 交互模式
python scripts/inference.py --interactive
```

---

## 实测效果

📊

以下均为训练完成后用 `projector_epoch_3.pt` **实际跑出的结果**，非示意。

### ① 图像理解 ✅ 强项

| 输入图片 | 模型输出 |
|---|---|
| 狗 + 自行车 | *"The image shows a dog laying on a sidewalk, surrounded by a row of buildings and a bicycle."* |
| 两只长颈鹿 | *"of two giraffes in a zoo. They are both standing next to each other. One is eating something from the top of the cage... There are people watching."* |

能准确识别**物体类别、场景、人员活动**。第二例中"有人围观"这个细节与图片完全吻合。

### ② 音频理解 ✅

| 输入音频 | 模型输出 |
|---|---|
| AudioCaps 样本 | *"...something whirs and hums followed by a loud engine..."* |
| AudioCaps 样本 | *"a man laughing and then another man laughing again as he speaks loudly... a loud splash"* |

第二条几乎完全命中（笑声、说话、splash 全部识别）。

### ③ 视频理解 ✅

| 输入视频 | 模型输出 |
|---|---|
| MSR-VTT 样本 | *"A woman is making a cake with little flowers and the whole cake is very pretty."* |

主体、动作、场景全部正确。

### ④ 纯文本问答 ⚠️ 已修复答非所问，但质量仍受底座限制

修复输入格式后（见[已知问题](#已知问题与修复记录)）：

| 提问 | 修复前 | 修复后 |
|---|---|---|
| 杭州在哪里？ | ❌ *"我的手机在用什么网络？"* | ✅ *"杭州是中国浙江省的省会城市，位于中国东南部沿海地区。它以西湖闻名，是著名的旅游胜地和文化中心。"* |
| 北京在中国的哪里？ | ❌ 跑题到"中国是世界上最大的经济体" | ✅ *"北京是中国的一个重要城市，位于中国北方。它是中国的首都。"* |

---

## 系统架构

🏗️

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
│  LayerNorm + Linear + GELU + Linear    │       │  (同结构，输入 512)  │
│        1152 → 1024                     │       │      512 → 1024     │
│        2,232,577 参数                   │       │    1,575,937 参数   │
└───────────────────┬────────────────────┘       └───────────┬─────────┘
                    │                                        │
                    └──────────────┬─────────────────────────┘
                                   │
                    Replace <image> / <video> / <audio>
                        placeholder embeddings
                                   │
                                   ▼
                        送入 Qwen3 计算 logits
```

**关键设计：**

- **视频与图像共用同一个 `VisionProjector`**，不引入额外参数
- 三种模态的占位符 Token 全部取自 Qwen3 词表自带的特殊 Token，**无需扩展词表**
- 冻结 SigLIP / Whisper / Qwen3，**仅训练投影层**（3,808,514 参数，0.25%）

---

## 项目结构

📂

```text
1.5B-MLLM/
├── mllm/                          # 核心代码包
│   ├── projectors.py              # 投影层（Vision / Audio）
│   ├── encoders.py                # 编码器封装（SigLIP / Whisper / Video）
│   ├── modeling_mllm.py           # 总模型（占位符替换 + Qwen3 前向）
│   ├── dataset.py                 # 多模态数据集 + collate_fn
│   ├── utils.py                   # 占位符分配、断点保存/加载
│   ├── video_io.py                # 视频解码兼容层（torchvision → PyAV）
│   └── audio_io.py                # 音频解码兼容层（librosa → soundfile → wave）
├── models/                        # 预训练权重（需下载）
├── checkpoints/                   # 训练产物（*.pt，43.6 MB/个）
├── data/                          # 数据目录
├── scripts/
│   ├── train.py                   # 投影层训练（AMP + 梯度累积 + 断点续训）
│   └── inference.py               # 推理引擎（KV Cache + 停止策略）
├── tools/
│   ├── download_models.py         # 一键下载三个预训练权重
│   ├── download_datasets.py       # 下载 COCO / AudioCaps / MSR-VTT
│   ├── prepare_data.py            # 数据集 → 项目 JSONL 格式
│   ├── check_data.py              # 训练前数据体检
│   ├── make_demo_data.py          # 生成零依赖演示数据
│   ├── monitor_train.py           # 训练监控 + 收敛判断
│   ├── plot_loss.py               # 生成 loss 曲线图
│   ├── test_trained_model.py      # 训练后四模态效果测试
│   │
│   ├── # ---- 诊断与回归（本项目实测用） ----
│   ├── diag_decode.py             # 解码配置对照（含"视觉是否起作用"验证）
│   ├── diag_eos.py                # EOS 概率与排名诊断
│   ├── diag_stop_methods.py       # 四种停止方案横向对比
│   ├── diag_prompt_format.py      # 裸文本 vs chat 模板 A/B 对照
│   ├── diag_template_vision.py    # 套模板是否影响图像链路
│   ├── test_desktop_app.py        # 桌面程序自检（37 项）
│   └── test_stopping.py           # 停止策略回归（34 项）
├── desktop_app.py                 # ★ 原生桌面程序（tkinter UI + 进程内推理）
├── start_desktop.bat              # 桌面版双击入口
├── 启动桌面版.bat                 # 同上，中文名快捷入口
├── main.py                        # 架构预检
├── requirements-min.txt           # 直接依赖（推荐）
└── requirements.txt               # 完整冻结版本
```

> 本项目**不含任何 Web 界面**：没有 HTTP 服务、没有前端页面、不内嵌浏览器。
> 桌面程序在同一个进程里直接加载模型并推理。

---

## 完整训练流程

🛠️

> 只想验证代码能跑？见上面的 [快速开始](#快速开始) 第 3 节。
> 下面是从公开数据集训练出可用权重的完整步骤。

### 1. 准备数据集

```bash
# 下载并转换三个公开数据集（约 22 GB，ModelScope 快速通道，实测约 15 分钟）
python tools/download_datasets.py
python tools/prepare_data.py all

# 训练前体检
python tools/check_data.py data/train_full.jsonl
```

转换完成后得到 `data/train_full.jsonl`（**31,450 条**）：

| 数据集 | 条数 | 模态 |
|---|---|---|
| COCO 2014 | 20,000 | 图像-文本 |
| AudioCaps | 10,571 | 音频-文本 |
| MSR-VTT | 879 | 视频-文本 |

**数据格式**（JSONL，每行一个 JSON）：

```jsonl
{"text": "<image> A cat sitting on a windowsill", "image_path": "000000123456.jpg"}
{"text": "<audio> A dog barking loudly", "audio_path": "audiocaps_100011.wav"}
{"text": "<video> A person is running in the park", "video_path": "msrvtt_video9770.mp4"}
{"text": "纯文本样本，无需媒体文件"}
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `text` | ✅ | 文本，用 `<image>` / `<audio>` / `<video>` 标记媒体位置 |
| `image_path` | 可选 | 相对 `--image_root` |
| `audio_path` | 可选 | 相对 `--audio_root` |
| `video_path` | 可选 | 相对 `--video_root` |

> **媒体根目录会自动推断**：优先 `data/{images,audios,videos}`；找不到则回退到 JSONL
> 同级的同名子目录。所以 `data/demo/train.jsonl` 这种结构可以直接训练。

### 2. 训练投影层

```bash
# 8GB 显存配置（本项目实测配置）
python scripts/train.py \
    --train_data_path data/train_full.jsonl \
    --batch_size 1 --grad_accum_steps 4 \
    --num_epochs 3 --max_length 2048

# 16GB+ 显存可以加大
python scripts/train.py --batch_size 2 --grad_accum_steps 8 --num_epochs 3
```

训练日志示例：

```text
Total parameters:     1,100,677,954
Trainable parameters: 3,808,514 (0.3460%)
[AMP] autocast dtype=torch.bfloat16 | GradScaler=关闭(bf16 不需要)
[Epoch 1] step=100 loss=3.5286 lr=1.000000e-03 elapsed=00:04:46
...
[Epoch 3] step=23580 loss=1.9236 lr=2.191698e-10 elapsed=38:13:13
[Checkpoint] Saved projector weights to checkpoints/projector_epoch_3.pt
训练完成！
```

**断点续训**（优化器、调度器、学习率进度全部恢复）：

```bash
python scripts/train.py --resume_from checkpoints/projector_epoch_1.pt --num_epochs 3
```

**训练监控：**

```bash
python tools/monitor_train.py train.log              # ASCII 曲线 + 收敛判断
python tools/monitor_train.py train.log --png curve.png
```

### 3. 训练结果（实测）

| 指标 | 数值 |
|---|---|
| 训练样本 | **31,450** 条（图像 20,000 / 音频 10,571 / 视频 879） |
| 训练轮数 | **3 epochs**（23,587 优化步） |
| 训练耗时 | **38 小时 13 分** |
| Loss | **7.65 → 1.92**（最低 1.36） |
| 可训练参数 | **3,808,514**（占总量 0.25%） |
| 训练硬件 | **RTX 4060 Laptop, 8GB VRAM** |
| 权重体积 | **43.6 MB** / 个 |

| Epoch | 步数范围 | Loss 变化 | 平均 Loss |
|---|---|---|---|
| 1 | 10 ~ 7,860 | 7.65 → 2.41 | 2.7979 |
| 2 | 7,870 ~ 15,720 | 2.41 → 2.16 | 2.4391 |
| 3 | 15,730 ~ 23,580 | 2.19 → 1.92 | 2.2242 |

Loss 呈平滑下降并逐步趋于平台，未出现发散。

### 4. 回归测试

```bash
python tools/test_trained_model.py        # 四模态效果测试
python tools/test_stopping.py             # 停止策略回归（34 项）
python tools/test_desktop_app.py          # 桌面程序自检（37 项，需图形界面）
```

---

## 技术细节

🔬

### 占位符替换机制

文本中的 `<image>` / `<audio>` / `<video>` 在 tokenize **之前**被替换为 Qwen3 词表自带的特殊 Token：

| 模态 | 占位符 Token | Token ID | 展开数量 |
|---|---|---|---|
| 图像 | `<\|image_pad\|>` | 151655 | 729 |
| 音频 | `<\|vision_pad\|>` | 151654 | 1500 |
| 视频 | `<\|video_pad\|>` | 151656 | 729 |

在 `forward` 首步提取这些 Token 的位置索引，将其 Embedding 直接替换为投影后的多模态特征：

```python
vision_feats  = self.vision_encoder(pixel_values)      # [B, 729, 1152]
vision_embeds = self.vision_projector(vision_feats)    # [B, 729, 1024]
image_mask    = (input_ids == self.config.image_token_id)
inputs_embeds[image_mask] = vision_embeds.reshape(-1, 1024)
```

使用词表自带 Token 意味着**无需扩展词表、无需训练新 embedding**。

### LayerNorm 与数值尺度对齐

投影层前加 LayerNorm，是因为预训练编码器（SigLIP）输出特征范数远高于 LLM 词嵌入。实测：

| 张量 | 标准差 |
|---|---|
| Qwen3 词嵌入 | 0.0292 |
| 投影层输出 | **0.0260** |

两者高度吻合，避免了"表征惯性"。

### 投影层初始化（训练稳定的关键）

`nn.Linear` 默认初始化（fan_in Kaiming）会让输出幅度过大，导致注入 LLM 后隐状态爆炸、
**fp16 梯度全变 inf、AMP 学习率进度卡死**。

本项目采用小方差初始化 + 可学习输出缩放：

```python
nn.init.normal_(module.weight, mean=0.0, std=0.02)   # 而非默认 Kaiming
nn.init.zeros_(module.bias)
self.out_scale = nn.Parameter(torch.tensor(0.1))     # 可学习缩放
```

训练后 `out_scale` 自动收敛到 **0.027（视觉）/ 0.044（音频）**。

### 精度策略

- 默认 **bfloat16**（动态范围大，AMP 下梯度不易溢出）
- 投影层保持 **fp32**（`GradScaler` 不允许 unscale fp16 梯度）
- 编码器 / 投影层 / LLM 三者 dtype 在 forward 中自动对齐

### 视频时间池化

视频帧经 SigLIP 编码后为 `[B, N_frames, 729, H]`，通过 `mean(dim=1)` 压缩为
`[B, 729, H]`，直接复用视觉投影层，占位符数量与单张图像一致。

### KV Cache 增量解码

手写生成循环：首步处理完整序列 + 多模态特征，后续步仅传单 Token 并复用
`past_key_values`。实测生成速度约 **32~43 字符/秒**（RTX 4060 Laptop）。

### 输入格式（很重要）

Qwen3 是**指令模型**，必须套它的 chat 模板才会进入"听话答题"模式：

```text
<|im_start|>user
你的问题<|im_end|>
<|im_start|>assistant
```

不套模板时它会退化成"文本续写"模式，拿训练语料里的题目往下编 ——
这是本项目早期"答非所问"的直接原因，详见[已知问题](#已知问题与修复记录)。

---

## 已知问题与修复记录

🐛

本项目在实测中发现并修复了两个**影响可用性**的关键问题，记录如下供参考。

### 问题 1：模型"停不下来"，总是被硬截断（仅缓解，未根治）

**现象**：回复永远不结束，总在 `max_new_tokens` 处被截断，末尾是"此外"、"还有"这种半截话。

**根因（实测确认）**：**训练数据里从未出现过任何 EOS token。**

```text
原始训练数据里 <|im_end|>   出现次数: 0
原始训练数据里 <|endoftext|> 出现次数: 0

训练时实际喂进去的序列（dataset.py）:
  '<|image_pad|> A bicycle replica with a clock as the front wheel.'
  尾部 token: [4065, 13284, 13]     # 13 是句号 '.'，不是 EOS
```

`dataset.py` 用的是 `add_special_tokens=True`，但本 tokenizer 的 `bos_token` 是 `None`，
**既不补 bos 也不补 eos**。3 万条样本的 `labels` 里因此一个 EOS 都没有 ——
模型学到的唯一规律是"文字后面还有文字"。

推理时 EOS 的概率低到不像被压制，而像根本不在候选里：

```text
第 1 步时 EOS 的可能性：
  <|im_end|>  (151645): 概率 3.6e-07   排名 20623
  <|endoftext|>(151643): 概率 4.7e-08   排名 55481
```

**排除的猜测**（都逐一实测过）：

| 猜测 | 结论 |
|---|---|
| `eos_token_id` 配置错误 | ❌ 配置是对的（151645 = `<\|im_end\|>`） |
| `repetition_penalty` 惩罚了 EOS | ❌ 关成 1.0 照样不停 |
| vLLM `min_tokens` 边界 Bug | ❌ 本项目不用 vLLM |

**当前缓解方案**（`scripts/inference.py`，默认开启）：

| 手段 | 作用 |
|---|---|
| 多 EOS 一起判定 | 同时认 `151645` 和 `151643`（官方声明了两个） |
| `min_new_tokens=40` | 防止模型张嘴就选 EOS，只回一句"好的。" |
| **句末回退** | 裁到最后一个完整句，保证以 `。`/`.` 结尾 |

效果对比（同一提示词实测）：

| | 结尾字符 | 是否完整句 |
|---|---|---|
| 关闭回退 | `'要'` | ❌ 半截话 |
| 开启回退（默认） | `'。'` | ✅ |

**根治方法**：在 `mllm/dataset.py` 里 tokenize 之后手动补 EOS，再重跑训练：

```python
eos_id = self.tokenizer.eos_token_id
if eos_id is not None and input_ids[-1].item() != eos_id:
    input_ids = torch.cat([input_ids, torch.tensor([eos_id], dtype=input_ids.dtype)])
```

### 问题 2：问它问题会答非所问（已修复）

**现象**：问"杭州在哪里？"，模型答出 `有山。有树。有水。有建筑。……` 这类模板堆砌，
或把训练语料里的题目往下编。

**先排除两个常见猜测：**

| 猜测 | 实测结论 |
|---|---|
| 数据集太小 / 投影层没练到位 | ❌ **不是**。纯文字提问时投影层**根本不参与计算**（`pixel_values=None`，前向直接跳过投影） |
| 底座模型的中文能力坏了 | ❌ **不是**。Qwen3 是冻结的，权重没变。直接加载底座问同一问题，它答得完全正常 |

**真正原因有两层：**

**① 训练数据里没有"问答"这件事**

```text
train_full.jsonl        31,450 条 | 含问号 4 个 | 含中文 0 条
  train_coco.jsonl      20,000 条   「<image> A bicycle replica with a clock...」
  train_audiocaps.jsonl 10,571 条   「<audio> Constant rattling noise...」
  train_msrvtt.jsonl       879 条   「<video> ...」
```

那 4 个问号还是 COCO 图片里恰好拍到问句文字（`"Who do you think you are?"`），
**不是问答样本；中文样本数为 0**。模型学到的是「给定媒体，续写一段英文描述」，
**从来没人教过它"被提问时该怎么回答"**。

**② 推理时没有套 Qwen3 的 chat 模板**

训练喂进去的是裸文本，而 Qwen3 是**指令模型**，不套 chat 模板就会退化成"文本续写"模式。

**修复**：推理时套上 chat 模板（`InferenceConfig.use_chat_template`），
并剥掉 `<think>` 思考段落（`strip_thinking`）。

| 问题 | 修复前 | 修复后 |
|---|---|---|
| 杭州在哪里？ | ❌ "我的手机在用什么网络？" | ✅ "杭州是中国浙江省的省会城市，位于中国东南部沿海地区。它以西湖闻名…" |
| 北京在中国的哪里？ | ❌ 跑题到"中国是世界上最大的经济体" | ✅ "北京是中国的一个重要城市，位于中国北方…" |

**意外收获**：套模板后模型**会自己输出 EOS 结束了**（停止原因从 `max_new_tokens`
变成 `eos`），不再完全依赖句末回退。

**图像链路不受影响**（实测关键词命中与裸文本持平）。

复现实验脚本：`tools/diag_prompt_format.py`、`tools/diag_template_vision.py`。

---

## 已知限制

⚠️

1. **底座能力上限**：LLM 为 0.6B，知识与推理深度受底座限制。
2. **中文纯文本问答质量差**：训练数据**中文样本为 0**，且无问答样本。修复输入格式后
   已能正常作答，但质量有限。**建议用图片提问来评估本模型。**
3. **训练数据无问答样本**：31,450 条全是英文 caption，模型不具备指令跟随能力。
4. **上下文 2048**：单图 729 token、30 秒音频 1500 token，因此**图像 + 音频同时输入会超限**
   （代码会自动丢弃被截断的模态）。需要同时输入请把 `--max_length` 提到 4096。
5. **单样本限制**：每条样本最多一张图 + 一段音频 + 一个视频（图像与视频互斥）。
6. **音频上限约 30 秒**：Whisper 会截断更长音频。
7. **未做 prompt/response 分段 mask**：当前 loss 覆盖整段文本（仅 mask 占位符）。
8. **不支持图像生成 / 语音合成**：本模型只做「多模态理解 → 输出文本」。
9. **中文多模态能力几乎为零**：训练数据为英文（COCO / AudioCaps / MSR-VTT）。
10. **模型不会主动停止**：见[问题 1](#问题-1模型停不下来总是被硬截断仅缓解未根治)，
    目前靠句末回退兜底。

---

## 常见问题

❓

<details>
<summary><b>推理输出是无意义文本？</b></summary>

投影层没有训练。仓库不含训练好的投影层权重，必须自己训练或从 Releases 下载。
</details>

<details>
<summary><b>报错 <code>ModuleNotFoundError: No module named 'torch'</code>？</b></summary>

依赖没装齐，或用了系统 Python 而不是虚拟环境：

```bash
.venv\Scripts\python.exe scripts\inference.py --help                # Windows
source .venv/bin/activate && python scripts/inference.py --help      # Linux/macOS
```
</details>

<details>
<summary><b>显存不足（OOM）？</b></summary>

- 训练时降低 `--batch_size`、提高 `--grad_accum_steps`
- 推理时减小 `--max_new_tokens`
- 关闭其他占用显存的程序
</details>

<details>
<summary><b>推理特别慢？</b></summary>

可能装的是 CPU 版 torch。确认：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

应输出 `True`。否则重装 CUDA 版：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
</details>

<details>
<summary><b>图像和音频能同时输入吗？</b></summary>

默认 `--max_length 2048` 下不行（729 + 1500 已接近上限）。提到 4096 可以，但显存占用会增加。
</details>

<details>
<summary><b>为什么是原生桌面程序而不是网页？</b></summary>

早先版本用 FastAPI + 网页前端，再用 pywebview 把网页"装"进原生窗口。那条链路在
Windows 上要靠 `pythonnet` 把 .NET CLR 挂进 Python 进程，会**间歇性失败**：

```text
RuntimeError: Failed to resolve Python.Runtime.Loader.Initialize
```

一旦失败窗口根本开不出来。现在改成 **tkinter 原生控件**（Python 标准库自带），
推理在同进程内直接完成，于是不需要 HTTP 服务、不占端口、不依赖 .NET。
</details>

<details>
<summary><b>能打包成单个 .exe 吗？</b></summary>

**不支持也不建议。** 5.23 GB 权重 + 2.5 GB torch 打出来会是 3~6 GB 的巨型 exe，
每次启动都要解压，杀软也容易误报。本项目采用"绿色版"方案：整个文件夹保留，
双击 `start_desktop.bat` 即可。
</details>

---

## 未来计划

🗺️

- [ ] **补 EOS token 后重训**（根治"停不下来"，见问题 1）
- [ ] **引入中文问答数据**（COCO-CN、中文 VQA、LLaVA 指令数据）—— 提升中文能力的关键
- [ ] 加入评测脚本（CIDEr / SPICE / VQA accuracy 等指标）
- [ ] 引入 Q-Former / Perceiver Resampler 压缩视觉 Token
- [ ] 支持更精细的视频时序建模（时间位置编码）
- [ ] 增加 LoRA 微调选项
- [ ] 桌面版支持流式输出（逐 token 显示）
- [ ] 实现 prompt/response 分段 mask

---

## 📄 开源协议

本项目代码采用 **MIT License**。

⚠️ **仓库不包含预训练模型权重和训练好的投影层权重。**
Qwen3、SigLIP、Whisper 及使用的公开数据集（COCO、AudioCaps、MSR-VTT）
均遵循各自原有许可协议，使用时请自行确认合规性。

---

## 🙏 致谢

- [LLaVA](https://github.com/haotian-liu/LLaVA) — 投影层对齐范式
- [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B) — 语言模型底座
- [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384) — 视觉编码器
- [Whisper](https://github.com/openai/whisper) — 音频编码器

---

如果这个项目对你有帮助，欢迎点一个 ⭐ Star！
