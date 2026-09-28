# 1.5B-MLLM
# 🚀 1.5B-MLLM: A Lightweight Multi-Modal Large Language Model

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Params](https://img.shields.io/badge/Total%20Params-1.5B-orange)
![Trainable](https://img.shields.io/badge/Trainable-3.8M%20(0.25%25)-brightgreen)

基于 **Qwen3-0.6B** 构建的轻量级多模态大模型（总参数量约 **1.5B**）。本项目从零实现了图像、音频、视频到文本的特征对齐，配套完整的训练脚本、KV Cache 加速推理，并附带一个**开箱即用的原生桌面程序**。

> **本项目已在 RTX 4060 Laptop (8GB) 上完成完整训练**：31,450 条多模态样本 × 3 epochs × 38 小时，训练过程与最终效果均有实测记录（见下文）。

---

## 📊 训练结果（实测）

![Loss Curve](loss_curve.png)

| 指标 | 数值 |
|---|---|
| 训练样本 | **31,450** 条（图像 20,000 / 音频 10,571 / 视频 879）|
| 训练轮数 | **3 epochs**（23,586 优化步）|
| 训练耗时 | **38 小时 13 分** |
| Loss | **7.65 → 1.92**（最低 1.36）|
| 可训练参数 | **3,808,514（占总量 0.25%）** |
| 训练硬件 | **RTX 4060 Laptop, 8GB VRAM** |
| 权重体积 | **43.6 MB** / 个 |

### 收敛过程

| Epoch | 步数范围 | Loss 变化 | 平均 Loss |
|---|---|---|---|
| 1 | 10 ~ 7,860 | 7.65 → 2.41 | 2.7979 |
| 2 | 7,870 ~ 15,720 | 2.41 → 2.16 | 2.4391 |
| 3 | 15,730 ~ 23,580 | 2.19 → 1.92 | 2.2242 |

Loss 呈平滑下降并逐步趋于平台，未出现发散或过拟合。

---

## 🎯 实测效果（训练权重验证）

以下全部为**训练完成后用 `projector_epoch_3.pt` 实际跑出的结果**，非示意。

### ① 图像理解

| 输入图片 | 模型输出 |
|---|---|
| `COCO_val2014_...073.jpg` | *"A motorcycle sitting by a bench with the back of a motorbike. The front is black and has a yellow stripe. The body is silver. The engine is bright."* |
| `COCO_val2014_...074.jpg` | *"There is a dog on the ground and a bike next to it. The ground is covered with cobblestone."* |
| `COCO_val2014_...136.jpg` | *"A giraffe is on a display of exhibits next to a giraffe that is sitting and eating. People are looking at it."* |

能准确识别**物体类别、颜色、场景材质**（摩托车/狗/长颈鹿、黑黄配色、鹅卵石地面）。

### ② 音频理解

| 输入音频 | 模型输出 | 数据集标签 |
|---|---|---|
| `audiocaps_100011.wav` | *"...something whirs and hums followed by a loud engine..."* | *Vibrations and humming of a power drill* |
| `audiocaps_100012.wav` | *"a man laughing and then another man laughing again as he speaks loudly... a loud splash"* | *A man speaking followed by a swoosh then a loud splash, then a man laughs* |

第二条几乎完全命中（笑声、说话、splash 全部识别）。

### ③ 视频理解

| 输入视频 | 模型输出 | 数据集标签 |
|---|---|---|
| `msrvtt_video7020.mp4` | *"A woman is making a cake with little flowers and the whole cake is very pretty."* | *a woman creating a fondant baby and flower* |

主体、动作、场景全部正确。

### ④ 纯文本（验证底座未退化）

> **问**：用一句话介绍杭州。
> **答**：杭州，中国浙江省的省会城市，位于长江三角洲地区，是长三角地区的经济中心和文化名城。

完整通顺的中文输出，证明**冻结训练策略未破坏 Qwen3 原有语言能力**。

---

## ✨ 项目亮点

- 🎯 **极简高效的投影层设计**：摒弃复杂的 Q-Former，采用 `2层MLP + GELU + LayerNorm (Pre-Norm)`。对齐不同模态的数值尺度（Norm Discrepancy），实测投影输出 std **0.026** 与 Qwen3 词嵌入 std **0.029** 高度吻合。

- 🧠 **四模态统一融合**：原生支持文本、图像、音频、视频。通过占位符替换（Placeholder Replacement）将多模态特征无缝嵌入 Qwen3 语言空间。

- 🎬 **轻量视频处理**：无需额外视频编码器，复用 SigLIP 逐帧编码 + 时间维 Mean Pooling，保持与单张图像一致的 Token 数（**729**）。

- ⚡ **冻结训练策略**：冻结 SigLIP / Whisper / Qwen3，**仅训练投影层**（3.81M 参数，0.25%）。让 1.5B 模型的全流程训练可以在 **8GB 消费级显卡**上完成。

- 🚀 **KV Cache 增量解码**：手写生成循环，首步处理完整序列 + 多模态特征，后续步仅传单 Token 并复用 `past_key_values`。

- 💻 **开箱即用的全栈部署**：FastAPI 后端 + 现代化前端（多模态上传、对话置顶、本地历史、深色主题）。

---

## 🏗️ 系统架构

<img width="758" height="930" alt="image" src="https://github.com/user-attachments/assets/ceae238e-7e73-44c6-ad4a-f16749c522ba" />



📂 项目结构

<img width="589" height="906" alt="image" src="https://github.com/user-attachments/assets/5e841a39-e6b4-482e-8d90-7659b9bf08f5" />

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

