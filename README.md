# Avatar-Forever 运行环境、模型配置与推理入口说明

本文说明 `/root/paddlejob/workspace/env_run` 项目的运行环境、模型权重目录、软链接配置，以及三个主要推理入口的参数和使用方法：

- `inference.py`：单卡推理；
- `inference_tp.py`：多卡 TP 推理；
- `inference_tp_interactive.py`：多卡交互式常驻推理。

所有命令默认在项目根目录执行：

```bash
cd /root/paddlejob/workspace/env_run
```

## 1. 运行环境要求

### 1.1 硬件和系统

推荐环境：

- Linux 5.10 或兼容版本；
- NVIDIA A800-SXM4-80GB；
- 每张 GPU 80 GiB 显存；
- NCCL 可用；
- 推理进程能够访问 CUDA 设备和 NVIDIA 驱动。

当前容器曾暴露过 4 张 GPU，因此实际启动时必须以 `nvidia-smi` 和 `torch.cuda.device_count()` 的结果为准。物理机即使有 8 张卡，如果容器只暴露 4 张，也只能使用 4 张。

检查 GPU 数量：

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
PY
```

`torchrun --nproc_per_node=N` 必须满足：

```text
N <= 当前可见 GPU 数量
```

并且 `CUDA_VISIBLE_DEVICES` 中的 GPU 数量要和 `N` 一致。例如：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ...
```

### 1.2 Python 环境

项目使用虚拟环境：

```text
/opt/avatarforever-venv
```

当前已验证的关键版本：

| 组件 | 版本或要求 |
|---|---|
| Python | 3.10.12 |
| PyTorch | 2.8.0+cu128 |
| transformers | 4.52.4 |
| av | 17.1.0 |
| safetensors | 已安装 |
| ltx-core | 1.0.0 |
| ltx-pipelines | 1.0.0 |
| CUDA | torch CUDA 可用，驱动由宿主机提供 |

`transformers` 建议固定为 `4.52.4`。模型配置和项目代码按 4.5x 版本编写，使用 5.x 可能出现 Gemma/SigLIP 结构不兼容。

如果需要在新环境中安装或修复版本，使用 `uv pip`，不要使用普通 `pip`：

```bash
cd /tmp
uv pip install --python /opt/avatarforever-venv/bin/python \
  --index-url http://pip.baidu-int.com/simple/ \
  --allow-insecure-host pip.baidu-int.com \
  transformers==4.52.4
```

项目依赖检查：

```bash
python - <<'PY'
import torch
import transformers
import av
from ltx_pipelines import ARA2VidDistilledPipeline
from ltx_core.model.video_vae import get_video_chunks_number

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("av:", av.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("imports: OK")
PY
```

### 1.3 模型权重

权重原始位置：

```text
/root/paddlejob/workspace/env_run/init_model/avatarforever-ltx-2.3-22b.safetensors
/root/paddlejob/workspace/env_run/init_model/gemma-3-12b-it-qat-q4_0-unquantized/
```

主模型：

```text
avatarforever-ltx-2.3-22b.safetensors
```

Gemma 目录至少应包含：

```text
config.json
model.safetensors.index.json
model-00001-of-00005.safetensors
model-00002-of-00005.safetensors
model-00003-of-00005.safetensors
model-00004-of-00005.safetensors
model-00005-of-00005.safetensors
```

权重完整性检查：

```bash
cd /root/paddlejob/workspace/env_run
python - <<'PY'
import json
from pathlib import Path
from safetensors import safe_open

root = Path("init_model")
main = root / "avatarforever-ltx-2.3-22b.safetensors"
gemma = root / "gemma-3-12b-it-qat-q4_0-unquantized"

print("main exists:", main.is_file(), "size:", main.stat().st_size if main.exists() else None)
print("gemma exists:", gemma.is_dir())
print("gemma shards:", len(list(gemma.glob("model-*.safetensors"))))

with safe_open(str(main), framework="pt") as f:
    cfg = json.loads(f.metadata()["config"])
    print("transformer layers:", cfg["transformer"]["num_layers"])
    print("attention heads:", cfg["transformer"]["num_attention_heads"])
PY
```

## 2. checkpoints 软链接配置

项目命令使用 `checkpoints/...` 作为统一入口。正确的目录结构是：

```text
/root/paddlejob/workspace/env_run/checkpoints
    -> init_model/checkpoints

/root/paddlejob/workspace/env_run/init_model/checkpoints/avatarforever-ltx-2.3-22b.safetensors
    -> ../avatarforever-ltx-2.3-22b.safetensors

/root/paddlejob/workspace/env_run/init_model/checkpoints/gemma-3-12b-it-qat-q4_0-unquantized
    -> ../gemma-3-12b-it-qat-q4_0-unquantized
```

首次配置或修复软链接：

```bash
cd /root/paddlejob/workspace/env_run

mkdir -p init_model/checkpoints

ln -sfn ../avatarforever-ltx-2.3-22b.safetensors \
  init_model/checkpoints/avatarforever-ltx-2.3-22b.safetensors

ln -sfn ../gemma-3-12b-it-qat-q4_0-unquantized \
  init_model/checkpoints/gemma-3-12b-it-qat-q4_0-unquantized

ln -sfn init_model/checkpoints checkpoints
```

检查软链接是否正确：

```bash
cd /root/paddlejob/workspace/env_run
ls -lah checkpoints init_model/checkpoints
realpath checkpoints
realpath checkpoints/avatarforever-ltx-2.3-22b.safetensors
realpath checkpoints/gemma-3-12b-it-qat-q4_0-unquantized
```

期望 `realpath` 指向：

```text
/root/paddlejob/workspace/env_run/init_model/checkpoints
/root/paddlejob/workspace/env_run/init_model/avatarforever-ltx-2.3-22b.safetensors
/root/paddlejob/workspace/env_run/init_model/gemma-3-12b-it-qat-q4_0-unquantized
```

如果 `realpath checkpoints/...` 失败，不能启动推理。

## 3. 共同参数说明

三个推理入口共享大部分参数。除非特别说明，下面的默认值来自当前代码。

### 3.1 模型参数

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--distilled-checkpoint-path` | Avatar-Forever DiT 主模型权重路径 | 必填 |
| `--gemma-root` | Gemma 权重目录 | 必填 |

### 3.2 生成参数

| 参数 | 含义 | 默认值 |
|---|---|---:|
| `--audio-path` | 输入音频路径；交互入口中可省略并运行时输入 | 普通入口必填 |
| `--prompt` | 文本提示词；支持 shell 多行字符串 | `DEFAULT_PROMPT` |
| `--seed` | 随机种子 | `42` |
| `--height` | 视频高度，单阶段推理要求为 32 的整数倍 | `512` |
| `--width` | 视频宽度，单阶段推理要求为 32 的整数倍 | `768` |
| `--num-frames` | 输出帧数，必须满足 `8n+1` | 普通入口 `2001`；交互入口可自动推断 |
| `--frame-rate` | 输出帧率 | `25.0` |
| `--stage1-sigmas` | 一阶段 sigma 序列 | `1.0 0.98125 0.909375 0.421875 0.0` |
| `--fast-infer` / `--no-fast-infer` | 是否启用快速推理路径 | 关闭 |

尺寸示例：

```text
352、512、544、768 等可以作为高度或宽度，因为它们是 32 的整数倍。
360、528、540 等不能直接用于一阶段推理，因为不能被 32 整除。
```

帧数示例：

```text
713、945、1177、1185、2001 都符合 8n+1。
714、1176 等不符合要求。
```

常用帧数换算：

```text
视频帧数 = 8 * ceil((ceil(音频时长 × 帧率) - 1) / 8) + 1
```

### 3.3 AR 参数

| 参数 | 含义 | 默认值 |
|---|---|---:|
| `--ar-video-chunk-size` | AR 每个视频 chunk 的大小 | `4` |
| `--ar-history-chunk-count` | 使用的历史 chunk 数；可传 `none` 关闭 | `1` |
| `--ar-sink-first-chunk` / `--no-ar-sink-first-chunk` | 是否使用首 chunk sink 策略 | 开启 |
| `--ar-relative-positions` / `--no-ar-relative-positions` | 是否使用相对位置 | 开启 |
| `--ar-history-feature-cache` / `--no-ar-history-feature-cache` | 是否缓存历史特征 | 关闭 |
| `--first-frame-prefix-condition` / `--no-first-frame-prefix-condition` | 是否复用首 chunk latent 和音频 token 作为动态条件 | 关闭 |
| `--first-frame-condition-position` | 首帧条件位置 | `prepend` |

### 3.4 首帧条件参数

| 参数 | 含义 | 默认值 |
|---|---|---:|
| `--first-frame-condition-image-path` | 首帧条件图片路径 | 无 |
| `--first-frame-image-strength` | 首帧图片条件强度 | `1.0` |
| `--first-frame-image-crf` | 首帧图片读取/压缩参数 | `0` |
| `--first-frame-channel-condition` / `--no-first-frame-channel-condition` | 是否启用首帧 channel condition | 开启 |
| `--first-frame-channel-condition-mode` | channel condition 模式：`add` 或 `gated` | `gated` |
| `--first-frame-channel-condition-init` | 初始化方式：`zero`、`xavier`、`kaiming` | `zero` |

未提供首帧图片时，默认逻辑会根据生成的首 chunk 派生 channel condition。

### 3.5 VAE 分块参数

| 参数 | 含义 | 默认值 |
|---|---|---:|
| `--spatial-tile-size` | 空间 tile 尺寸 | `512` |
| `--spatial-tile-overlap` | 空间 tile 重叠像素 | `64` |
| `--temporal-tile-size` | 时间 tile 尺寸 | `256` |
| `--temporal-tile-overlap` | 时间 tile 重叠帧数 | `8` |

### 3.6 输出参数

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--output-dir` | 默认输出目录 | `videos/` |
| `--output-path` | 明确指定输出 MP4 路径 | 无 |
| `--video-crf` | x264 CRF，数值越低质量越高、文件越大 | `12` |
| `--video-preset` | x264 编码速度/压缩率预设 | `fast` |

普通入口和 TP 入口如果不指定 `--output-path`，会根据帧数、chunk 和历史参数生成默认文件名。交互入口会按任务序号和时间戳生成独立文件名。

## 4. 入口一：单卡 `inference.py`

### 4.1 适用场景

- 单张 GPU 调试；
- 验证提示词和输出格式；
- 对比 TP 结果；
- 不需要多进程 NCCL。

普通入口会在本次进程中加载模型，生成一次视频后退出。模型不会用于多轮交互。

### 4.2 最小命令

```bash
cd /root/paddlejob/workspace/env_run

CUDA_VISIBLE_DEVICES=0 python -u inference.py \
  --distilled-checkpoint-path checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio-path eng_test_0.wav \
  --prompt "<提示词>" \
  --height 352 \
  --width 544 \
  --frame-rate 20 \
  --num-frames 945 \
  --output-path output/result_single.mp4
```

### 4.3 使用代码内默认提示词

省略 `--prompt` 即使用当前 `inference.py` 中的 `DEFAULT_PROMPT`：

```bash
CUDA_VISIBLE_DEVICES=0 python -u inference.py \
  --distilled-checkpoint-path checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio-path eng_test_0.wav \
  --height 352 \
  --width 544 \
  --frame-rate 20 \
  --num-frames 945 \
  --output-path output/result_single_default.mp4
```

注意：当前 `DEFAULT_PROMPT` 是音频驱动动作和时序稳定性描述，并不包含 home-office 场景描述。如果需要人物、场景和镜头构图，需要显式传入完整场景提示词。

### 4.4 多行 prompt 写法

推荐使用单引号包住多行文本：

```bash
python -u inference.py \
  --distilled-checkpoint-path checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio-path eng_test_0.wav \
  --prompt '第一段场景描述。

第二段动作描述。

第三段一致性约束。' \
  --height 352 --width 544 --frame-rate 20 --num-frames 945 \
  --output-path output/result_multiline.mp4
```

如果直接在终端粘贴很长的多行命令，建议先将命令保存为 shell 脚本，避免 shell 引号或换行被截断。

## 5. 入口二：多卡 TP `inference_tp.py`

### 5.1 适用场景

- 使用多张 GPU 对 DiT 做张量并行；
- 当前已验证 TP=2 和 TP=4；
- TP=8 只有在容器确实暴露 8 张 GPU 时才能启动。

当前 TP 实现的特点：

- 每个 rank 绑定一张 GPU；
- 所有 rank 共同参与每一个 DiT AR step；
- DiT 的 Linear 和 attention 相关权重按 TP rank 切分；
- rank0 写视频，其他 rank 继续消费生成器以保持 collective 同步；
- `Total including encode` 不包含模型加载时间。

### 5.2 TP=4 默认参数命令

```bash
cd /root/paddlejob/workspace/env_run

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  inference_tp.py \
  --distilled-checkpoint-path checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio-path eng_test_0.wav \
  --height 352 \
  --width 544 \
  --frame-rate 20 \
  --num-frames 945 \
  --prompt "<提示词>" \
  --output-path output/result_tp4.mp4 \
  2>&1 | tee logs/inference_tp4.log
```

### 5.3 TP=2 命令

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node=2 \
  inference_tp.py \
  --distilled-checkpoint-path checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio-path eng_test_0.wav \
  --height 352 \
  --width 544 \
  --frame-rate 20 \
  --num-frames 945 \
  --prompt "<提示词>" \
  --output-path output/result_tp2.mp4 \
  2>&1 | tee logs/inference_tp2.log
```

### 5.4 TP=8 命令

只有 `nvidia-smi` 确认可见 8 张 GPU 时才使用：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node=8 \
  inference_tp.py \
  --distilled-checkpoint-path checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio-path eng_test_0.wav \
  --height 352 \
  --width 544 \
  --frame-rate 20 \
  --num-frames 945 \
  --prompt "<提示词>" \
  --output-path output/result_tp8.mp4 \
  2>&1 | tee logs/inference_tp8.log
```

如果容器只暴露 4 张 GPU，不能用 `--nproc_per_node=8`。

### 5.5 TP 入口的重要注意事项

- 必须用 `torchrun` 启动，不能直接 `python inference_tp.py`；
- `--nproc_per_node`、`CUDA_VISIBLE_DEVICES` 和实际可见 GPU 数量必须匹配；
- 项目导入和 NCCL 初始化顺序已经在脚本中处理，不要把项目模块提前移动到进程组初始化之前；
- `--num-frames` 必须满足 `8n+1`；
- 高分辨率和长视频会显著增加显存和推理时间；
- TP=4 相比 TP=2 有加速，但不是线性加速；
- `Total including encode` 不包含模型加载和进程启动时间。

## 6. 入口三：交互式常驻 `inference_tp_interactive.py`

### 6.1 适用场景

- 连续生成多个视频；
- 运行时输入不同音频和提示词；
- 希望后续任务复用已经创建的 pipeline；
- 每轮自动生成独立日志。

启动参数中的音频路径和 prompt 可以省略，程序启动后由 rank0 从终端读取。其他 rank 不读取 stdin，而是通过分布式广播接收任务信息。

### 6.2 推荐 TP=4 启动命令

```bash
cd /root/paddlejob/workspace/env_run

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  inference_tp_interactive.py \
  --distilled-checkpoint-path checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --frame-rate 15 \
  --height 352 \
  --width 544 \
  --ar-video-chunk-size 4 \
  --ar-history-chunk-count 1 \
  --ar-sink-first-chunk \
  --ar-relative-positions \
  --output-dir output/interactive_runs
```

交互入口不需要追加：

```bash
2>&1 | tee logs/inference_tp4_interactive.log
```

因为每轮推理已经会生成独立日志，路径为：

```text
logs/interactive_0001_YYYYMMDD_HHMMSS.log
```

### 6.3 交互输入流程

模型初始化后，rank0 会提示：

```text
模型已加载并常驻显卡。
请输入音频路径：
```

输入音频路径，例如：

```text
eng_test_0.wav
```

随后输入 prompt。提示词支持多行和空行，输入单独一行 `:done` 或 `:end` 结束：

```text
A friendly young professional sits in a softly lit home office.

Natural audio-driven speaking motion with accurate lip synchronization.

Maintain strong temporal consistency across all frames.
:done
```

如果在提示词阶段直接输入空文本并结束，则使用当前 `inference.py` 中的 `DEFAULT_PROMPT`。这个默认 prompt 只包含动作和时序约束，不包含 home-office 场景描述。

如果没有在启动命令中传入 `--num-frames`，脚本会读取音频时长并自动计算帧数：

```text
raw_frames = ceil(audio_duration × frame_rate)
num_frames = 8 × ceil((raw_frames - 1) / 8) + 1
```

程序完成一轮后会打印视频路径和日志路径，并询问：

```text
再来一个？输入 y 继续，输入 n 或退出命令结束：
```

输入 `y`、`yes`、`继续` 或 `是` 开始下一轮。

### 6.4 退出命令

以下命令在音频路径、prompt 和“再来一个？”阶段均可使用：

```text
:q
:quit
/exit
/quit
exit
quit
```

退出时脚本会释放 pipeline、清理 CUDA cache、销毁进程组并退出。正常退出后显存应回落到空闲状态。

### 6.5 固定帧数的交互式性能对比

为了和固定基准严格比较，可以在启动命令中传入 `--num-frames`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  inference_tp_interactive.py \
  --distilled-checkpoint-path checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --frame-rate 25 \
  --height 352 \
  --width 544 \
  --num-frames 1177 \
  --ar-video-chunk-size 4 \
  --ar-history-chunk-count 1 \
  --ar-sink-first-chunk \
  --ar-relative-positions \
  --output-dir output/interactive_runs
```

此时每一轮都会使用固定 1177 帧，不再根据音频时长自动改变帧数。

### 6.6 交互入口日志

每轮任务生成：

```text
output/interactive_runs/interactive_0001_20261015_150233.mp4
logs/interactive_0001_20261015_150233.log
```

每轮日志包括：

- 任务序号；
- 音频路径和音频时长；
- 完整 prompt，包括多行和空行；
- 帧数和帧率；
- 输出路径；
- AR generation time；
- Total including encode；
- 峰值显存；
- 异常时的堆栈信息。

需要注意：当前实测表明，交互入口打印“模型加载完成”时，不一定代表完整权重已经完全 materialize 到 GPU；首次推理可能触发延迟加载。后续任务没有观察到重新加载整套权重的明显证据，但首次推理前后显存会发生变化。

## 7. 默认提示词和场景提示词

### 7.1 当前代码默认提示词

当前 `inference.py` 的 `DEFAULT_PROMPT` 是：

```text
Natural audio-driven speaking motion with accurate lip synchronization, smooth and continuous facial animation, subtle head movement, natural blinking, gentle breathing, and relaxed upper-body motion. Expressions and gestures should respond naturally to the rhythm, tone, and emotion of the speech while remaining restrained and realistic.

Maintain strong temporal consistency across all frames. Keep facial appearance, identity, pose, body structure, clothing details, lighting, and background stable throughout the video. Avoid sudden motion changes, excessive gestures, unnatural expression shifts, frame-to-frame appearance variation, flickering, jitter, ghosting, texture instability, temporal artifacts, or deformation. All motion should be coherent, fluid, stable, and naturally driven by the audio.
```

### 7.2 Home-office 场景示例

如果需要 home-office 场景，请显式传入场景描述和动作约束的组合 prompt，例如：

```text
A friendly young professional sits in a softly lit home office in a medium close-up shot, facing the camera with natural eye contact. Realistic conversational video-call framing, subtle head motion, natural blinking, soft facial expressions, clear lip sync, gentle breathing, and small relaxed hand gestures while speaking. Single continuous shot with a fixed camera; no cuts, transitions, angle changes, or viewpoint changes.

Natural audio-driven speaking motion with accurate lip synchronization, smooth and continuous facial animation, subtle head movement, natural blinking, gentle breathing, and relaxed upper-body motion. Expressions and gestures should respond naturally to the rhythm, tone, and emotion of the speech while remaining restrained and realistic.

Maintain strong temporal consistency across all frames. Keep facial appearance, identity, pose, body structure, clothing details, lighting, and background stable throughout the video. Avoid sudden motion changes, excessive gestures, unnatural expression shifts, frame-to-frame appearance variation, flickering, jitter, ghosting, texture instability, temporal artifacts, or deformation. All motion should be coherent, fluid, stable, and naturally driven by the audio.
```

## 8. 常用检查和排障

### 8.1 检查 GPU 是否可用

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

如果出现 `No CUDA GPUs are available` 或 `failed to communicate with the NVIDIA driver`，先恢复 GPU 设备访问，不要启动模型推理。

### 8.2 检查命令参数

```bash
python inference.py --help
python inference_tp.py --help
python inference_tp_interactive.py --help
```

### 8.3 检查 Python 语法

```bash
python -m py_compile inference.py inference_tp.py inference_tp_interactive.py util.py
```

### 8.4 检查输出视频

```bash
python - <<'PY'
from pathlib import Path
import av

path = Path("output/result.mp4")
container = av.open(str(path))
video = next(s for s in container.streams if s.type == "video")
audio = next((s for s in container.streams if s.type == "audio"), None)

print("size:", video.width, "x", video.height)
print("fps:", float(video.average_rate))
print("frames metadata:", video.frames)
print("video duration:", float(video.duration * video.time_base))
print("audio duration:", float(audio.duration * audio.time_base) if audio and audio.duration else None)
print("decoded frames:", sum(1 for _ in container.decode(video=0)))
PY
```

### 8.5 NCCL 初始化注意事项

`inference_tp.py` 和 `inference_tp_interactive.py` 已将项目相关 import 放在 `dist.init_process_group()` 之后。不要把 `ltx_pipelines`、`ltx_core` 等重型项目导入提前到 NCCL 初始化之前，否则特定驱动和库组合下可能触发 NCCL 初始化崩溃。

## 9. 产物目录约定

推荐使用以下目录：

```text
logs/                         推理日志
output/                       视频和源码副本
output/interactive_runs/     交互式视频
checkpoints/                  模型软链接入口
init_model/                  原始模型权重
```

普通入口和 TP 入口建议使用 `logs/` 保存终端日志：

```bash
2>&1 | tee logs/inference_tp4.log
```

交互入口不建议使用全局 `tee`，因为每轮任务已经有独立日志。

## 10. 推荐启动顺序

```bash
cd /root/paddlejob/workspace/env_run

# 1. 检查 GPU 和 Python 环境
nvidia-smi
python -c "import torch, transformers; print(torch.cuda.is_available(), torch.cuda.device_count(), transformers.__version__)"

# 2. 检查软链接
realpath checkpoints
realpath checkpoints/avatarforever-ltx-2.3-22b.safetensors
realpath checkpoints/gemma-3-12b-it-qat-q4_0-unquantized

# 3. 先用短视频或较低分辨率做冒烟测试
#    注意 num-frames 必须满足 8n+1，例如 9、17、25

# 4. 再运行完整 TP=2 或 TP=4 推理

# 5. 连续任务使用 inference_tp_interactive.py
```

## 11. 关键限制

- 一阶段推理的高度和宽度必须是 32 的整数倍；
- `num_frames` 必须满足 `8n+1`；
- TP 进程数不能超过可见 GPU 数量；
- 多卡入口必须使用 `torchrun`；
- 交互入口的每个 rank 不会同时读取命令行，只有 rank0 负责输入；
- 交互入口支持多行 prompt，但必须用单独一行 `:done` 或 `:end` 结束；
- `Total including encode` 不包含模型加载时间；
- prompt 会影响生成内容和稳定性，但通常不改变 TP 通信结构；
- 当前 TP 实现是完整 DiT 张量并行，不支持直接把若干张 DiT 卡和一张独立 VAE 卡拼接成新的布局；
- `TP=3` 当前不适用，因为模型 attention heads 为 32，不能被 3 整除。
