# Avatar-Forever 部署配置与流式推理使用指南

> 依据 `async_vae_stream_20261019/` 目录内的 `USAGE_AND_SETUP_20261016.md`、`STREAMING_ASYNC_VAE.md`、
> `PATH_README.md`、`MANIFEST.md` 及四份 2026-10-19 报告整理。
>
> 原始项目根目录：`/root/paddlejob/workspace/env_run`
> 本工作区代码副本：`/data/avatarforever/avatarforever-code-streaming`
> 下文命令统一以 `cd /root/paddlejob/workspace/env_run` 为起点；在本工作区执行时把该路径换成
> `cd /data/avatarforever/avatarforever-code-streaming` 即可。
>
> 目录内文档写作时的实测环境为 k8s pod / 远程 VM，主机 IP 示例为 `10.54.129.139`。

---

## 1. 项目配置要求

### 1.1 硬件与操作系统

| 项目 | 要求 |
|---|---|
| 操作系统 | Linux 5.10 或兼容版本（实测 Linux 5.15） |
| GPU | NVIDIA A800-SXM4-80GB，单卡 80 GiB 显存 |
| GPU 数量 | 流式入口最低 1 张；推荐 2 张（DiT 与 VAE 分卡） |
| 驱动/CUDA | 驱动由宿主机提供，PyTorch 自带 CUDA 12.8 运行时 |
| 通信 | 多卡 TP 入口需要 NCCL 可用；流式入口 TP=1，不依赖 NCCL 集合通信 |
| 端口 | 默认 `8080`，需可从浏览器侧访问或被端口转发 |

显存实测（192×352、20 fps、chunk size 4、TP=1、`--fast-infer`）：

| 部署方式 | DiT 峰值 | VAE 峰值 | 说明 |
|---|---:|---:|---|
| 单卡（`--decode-device same`） | 68.54 GiB | 与 DiT 共享 | 80 GB 卡余量约 11 GiB |
| 双卡（`--decode-device auto`） | 68.54 GiB（GPU0） | 1.17 GiB（GPU1） | 稳定时延快约 2.6%，主要价值是分离 |

注意：`nvidia-smi` 显示的占用通常比 PyTorch allocator 峰值再高 0.5–1.5 GiB。

检查 GPU：

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
PY
```

`torchrun --nproc_per_node=N` 必须满足 `N <= 当前可见 GPU 数量`，且 `CUDA_VISIBLE_DEVICES` 的个数与 `N` 一致。

### 1.2 Python 环境

项目虚拟环境：`/opt/avatarforever-venv`

| 组件 | 版本要求 |
|---|---|
| Python | 3.10.12 |
| PyTorch | 2.8.0+cu128 |
| transformers | 4.52.4（务必固定，5.x 可能与 Gemma/SigLIP 结构不兼容） |
| av（PyAV） | 17.1.0 |
| safetensors | 已安装 |
| ltx-core / ltx-pipelines | 1.0.0 |
| PyYAML | 流式入口读取 YAML 配置所需（缺失时启动会提示 `pip install pyyaml`） |
| numpy | `util.prepare_padded_audio()` 补静音时使用 |

安装/修复版本请用 `uv pip`（不要用普通 `pip`）：

```bash
cd /tmp
uv pip install --python /opt/avatarforever-venv/bin/python \
  --index-url http://pip.baidu-int.com/simple/ \
  --allow-insecure-host pip.baidu-int.com \
  transformers==4.52.4
```

依赖自检：

```bash
python - <<'PY'
import torch
import transformers
import av
import yaml
from ltx_pipelines import ARA2VidDistilledPipeline
from ltx_core.model.video_vae import get_video_chunks_number

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("av:", av.__version__)
print("yaml:", yaml.__version__)
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

Gemma 目录至少包含：

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

### 1.4 checkpoints 软链接（要是已经满足路径要求可不做）

所有入口统一使用 `checkpoints/...` 路径，需要以下结构：

```text
/root/paddlejob/workspace/env_run/checkpoints
    -> init_model/checkpoints

/root/paddlejob/workspace/env_run/init_model/checkpoints/avatarforever-ltx-2.3-22b.safetensors
    -> ../avatarforever-ltx-2.3-22b.safetensors

/root/paddlejob/workspace/env_run/init_model/checkpoints/gemma-3-12b-it-qat-q4_0-unquantized
    -> ../gemma-3-12b-it-qat-q4_0-unquantized
```

首次配置或修复：

```bash
cd /root/paddlejob/workspace/env_run
mkdir -p init_model/checkpoints
ln -sfn ../avatarforever-ltx-2.3-22b.safetensors \
  init_model/checkpoints/avatarforever-ltx-2.3-22b.safetensors
ln -sfn ../gemma-3-12b-it-qat-q4_0-unquantized \
  init_model/checkpoints/gemma-3-12b-it-qat-q4_0-unquantized
ln -sfn init_model/checkpoints checkpoints
```

校验（`realpath` 失败则不能启动推理）：

```bash
cd /root/paddlejob/workspace/env_run
ls -lah checkpoints init_model/checkpoints
realpath checkpoints
realpath checkpoints/avatarforever-ltx-2.3-22b.safetensors
realpath checkpoints/gemma-3-12b-it-qat-q4_0-unquantized
```

期望输出：

```text
/root/paddlejob/workspace/env_run/init_model/checkpoints
/root/paddlejob/workspace/env_run/init_model/avatarforever-ltx-2.3-22b.safetensors
/root/paddlejob/workspace/env_run/init_model/gemma-3-12b-it-qat-q4_0-unquantized
```

### 1.5 关键目录与文件

```text
/root/paddlejob/workspace/env_run/
├── inference.py                      入口一：单卡推理
├── inference_tp.py                   入口二：多卡 TP 推理
├── inference_tp_interactive.py       入口三：交互式常驻推理（含 --audio-dir 批量模式）
├── inference_tp_folder_ar.py         入口四：文件夹 + 跨视频滚动 AR
├── inference_tp_folder_ar_stream.py  入口五：异步 VAE + 流式前端（本指南重点）
├── cross_video_ar_pipeline.py        CrossVideoARPipeline（跨视频 sink/滚动窗口）
├── streaming_vae.py                  异步 VAE 解码、帧广播、流式 pipeline
├── stream_server.py                  无第三方依赖的 HTTP/MJPEG 服务
├── stream_config.yaml                后端默认配置（入口自动加载）
├── frontend_config.yaml              前端播放/同步参数（入口自动加载）
├── run_stream.sh                     流式入口一键启动脚本
├── frontend/stream_player.html       浏览器前端页面
├── util.py                           参数校验、tiling、首帧条件、padded WAV
├── ltx_tp/                           张量并行 shim
├── packages/                         ltx-core / ltx-pipelines 源码副本（不参与运行）
├── checkpoints -> init_model/checkpoints
├── init_model/                       原始权重
├── audio/ audio_sentence/            输入音频目录
├── logs/                             逐 job 日志
└── output/                           视频、padded WAV、summary
```

说明：旧文档提到的 `split_audio_5s.py`、`timed_multijob_run.py` 不在本快照中；`stitch_segments.py`、
`verify_stitched_audio.py`、`gpucheck.py`、`bench_gpu.py` 在代码副本中存在。

注意：`/data/avatarforever/avatarforever-code-streaming` 是**代码副本**，其中没有
`init_model/`、`checkpoints/`、`audio_sentence/`、`logs/`、`output/`。直接在该目录启动前需要：

```bash
cd /data/avatarforever/avatarforever-code-streaming

# 1) 建立 logs/ 与输出目录（入口不会自动创建 logs/，逐 job 日志写入其中）
mkdir -p logs output

# 2) 准备音频目录（副本里已有 audio/，可直接用；或链接/复制自己的音频）
ls -1 audio | head

# 3) 准备权重：软链接到真实模型位置（<weights-root> 换成实际存放权重的目录；
#    本机当前未挂载权重，必须先准备好再启动）
mkdir -p init_model/checkpoints
ln -sfn <weights-root>/avatarforever-ltx-2.3-22b.safetensors \
  init_model/checkpoints/avatarforever-ltx-2.3-22b.safetensors
ln -sfn <weights-root>/gemma-3-12b-it-qat-q4_0-unquantized \
  init_model/checkpoints/gemma-3-12b-it-qat-q4_0-unquantized
ln -sfn init_model/checkpoints checkpoints

# 4) 副本的 stream_config.yaml 使用 audio-dir: audio（不是 audio_sentence）
./run_stream.sh --max-files 2 --linger-seconds 60
```

如果权重不在本机，把入口指向已有路径即可，不必建软链接：

```bash
cd /data/avatarforever/avatarforever-code-streaming
./run_stream.sh \
  --distilled-checkpoint-path <weights-root>/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root <weights-root>/gemma-3-12b-it-qat-q4_0-unquantized
```

### 1.6 启动前校验清单

```bash
cd /root/paddlejob/workspace/env_run

# 1) GPU 与依赖
python -c "import torch, transformers, av, yaml; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), transformers.__version__, av.__version__)"

# 2) 权重软链接
realpath checkpoints
realpath checkpoints/avatarforever-ltx-2.3-22b.safetensors
realpath checkpoints/gemma-3-12b-it-qat-q4_0-unquantized

# 3) Python 语法
python -m py_compile inference.py inference_tp.py inference_tp_interactive.py \
  inference_tp_folder_ar.py inference_tp_folder_ar_stream.py \
  cross_video_ar_pipeline.py streaming_vae.py stream_server.py util.py

# 4) 启动脚本语法
bash -n run_stream.sh

# 5) 参数可用（会打印帮助并退出）
./run_stream.sh --help
```

---

## 2. 快速开始（使用 `run_stream.sh`）

`run_stream.sh` 做三件事：切到脚本所在目录、设置 `CUDA_VISIBLE_DEVICES`、用
`torchrun --nproc_per_node=1` 启动 `inference_tp_folder_ar_stream.py`，并默认加载同目录的
`stream_config.yaml`。

脚本全文行为：

| 行为 | 说明 |
|---|---|
| 工作目录 | `cd "$(dirname "${BASH_SOURCE[0]}")"`，因此 YAML 里的相对路径都相对项目根目录 |
| GPU | `CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"`，已设置则沿用 |
| 配置文件 | 默认 `$AVATAR_STREAM_CONFIG`，否则脚本同目录的 `stream_config.yaml` |
| 位置参数 | 第一个不以 `-` 开头的参数会被当作配置文件路径并 `shift` |
| 显式配置 | 参数里已有 `--config` / `--config=` / `--no-config` 时，不再附加隐式 `--config` |
| 启动命令 | `exec torchrun --nproc_per_node=1 inference_tp_folder_ar_stream.py "${config_args[@]}" "$@"` |

### 2.1 三步启动

**第 1 步：确认输入目录**

快照里的 `stream_config.yaml` 读取 `audio_sentence`（7 个 WAV）；`avatarforever-code-streaming`
副本里该键是 `audio-dir: audio`。两者都以 `ls` 结果为准，换成自己的目录用 `--audio-dir`：

```bash
cd /root/paddlejob/workspace/env_run
ls -1 audio_sentence                     # 快照默认目录
./run_stream.sh --audio-dir audio        # 指定其它目录
```

**第 2 步：启动**

```bash
cd /root/paddlejob/workspace/env_run
./run_stream.sh
```

**第 3 步：打开前端**

启动日志会打印访问地址，形如：

```text
CUDA_VISIBLE_DEVICES=0,1
配置：/root/paddlejob/workspace/env_run/stream_config.yaml
已加载配置：/root/paddlejob/workspace/env_run/stream_config.yaml（27 项；命令行参数优先）
已加载前端配置：/root/paddlejob/workspace/env_run/frontend_config.yaml（9 项）
前端监听：0.0.0.0:8080
主机浏览器访问：http://10.54.129.139:8080/
VM 内自测：http://127.0.0.1:8080/
状态接口：http://127.0.0.1:8080/status.json
```

本环境确认可用：

- 主机浏览器：`http://10.54.129.139:8080/`（VM IP 变化时以日志“主机浏览器访问”行为准）
- VM 内浏览器：`http://127.0.0.1:8080/`

### 2.2 启动后的自测命令

```bash
# VM 内健康检查
curl -s http://127.0.0.1:8080/health

# 运行状态（phase / job / 帧进度 / 解码耗时）
curl -s http://127.0.0.1:8080/status.json

# 前端实际生效的参数（9 项）
curl -s http://127.0.0.1:8080/frontend_config.json

# 第 1 个 job 的音频（padded WAV，支持 Range）
curl -s -o /tmp/job1.wav http://127.0.0.1:8080/audio/1
```

`/health` 期望返回：

```text
ok
```

### 2.3 常用启动方式（完整命令）

```bash
cd /root/paddlejob/workspace/env_run

# 1) 使用默认 stream_config.yaml
./run_stream.sh

# 2) 使用另一份配置
./run_stream.sh my_config.yaml

# 3) 默认配置 + 命令行覆盖
./run_stream.sh --max-files 7 --port 8081

# 4) 只处理已有文件，不进入监听
./run_stream.sh --no-watch

# 5) 单卡部署：DiT 与 VAE 同卡
CUDA_VISIBLE_DEVICES=0 ./run_stream.sh --decode-device same

# 6) 忽略 YAML，完全用内置默认值 + 命令行
./run_stream.sh --no-config \
  --distilled-checkpoint-path checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio-dir audio_sentence --no-watch --max-files 2 \
  --output-dir output/manual_run --frame-rate 20 --height 192 --width 352 \
  --decode-device auto --port 8090 --linger-seconds 60

# 7) 需要归档 MP4 时
./run_stream.sh --package-video

# 8) headless / 压测：不按帧率节流，任务结束立即退出
./run_stream.sh --no-watch --no-stream-pacing --linger-seconds 0

# 9) 查看帮助（确认 YAML 与参数定义协同工作）
./run_stream.sh --help
```

直接调用 Python 入口（不经脚本）也可以，但必须用 `torchrun`：

```bash
cd /root/paddlejob/workspace/env_run

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=1 \
  inference_tp_folder_ar_stream.py --config stream_config.yaml \
  --audio-dir audio_sentence --no-watch --max-files 2 \
  --output-dir output/direct_run --frame-rate 20 --height 192 --width 352 \
  --decode-device auto --port 8080 --linger-seconds 60
```

### 2.4 默认行为：不打包 MP4

默认 `package-video: false`，只做 MJPEG 画面 + `/audio/<n>` 音频直播：

- 输出目录只有 `audio_padded/*.wav` 和 `streaming_summary.json`；
- `streaming_summary.json` 中每项为 `"output": null, "packaged": false`；
- `/video/<n>` 不存在，前端显示“已完成 · 未打包视频”。

需要回看/下载/画质对比时：

```bash
cd /root/paddlejob/workspace/env_run
./run_stream.sh --package-video --max-files 1 --no-watch --linger-seconds 30
```

开启后每个 job 生成 `<音频名>_<序号>_<时间戳>.mp4`，音轨使用 padded WAV，视频与音频严格等长。

### 2.5 停止

前台运行时按一次 `Ctrl+C`。`--linger-seconds` 到点后进程也会自行退出（默认 YAML 为 10 秒，
代码内置默认 300 秒）。调试时建议把 linger 调大，避免页面突然打不开：

```bash
cd /root/paddlejob/workspace/env_run
./run_stream.sh --linger-seconds 3600
```

### 2.6 远程访问与端口转发

直连不通时在本地主机建 SSH 隧道：

```bash
ssh -N -L 8080:127.0.0.1:8080 <user>@<vm-ip>
```

然后浏览器打开 `http://127.0.0.1:8080/`。经过跳板机时：

```bash
ssh -J <user>@<jump-host> -N -L 8080:127.0.0.1:8080 <user>@<vm-ip>
```

端口面板填写规则：

| 字段 | 建议值 | 说明 |
|---|---|---|
| 本地监听 | `127.0.0.1:8080` | 不要写 `localhost`，避免解析到 `::1` |
| 远程目标（转发 agent 与项目同 pod/VM） | `127.0.0.1:8080` | agent 在项目网络命名空间内 |
| 远程目标（agent 在宿主/集群外） | `<pod-or-vm-ip>:8080`，如 `10.54.129.139:8080` | 不能用 `0.0.0.0` |
| 浏览器地址 | `http://127.0.0.1:8080/` | — |

`0.0.0.0:8080` 是服务端 bind 地址，不能作为连接目标。`stream_server.py` 同时监听 IPv4
`0.0.0.0` 与 IPv6 `[::1]`；若仍报 `ERR_EMPTY_RESPONSE`，先把转发目标从 `localhost:8080`
改成 `127.0.0.1:8080`。

---

## 3. `stream_config.yaml` 参数配置指南

### 3.1 加载顺序与写法规则

```text
代码内置默认值  <  stream_config.yaml（或 --config 指定文件）  <  命令行参数
```

实现方式：先用只含 `--config` / `--no-config` 的轻量 parser 预解析，再把 YAML 的每个键
`set_defaults` 到完整 parser，最后正常 `parse_args()`。

| 规则 | 说明 |
|---|---|
| 选择文件 | 默认脚本同目录 `stream_config.yaml`；`$AVATAR_STREAM_CONFIG` 可覆盖；`--config <file>` 显式指定 |
| 忽略文件 | `--no-config`（显式 `--config` 仍会加载） |
| 键名 | `max-files` 与 `max_files` 等价 |
| 布尔值 | 写 `true` / `false`，不要加引号 |
| 反向开关 | 支持 `no-` 前缀，如 `no-trim-to-full-chunks: false` 等价于 `trim-to-full-chunks: true` |
| 路径 | 相对**当前工作目录**；`run_stream.sh` 会先 `cd` 到项目根目录 |
| 未知键 | 启动即报错并列出键名，避免拼写错误被静默忽略 |
| 顶层结构 | 必须是 YAML mapping |
| required 参数 | YAML 提供了模型路径后，命令行不再强制要求 |

### 3.2 参数总表（27 项 + 可选的 `prompt`）

**模型**

| 键 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `distilled-checkpoint-path` | path | `checkpoints/avatarforever-ltx-2.3-22b.safetensors` | DiT 主权重 |
| `gemma-root` | path | `checkpoints/gemma-3-12b-it-qat-q4_0-unquantized` | Gemma 文本编码器目录 |

**音频输入与监听**

| 键 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `audio-dir` | path | `audio_sentence` | 音频目录；按文件名字典序处理 |
| `max-files` | int | `100` | 最多处理的文件数（上限，不必达到）；`--watch` 下累计到此数停止 |
| `watch` | bool | `true` | 处理完已有文件后持续扫描新文件 |
| `poll-interval` | float | `1.0` | watch 扫描间隔（秒） |
| `stable-seconds` | float | `1.0` | 文件 size+mtime 保持不变多久才处理，避免读到半截 WAV |
| `watch-sort` | `name`/`mtime` | `name` | 新文件排序方式 |
| `idle-exit-seconds` | float | `30`（快照值） | 连续多久无新文件后停止监听；`0` 表示永不退出 |

支持的音频扩展名：`.wav` `.mp3` `.flac` `.m4a` `.aac` `.ogg` `.opus` `.wma` `.aif` `.aiff`。

**输出与编码**

| 键 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `output-dir` | path | `output/sentence_test_0` | padded WAV、summary、MP4 的输出根目录 |
| `frame-rate` | float | `20` | 输出帧率；同时决定 MJPEG 节流速度 |
| `height` | int | `192` | 视频高度，必须是 32 的整数倍 |
| `width` | int | `352` | 视频宽度，必须是 32 的整数倍 |
| `video-crf` | int | `12` | x264 CRF（0–51，越低质量越高、文件越大），仅在打包时生效 |
| `package-video` | bool | `false` | 每个 job 结束时编码/封装 MP4；关闭时只直播 |

**自回归 / 异步 VAE**

| 键 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `fast-infer` | bool | `true` | 快速推理路径；**入口会强制设为 True**，YAML 该项实际不改变行为 |
| `ar-video-chunk-size` | int | `4` | 每个 AR chunk 的 latent 帧数 |
| `ar-sink-first-chunk` | bool | `true` | 视频内首 chunk 作为 sink 锚点 |
| `ar-relative-positions` | bool | `true` | 使用相对位置编码（关闭会显著影响跨 chunk 稳定性） |
| `window-chunks` | int | `2` | 滚动窗口大小（含当前 chunk）；跨视频时窗口在全局 chunk 序列上滑动 |
| `carry-history` | bool | `true` | 跨视频携带历史 chunk；关闭即“每个音频独立生成”的基线 |
| `trim-to-full-chunks` | bool | `true` | 非末段向上扩展到完整 chunk 边界，并给音轨补静音 |
| `decode-device` | str | `auto` | VAE 设备：`auto`（DiT 下一张可见 GPU）/ `same`（同卡）/ `cuda:1` 等显式值 |

`--watch` 模式下下一段文件未知，因此每一段都按“非末段”处理，全部向上扩展。

**HTTP / MJPEG**

| 键 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `host` | str | `0.0.0.0` | HTTP 绑定地址；只走隧道时建议 `127.0.0.1` |
| `port` | int | `8080` | HTTP 端口；`0` 表示自动选空闲端口 |
| `jpeg-quality` | int | `85` | MJPEG 的 JPEG 质量（值越大越清晰、带宽越高；入口不做范围校验） |
| `linger-seconds` | float | `10`（快照值） | 所有 job 结束后 Web 服务继续存活的秒数；`0` 立即退出 |

**提示词（默认注释掉）**

| 键 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `prompt` | str | 注释 | 省略时使用入口内置的 home-office 数字人提示词；需要覆盖时取消注释并填写 |

### 3.3 场景化配置片段

**A. 数字人直播（默认配置的等价 YAML）**

```yaml
distilled-checkpoint-path: checkpoints/avatarforever-ltx-2.3-22b.safetensors
gemma-root: checkpoints/gemma-3-12b-it-qat-q4_0-unquantized
audio-dir: audio_sentence
max-files: 100
watch: true
poll-interval: 1.0
stable-seconds: 1.0
watch-sort: name
idle-exit-seconds: 0
output-dir: output/live_run
frame-rate: 20
height: 192
width: 352
video-crf: 12
package-video: false
fast-infer: true          # 入口会强制为 True，写出仅为与默认文件保持一致
ar-video-chunk-size: 4
ar-sink-first-chunk: true
ar-relative-positions: true
window-chunks: 2
carry-history: true
trim-to-full-chunks: true
decode-device: auto
host: 0.0.0.0
port: 8080
jpeg-quality: 85
linger-seconds: 3600
```

启动：

```bash
cd /root/paddlejob/workspace/env_run
./run_stream.sh live.yaml
```

**B. 单卡冒烟（低显存、跑完即退）**

```yaml
distilled-checkpoint-path: checkpoints/avatarforever-ltx-2.3-22b.safetensors
gemma-root: checkpoints/gemma-3-12b-it-qat-q4_0-unquantized
audio-dir: audio_sentence
max-files: 1
watch: false
output-dir: output/smoke_single_gpu
frame-rate: 20
height: 192
width: 352
package-video: true
decode-device: same
port: 8091
linger-seconds: 0
```

```bash
cd /root/paddlejob/workspace/env_run
CUDA_VISIBLE_DEVICES=0 ./run_stream.sh smoke_single_gpu.yaml --no-stream-pacing
```

**C. 高质量归档（提高分辨率、打包 MP4）**

```yaml
distilled-checkpoint-path: checkpoints/avatarforever-ltx-2.3-22b.safetensors
gemma-root: checkpoints/gemma-3-12b-it-qat-q4_0-unquantized
audio-dir: audio_sentence
max-files: 20
watch: false
output-dir: output/archive_544p
frame-rate: 20
height: 352
width: 544
video-crf: 12
package-video: true
decode-device: auto
port: 8082
linger-seconds: 60
```

```bash
cd /root/paddlejob/workspace/env_run
./run_stream.sh archive_544p.yaml
```

注意：352×544 下单 pipeline 生成慢于实时，首 chunk 需要数秒；显存占用也显著高于 192×352。

### 3.4 与 `--max-files` / `--watch` 的组合语义

以 `audio_sentence` 只有 7 个文件、`max-files: 100`、`watch: true` 为例：

1. 启动后处理已有 7 个文件；
2. 随后进入等待，页面提示“等待下一段音频…”，MJPEG 连接保持；
3. 新文件稳定 `stable-seconds` 后自动处理，累计到 100 个停止；
4. `idle-exit-seconds: 0` 时无新音频也不退出；快照默认是 `30`，即空闲 30 s 后结束监听（按需修改）；
5. `linger-seconds` 只在监听循环**退出后**开始计时。

限制：目录扫描在主推理循环内，新文件要等当前 job 生成/编码返回后才被发现；文件不会丢，
但不会被提前预生成。

### 3.5 常见配置错误

| 现象 | 原因与处理 |
|---|---|
| `配置文件 ... 中存在未知选项：'max_file'` | 键名拼写错误；改为 `max-files` |
| `配置文件 ... 的顶层必须是键值映射` | YAML 顶层不是 mapping（如写成了列表） |
| `缺少必填参数（命令行或配置文件）：--distilled-checkpoint-path` | `--no-config` 且未提供模型路径 |
| `无法读取配置文件` | 路径相对当前工作目录；不在项目根目录时用绝对路径 |
| 配置修改不生效 | 命令行同名参数优先；检查启动日志的“已加载配置”行 |
| `fast-infer: false` 无效 | 流式入口强制 `fast_infer=True`（原因是需要模块常驻） |

---

## 4. `frontend_config.yaml` 参数配置指南

### 4.1 工作机制

```text
frontend_config.yaml
        │ 入口启动时读取 + 类型/范围校验
        ▼
ServerState.frontend_config
        │ GET /frontend_config.json
        ▼
stream_player.html 启动时 fetch，覆盖内置常量
        │
        ▼
开始轮询 /status.json + 音画同步循环
```

| 规则 | 说明 |
|---|---|
| 默认文件 | 入口同级 `frontend_config.yaml` |
| 环境变量 | `$AVATAR_FRONTEND_CONFIG` |
| 显式指定 | `--frontend-config other.yaml` |
| 忽略 | `--no-frontend-config`（使用内置默认值） |
| 失败回退 | 页面请求 `/frontend_config.json` 失败时自动用内置默认值 |
| 生效方式 | 修改后重启入口 + 刷新页面；不需要重新构建前端 |
| 校验 | 未知键、非数字、越界值启动即报错 |

### 4.2 参数表（9 项，含取值范围）

| 配置项 | 类型 | 默认值 | 合法范围 | 作用 |
|---|---|---:|---|---|
| `audio_stall_ms` | int | `60` | 0–5000 | 多久没有新视频帧就暂停音频（毫秒） |
| `audio_drift_seconds` | float | `0.20` | 0–10 | 音画偏差超过多少秒时 seek 纠正 |
| `status_poll_ms` | int | `100` | 20–10000 | 页面轮询 `/status.json` 的间隔 |
| `sync_interval_ms` | int | `50` | 10–5000 | 音画同步检查循环间隔 |
| `status_timeout_ms` | int | `2500` | 100–60000 | 单次状态请求超时 |
| `stream_reconnect_ms` | int | `3000` | 0–60000 | MJPEG 断线后的最短重连间隔 |
| `audio_retry_backoff_ms` | int | `5000` | 0–60000 | 音频播放失败/被拦截后的重试间隔 |
| `mjpeg_keepalive_seconds` | float | `0.25` | 0.05–60 | job 间无新帧时后端重发最后一帧的间隔 |
| `sound_enabled` | bool | `true` | true/false | 页面是否默认尝试开启声音 |

调参建议：

- 20 fps 的自然帧间隔是 50 ms，`audio_stall_ms=60` 只略高于一帧；听感仍卡顿时可调到 100–200 ms；
- `sync_interval_ms` 应小于 `audio_stall_ms`；
- `status_poll_ms` 越小响应越快，但请求更密集；
- `mjpeg_keepalive_seconds` 只保证“流不断”，不保证“内容不停顿”。

### 4.3 示例

```yaml
audio_stall_ms: 120
audio_drift_seconds: 0.25
status_poll_ms: 100
sync_interval_ms: 50
status_timeout_ms: 2500
stream_reconnect_ms: 3000
audio_retry_backoff_ms: 5000
mjpeg_keepalive_seconds: 0.25
sound_enabled: true
```

```bash
cd /root/paddlejob/workspace/env_run

# 使用自定义前端配置
./run_stream.sh --frontend-config frontend_config.yaml

# 忽略前端配置，使用内置默认值
./run_stream.sh --no-frontend-config
```

### 4.4 常见错误

| 报错 | 处理 |
|---|---|
| `前端配置 ... 中的 'audio_stall_ms'=99999 超出范围 [0, 5000]。` | 按范围表修改 |
| `前端配置 ... 中存在未知选项：'unknown_key'` | 只允许上表 9 个键（下划线写法） |
| `前端配置 ... 中的 'sound_enabled' 必须是 true/false。` | 布尔值不加引号 |

---

## 5. 各程序入口的作用与参数配置

### 5.1 入口总览

| 入口 | 启动方式 | 适用场景 | 音频输入 | 是否常驻 | 是否流式播放 |
|---|---|---|---|---|---|
| `inference.py` | `python` | 单卡调试、提示词/输出格式验证 | `--audio-path` | 否 | 否 |
| `inference_tp.py` | `torchrun` | 多卡 TP 批量推理 | `--audio-path` | 否 | 否 |
| `inference_tp_interactive.py` | `torchrun` | 连续多轮、运行时输入、批量目录 | `--audio-path` 或 `--audio-dir` | 是（可交互） | 否 |
| `inference_tp_folder_ar.py` | `torchrun` | 文件夹批处理 + 跨视频 AR 上下文 | `--audio-dir` | 否 | 否 |
| `inference_tp_folder_ar_stream.py` | `torchrun`（TP=1） | 边生成边播放、数字人直播 | `--audio-dir` | 是（`--watch`） | 是（MJPEG） |

所有入口共享 `inference.build_arg_parser()` 的基础参数，入口再追加各自参数。

### 5.2 基础参数（各入口共用）

**模型**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--distilled-checkpoint-path` | 必填 | DiT 主权重路径 |
| `--gemma-root` | 必填 | Gemma 权重目录 |

**生成**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--audio-path` | 必填（交互/文件夹入口除外） | 输入音频 |
| `--prompt` | 各入口不同 | 文本提示词，支持多行 |
| `--seed` | `42` | 随机种子 |
| `--height` | `512` | 高度，必须为 32 的整数倍 |
| `--width` | `768` | 宽度，必须为 32 的整数倍 |
| `--num-frames` | `2001` | 帧数，必须满足 `8n+1`；**入口三/四/五会把它改为可选（默认 `None`），按每个音频时长自动推断** |
| `--frame-rate` | `25.0` | 输出帧率 |
| `--stage1-sigmas` | `1.0 0.98125 0.909375 0.421875 0.0` | 一阶段 sigma 序列 |
| `--fast-infer` / `--no-fast-infer` | 关闭 | 快速推理路径（流式入口强制开启） |

尺寸必须能被 32 整除：`352`、`512`、`544`、`768` 可用；`360`、`528`、`540` 不可用。
帧数换算：

```text
视频帧数 = 8 * ceil((ceil(音频时长 × 帧率) - 1) / 8) + 1
```

**自回归**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--ar-video-chunk-size` | `4` | 每个视频 chunk 的 latent 帧数 |
| `--ar-history-chunk-count` | `1`（可传 `none`） | 历史 chunk 数；文件夹入口由 `--window-chunks` 取代 |
| `--ar-sink-first-chunk` / `--no-...` | 开启 | 首 chunk sink 策略 |
| `--ar-relative-positions` / `--no-...` | 开启 | 相对位置编码 |
| `--ar-history-feature-cache` / `--no-...` | 关闭 | 历史特征缓存 |
| `--first-frame-prefix-condition` / `--no-...` | 关闭 | 复用首 chunk latent 与音频 token 作为动态条件 |
| `--first-frame-condition-position` | `prepend` | `prepend` / `append` |

**首帧条件**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--first-frame-condition-image-path` | 无 | 首帧参考图 |
| `--first-frame-image-strength` | `1.0` | 图片条件强度 |
| `--first-frame-image-crf` | `0` | 首帧图片压缩参数 |
| `--first-frame-channel-condition` / `--no-...` | 开启 | 首帧 channel condition |
| `--first-frame-channel-condition-mode` | `gated` | `add` / `gated` |
| `--first-frame-channel-condition-init` | `zero` | `zero` / `xavier` / `kaiming` |

**VAE tiling 与输出**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--spatial-tile-size` | `512` | 空间 tile 尺寸 |
| `--spatial-tile-overlap` | `64` | 空间 tile 重叠像素 |
| `--temporal-tile-size` | `256` | 时间 tile 尺寸 |
| `--temporal-tile-overlap` | `8` | 时间 tile 重叠帧数 |
| `--output-dir` | `videos/` | 输出目录 |
| `--output-path` | 无 | 显式指定 MP4 路径 |
| `--video-crf` | `12` | x264 CRF |
| `--video-preset` | `fast` | x264 预设 |

### 5.3 入口一：`inference.py`（单卡）

作用：单卡生成一次视频后退出，适合调试提示词、验证输出格式、与 TP 结果对照。

特点：不需要 `torchrun`；`--audio-path` 必填；默认 `DEFAULT_PROMPT` 只包含动作与时序约束，
不包含场景描述。

```bash
cd /root/paddlejob/workspace/env_run

CUDA_VISIBLE_DEVICES=0 python -u inference.py \
  --distilled-checkpoint-path checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio-path eng_test_0.wav \
  --prompt 'A friendly young professional sits in a softly lit home office in a medium close-up shot, facing the camera with natural eye contact. Realistic conversational video-call framing, subtle head motion, natural blinking, clear lip sync.' \
  --height 352 \
  --width 544 \
  --frame-rate 20 \
  --num-frames 945 \
  --output-path output/result_single.mp4
```

使用代码内默认提示词：

```bash
cd /root/paddlejob/workspace/env_run

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

多行提示词（单引号包裹）：

```bash
cd /root/paddlejob/workspace/env_run

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

查看参数：

```bash
cd /root/paddlejob/workspace/env_run
python inference.py --help
```

### 5.4 入口二：`inference_tp.py`（多卡 TP）

作用：把 DiT 的 Linear/attention 权重按 TP rank 切分，每个 rank 绑定一张 GPU，共同参与每个 AR step；
rank0 写视频，其他 rank 消费生成器保持 collective 同步。

特点：必须用 `torchrun`；`--nproc_per_node`、`CUDA_VISIBLE_DEVICES` 与实际可见 GPU 数必须匹配；
已验证 TP=2 / TP=4；TP=8 仅在容器暴露 8 张卡时可用；`TP=3` 不适用（attention heads=32 不能被 3 整除）。

TP=2：

```bash
cd /root/paddlejob/workspace/env_run

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
  --prompt 'A friendly young professional sits in a softly lit home office in a medium close-up shot, facing the camera with natural eye contact.' \
  --output-path output/result_tp2.mp4 \
  2>&1 | tee logs/inference_tp2.log
```

TP=4：

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
  --prompt 'A friendly young professional sits in a softly lit home office in a medium close-up shot, facing the camera with natural eye contact.' \
  --output-path output/result_tp4.mp4 \
  2>&1 | tee logs/inference_tp4.log
```

TP=8（仅在确认真有 8 张可见 GPU 时）：

```bash
cd /root/paddlejob/workspace/env_run

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
  --prompt 'A friendly young professional sits in a softly lit home office in a medium close-up shot, facing the camera with natural eye contact.' \
  --output-path output/result_tp8.mp4 \
  2>&1 | tee logs/inference_tp8.log
```

注意事项：

- 不能直接 `python inference_tp.py`；
- `Total including encode` 不含模型加载与进程启动时间；
- 项目重型 import 已放在 `dist.init_process_group()` 之后，不要提前，否则可能触发 NCCL 初始化崩溃。

### 5.5 入口三：`inference_tp_interactive.py`（交互式常驻）

作用：模型常驻显存，连续生成多个视频；支持运行时输入音频/提示词，也支持批量目录模式。

入口独有参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--audio-dir` | 无 | 批量模式：按文件名顺序处理整个目录，处理完退出；不传则保持交互模式 |

该入口会把 `--audio-path`、`--prompt`、`--num-frames` 变为可选：未提供时由 rank0 从 stdin 读取或按音频时长推断。

交互式启动：

```bash
cd /root/paddlejob/workspace/env_run

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  inference_tp_interactive.py \
  --distilled-checkpoint-path checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --frame-rate 20 \
  --height 352 \
  --width 544 \
  --ar-video-chunk-size 4 \
  --ar-history-chunk-count 1 \
  --ar-sink-first-chunk \
  --ar-relative-positions \
  --output-dir output/interactive_runs
```

启动后按提示输入音频路径：

```text
eng_test_0.wav
```

再输入多行提示词，单独一行 `:done` 或 `:end` 结束：

```text
A friendly young professional sits in a softly lit home office.

Natural audio-driven speaking motion with accurate lip synchronization.

Maintain strong temporal consistency across all frames.
:done
```

每轮结束提示“再来一个？输入 y 继续，输入 n 或退出命令结束：”，输入 `y` / `yes` / `继续` / `是` 继续。
退出命令（任意输入阶段均可用）：

```text
:q
:quit
/exit
/quit
exit
quit
```

批量目录模式（无人值守）：

```bash
cd /root/paddlejob/workspace/env_run

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  inference_tp_interactive.py \
  --distilled-checkpoint-path checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio-dir audio \
  --output-dir output/batch_audio_5s_192x352_fps20 \
  --frame-rate 20 \
  --height 192 \
  --width 352 \
  --fast-infer \
  --ar-video-chunk-size 4 \
  --ar-history-chunk-count 1 \
  --ar-sink-first-chunk \
  --ar-relative-positions \
  --prompt 'A friendly young Chinese professional sits in a softly lit home office in a medium close-up shot, facing the camera with natural eye contact.'
```

批量模式约定：

- 目录按文件名字典序处理，非音频扩展名忽略；
- 给 `--prompt` 则全部复用且不读 stdin；不给则启动时交互输入一次；
- 给 `--num-frames` 则全部固定；否则按每个音频时长推断 `8n+1`；
- 输出命名 `<音频名>_<序号>_<时间戳>.mp4`，日志 `logs/<同名>.log`；
- 单个音频读取失败会跳过；生成过程中的异常会终止整个进程（多卡集合通信要求同步）。

固定帧数做性能对比：

```bash
cd /root/paddlejob/workspace/env_run

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

该入口不要追加全局 `tee`，因为每轮已生成独立日志：

```text
output/interactive_runs/interactive_0001_20261015_150233.mp4
logs/interactive_0001_20261015_150233.log
```

### 5.6 入口四：`inference_tp_folder_ar.py`（文件夹 + 跨视频 AR）

作用：只从文件夹读音频；换音频文件时**不清空 chunk 历史**，窗口在全局 chunk 序列上滚动，
形成跨视频的 AR 上下文；处理完目录即退出。

入口独有参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--audio-dir` | `audio` | 音频目录 |
| `--max-files` | 无 | 只处理前 N 个（调试用） |
| `--window-chunks` | `2` | 滚动窗口大小（chunk 数，含当前 chunk） |
| `--trim-to-full-chunks` / `--no-...` | 开启 | 非末个视频扩展到完整 chunk 边界并补静音 |
| `--sink-first-video-chunk` / `--no-...` | 开启 | 用初始视频 chunk 0 作全局 sink |
| `--carry-history` / `--no-...` | 开启 | 是否携带跨视频历史（关闭即独立生成基线） |
| `--summary-path` | `<output-dir>/cross_video_summary.json` | 汇总 JSON 路径 |

语义要点：

- 第 k（k≥2）个视频第 j 个 chunk 的可见集合 = `sink` + 最近 `W-1` 个 chunk（可能来自上一个视频）+ 当前 chunk；
- 来自上一视频的 chunk 只出现在新视频第 0 个 chunk 的窗口里，之后自动滑出；
- 本入口窗口完全由 `--window-chunks` 决定，`--ar-history-chunk-count` 不生效；
- `--window-chunks` 上限校验为 `chunk_size + 8`，超过会报错（显存随窗口增长）；
- 输出 `<音频名>_<序号>_<时间戳>.mp4`，每轮独立日志；不使用 `--output-path`；
- 生成异常 fail-fast（多卡集合通信需要所有 rank 同步）。

```bash
cd /root/paddlejob/workspace/env_run

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  inference_tp_folder_ar.py \
  --distilled-checkpoint-path checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio-dir audio \
  --output-dir output/cross_video_ar \
  --window-chunks 2 \
  --frame-rate 20 \
  --height 192 \
  --width 352 \
  --fast-infer \
  --ar-video-chunk-size 4 \
  --ar-sink-first-chunk \
  --ar-relative-positions \
  --prompt 'A friendly young Chinese professional sits in a softly lit home office in a medium close-up shot, facing the camera with natural eye contact.'
```

关闭跨视频上下文做 A/B 基线：

```bash
cd /root/paddlejob/workspace/env_run

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  inference_tp_folder_ar.py \
  --distilled-checkpoint-path checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio-dir audio \
  --output-dir output/cross_video_ar_nohistory \
  --window-chunks 2 \
  --no-carry-history \
  --frame-rate 20 \
  --height 192 \
  --width 352 \
  --fast-infer \
  --ar-video-chunk-size 4 \
  --ar-sink-first-chunk \
  --ar-relative-positions \
  --prompt 'A friendly young Chinese professional sits in a softly lit home office in a medium close-up shot, facing the camera with natural eye contact.'
```

只跑前 2 个文件做冒烟：

```bash
cd /root/paddlejob/workspace/env_run

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
  inference_tp_folder_ar.py \
  --distilled-checkpoint-path checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio-dir audio_sentence \
  --max-files 2 \
  --output-dir output/folder_ar_smoke \
  --window-chunks 2 \
  --frame-rate 20 --height 192 --width 352 --fast-infer \
  --ar-video-chunk-size 4 --ar-sink-first-chunk --ar-relative-positions \
  --prompt 'A friendly young Chinese professional sits in a softly lit home office.'
```

### 5.7 入口五：`inference_tp_folder_ar_stream.py`（异步 VAE + 流式前端）

作用：在入口四的基础上，把 VAE 从“DiT 全部结束后一次性解码”改成**每个 chunk 完成后后台异步解码**，
同时提供 HTTP/MJPEG 前端，实现边生成边播放。**当前只支持 TP=1**（`WORLD_SIZE != 1` 直接退出）。

链路：

```text
DiT chunk k (GPU0)
      │  chunk 完成回调（unpatchify → CPU latent 快照）
      ▼
AsyncChunkVAEDecoder (GPU1，独立 CUDA stream)
      │  window = 当前 chunk + 前一个 chunk
      ├──► FrameBroadcaster ──► MJPEG ──► 浏览器 <img>
      └──► 逐帧缓冲 ──► [--package-video] encode_video ──► MP4（含 padded 音轨）
```

入口独有参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--host` | `0.0.0.0` | HTTP 绑定地址 |
| `--port` | `8080` | HTTP 端口；`0` 自动选空闲端口 |
| `--public-host` | 无 | 只改变终端打印的 URL，不开放 NAT/防火墙 |
| `--decode-device` | `auto` | `auto`=DiT 的下一张可见 GPU；`same`=同卡；或 `cuda:1` |
| `--decode-window-latent-frames` | `2 × chunk_size` | VAE 解码窗口（latent 帧）；须大于 chunk size 才启用末帧保留 |
| `--no-stream-pacing` | 关闭 | 不按 `--frame-rate` 节流；适合 headless/压测 |
| `--jpeg-quality` | `85` | MJPEG JPEG 质量 |
| `--package-video` / `--no-package-video` | 关闭 | 每个 job 结束时编码/封装 MP4 |
| `--frontend-config` | `frontend_config.yaml` | 前端参数 YAML |
| `--no-frontend-config` | 关闭 | 忽略前端 YAML |
| `--linger-seconds` | `300` | 最后一个 job 后 Web 服务存活秒数；`0` 立即退出 |
| `--watch` / `--no-watch` | 关闭 | 持续扫描 `--audio-dir` |
| `--poll-interval` | `1.0` | watch 扫描间隔（秒） |
| `--stable-seconds` | `1.0` | 新文件稳定时长（秒） |
| `--watch-sort` | `name` | `name` / `mtime` |
| `--idle-exit-seconds` | `0` | 空闲多久后停止监听；`0` 表示一直等 |
| `--config` / `--no-config` | 自动加载 | 后端 YAML 配置开关 |

继承自入口四的参数：`--audio-dir`（默认 `audio`）、`--max-files`、`--window-chunks`、
`--trim-to-full-chunks`、`--sink-first-video-chunk`、`--carry-history`、`--summary-path` 等。

行为细节：

- 每完成一个 chunk 提交一次 latent 快照，回调只入队不等待解码，不阻塞 DiT；
- 解码窗口默认 8 个 latent 帧；非末尾窗口保留最后 8 个像素帧，等下一窗口带未来上下文重解码后再发布，
  chunk 边界 PSNR 由约 33–37 dB 提升到约 44 dB；
- 有浏览器订阅时按 `--frame-rate` 节流；无订阅时不等待播放，headless 不被拖慢；
- 非末段向上扩展到完整 chunk 边界，并生成等长的 padded WAV（`<output-dir>/audio_padded/`），
  同时用于模型输入、`/audio/<n>` 与 MP4 音轨；
- watch 模式下 job 间 MJPEG 连接保持，`FrameBroadcaster` 每 `mjpeg_keepalive_seconds` 重发最后一帧；
- 前端只在首次加载模型时显示“模型加载中”，job 间保留最后一帧并显示“正在准备下一段…”；
- `streaming_summary.json` 记录每个 job 的帧数、chunk 数、AR/总耗时、DiT/VAE 峰值显存、窗口统计。

**完整命令：2 个 job + 前端 + VAE 第二张卡**

```bash
cd /root/paddlejob/workspace/env_run

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=1 \
  inference_tp_folder_ar_stream.py \
  --distilled-checkpoint-path checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio-dir audio --max-files 2 \
  --output-dir output/folder_ar_tp1_stream_2job \
  --frame-rate 20 --height 192 --width 352 --fast-infer \
  --ar-video-chunk-size 4 --ar-sink-first-chunk --ar-relative-positions \
  --decode-device auto --port 8080 --linger-seconds 300 \
  --prompt 'A friendly young professional sits in a softly lit home office in a medium close-up shot, facing the camera with natural eye contact. Realistic conversational video-call framing, subtle head motion, natural blinking, clear lip sync.'
```

**完整命令：全目录 headless（不节流、不等待播放）**

```bash
cd /root/paddlejob/workspace/env_run

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=1 \
  inference_tp_folder_ar_stream.py \
  --distilled-checkpoint-path checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio-dir audio \
  --output-dir output/folder_ar_tp1_stream_all \
  --frame-rate 20 --height 192 --width 352 --fast-infer \
  --ar-video-chunk-size 4 --ar-sink-first-chunk --ar-relative-positions \
  --decode-device auto --port 8080 --no-stream-pacing --linger-seconds 0
```

**完整命令：持续监听（数字人直播）**

```bash
cd /root/paddlejob/workspace/env_run

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=1 \
  inference_tp_folder_ar_stream.py \
  --distilled-checkpoint-path checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio-dir audio_sentence --watch \
  --poll-interval 1.0 --stable-seconds 1.0 --watch-sort name --idle-exit-seconds 0 \
  --max-files 100 \
  --output-dir output/folder_ar_tp1_stream_live \
  --frame-rate 20 --height 192 --width 352 --fast-infer \
  --ar-video-chunk-size 4 --ar-sink-first-chunk --ar-relative-positions \
  --decode-device auto --port 8080 --linger-seconds 3600
```

**完整命令：单卡 + 冒烟 + 打包**

```bash
cd /root/paddlejob/workspace/env_run

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
  inference_tp_folder_ar_stream.py --config stream_config.yaml \
  --no-watch --max-files 1 --audio-dir audio_sentence \
  --output-dir output/smoke_package_single_gpu \
  --decode-device same --package-video \
  --no-stream-pacing --linger-seconds 20 --port 8091
```

**完整命令：不打包（默认）冒烟**

```bash
cd /root/paddlejob/workspace/env_run

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=1 \
  inference_tp_folder_ar_stream.py --config stream_config.yaml \
  --no-watch --max-files 1 --audio-dir audio_sentence \
  --output-dir output/smoke_no_package \
  --linger-seconds 20 --port 8092
```

**完整命令：使用自定义前端配置并指定解码窗口**

```bash
cd /root/paddlejob/workspace/env_run

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=1 \
  inference_tp_folder_ar_stream.py --config stream_config.yaml \
  --frontend-config frontend_config.yaml \
  --decode-window-latent-frames 8 \
  --no-watch --max-files 2 --audio-dir audio_sentence \
  --output-dir output/custom_window \
  --port 8083 --linger-seconds 120
```

实测参考（192×352、20 fps、chunk 4、window 2 chunk）：

| 场景 | 结果 |
|---|---|
| 2-job 前端 | 178 帧 = 89 + 89；AR 38.66 s / 4.11 s；PSNR 44.11 / 43.87 dB |
| 14 段全目录 | 1182 帧；AR 90.56 s；含编码 98.69 s；稳态 4.64–4.71 s/段 |
| 默认不打包 | 193 帧，仅 padded WAV + summary，`output=null` |
| 开启打包 | 65 帧，MP4 音视频均 3.25 s，编码额外 0.334 s |

### 5.8 支撑模块与工具

| 文件 | 作用 |
|---|---|
| `cross_video_ar_pipeline.py` | `CrossVideoARPipeline`：sink 作为 `denoise_mask=0` 前缀，携带 chunk 进入 chunk 索引空间，窗口真正滚动；新增向后兼容的 `on_chunk_ready` / `set_streaming_chunk_callback()` |
| `streaming_vae.py` | `FrameBroadcaster`（MJPEG 帧广播 + keepalive）、`AsyncChunkVAEDecoder`（异步窗口解码 + 播放链）、`StreamingVideoDecoderProxy`、`StreamingCrossVideoARPipeline` |
| `stream_server.py` | 无第三方依赖的 `ThreadingHTTPServer`；IPv4 + IPv6 loopback 双栈；`/`、`/stream.mjpg`、`/status.json`、`/frontend_config.json`、`/audio/<n>`、`/video/<n>`、`/health` |
| `frontend/stream_player.html` | 直播画面、音频、进度、最终视频列表；启动时先取配置再轮询/同步 |
| `util.py` | `valid_num_frames`、`optional_int`、`build_first_frame_images`、`encode_first_frame_channel_condition`、`build_tiling_config`、`prepare_padded_audio`、`resolve_output_path` |
| `stitch_segments.py` | 把目录里的分段 MP4 按文件名顺序拼成连续视频（h264/AAC，支持段号角标、段首红框、边界冻结帧） |
| `verify_stitched_audio.py` | 拼接音轨回归：逐边界搜索最优对齐偏移、残差 RMS、近静音长度 |
| `gpucheck.py` | 显卡自检 |
| `bench_gpu.py` | 轻量基准 |

`prepare_padded_audio()` 行为：用 PyAV 解码源音频 → 统一为 float32 → 按
`video_duration` 补静音或截断 → 写 16-bit PCM WAV 到 `<output-dir>/audio_padded/`。

常用工具命令：

```bash
cd /root/paddlejob/workspace/env_run

# GPU 自检（无参数，直接打印版本/可用性/张量计算）
python gpucheck.py

# 分段视频拼接：第一个位置参数是目录；默认输出 <目录>/stitched_<目录名>.mp4
# <seg-dir> 放分段 mp4；--audio-source 可给音频目录或一条长音频
python stitch_segments.py <seg-dir> \
  -o output/stitched.mp4 \
  --audio-source <seg-dir 对应的音频目录或长音频.wav>

# 只看部分分段、关闭角标与红框
python stitch_segments.py <seg-dir> \
  --pattern 'seg0*.mp4' \
  --no-labels --no-boundary-mark \
  -o output/stitched_seg0x.mp4

# 拼接音轨回归验证：需要 stitch_segments.py 写出的 <输出名>.segments.txt
python verify_stitched_audio.py output/stitched.mp4 \
  --segments output/stitched.segments.txt \
  --source <原始长音频.wav>
```

`stitch_segments.py` 主要参数：`directory`（位置参数）、`-o/--output`、`--pattern`、
`--audio-source`（音频目录或长音频）、`--trim-aac-padding`、`--pad-audio-to-video`、
`--labels`、`--boundary-mark`、`--pause-frames`、`--crf`、`--preset`。

---

## 6. HTTP 接口与运行状态

| 路径 | 内容 |
|---|---|
| `/` `/index.html` | 前端页面 |
| `/stream.mjpg` | `multipart/x-mixed-replace` MJPEG 直播流 |
| `/status.json` | phase、job、帧进度、VAE 解码耗时/窗口、`package_video`、最终视频列表 |
| `/frontend_config.json` | 前端 YAML 的最终生效值（`/frontend-config.json` 亦可） |
| `/audio/<n>` | 第 n 个 job 的 padded 音频（支持 Range） |
| `/video/<n>` | 第 n 个 job 的最终 MP4（支持 Range）；未打包时 404 |
| `/health` | 固定返回 `ok` |

`/status.json` 中的 `phase`：`loading` → `generating` → `encoding` →（watch 模式 `waiting`）→ `done`，
异常为 `error`。

MJPEG 本身不带音轨：页面用 `<audio src="/audio/<n>">` 播放 padded 音频，按
`frames_published / fps` 对齐；超过 `audio_stall_ms` 无新帧暂停音频，偏差超过
`audio_drift_seconds` 做 seek 纠正。开启打包时 MP4 内封装同一份音轨。

---

## 7. 排障速查

| 现象 | 排查与处理 |
|---|---|
| 页面报 `ERR_EMPTY_RESPONSE` | 转发目标别填 `0.0.0.0` / `localhost`；VM 内用 `127.0.0.1:8080`，集群外用 `<pod-ip>:8080` |
| `服务连接超时/中断` | 先 `curl -s http://127.0.0.1:8080/health`；确认进程未被 `linger-seconds` 退出 |
| 页面一直“等待下一段音频” | watch 模式正常态；检查 `audio-dir`、文件权限、`stable-seconds` |
| 新文件不被处理 | 文件仍在写入（size/mtime 变化）；或当前 job 尚未结束（扫描在主循环内） |
| 没有 MP4 文件 | 默认 `package-video: false`；加 `--package-video` 或改 YAML |
| 显存不足 | 先 `--height 192 --width 352`；单卡用 `--decode-device same`；减小 `window-chunks` / 窗口 |
| VAE 与 DiT 抢显存 | `--decode-device auto`（需至少 2 张可见 GPU） |
| chunk 边界质量下降 | 确保 `--decode-window-latent-frames > --ar-video-chunk-size`（默认 2× 满足） |
| `currently supports TP=1 only` | 流式入口不支持多卡 TP；改用入口四或把 VAE 放第二张卡 |
| `--num-frames` 报错 | 必须满足 `8n+1`（如 9、17、25、89、193） |
| 高宽报错 | 必须是 32 的整数倍（352、512、544、768） |
| `--window-chunks ... unreasonably large`（仅入口四） | 上限为 `ar_video_chunk_size + 8`；流式入口只校验 `>= 1`，但显存仍随窗口增长 |
| NCCL 初始化崩溃 | 不要提前 import `ltx_pipelines`/`ltx_core`；使用自带入口 |
| 终端 `localStorage SecurityError` | 来自 IDE/端口转发 webview，本项目页面不使用 `localStorage` |
| 音频听感卡顿 | 把 `audio_stall_ms` 调到 100–200 ms；确认 `sync_interval_ms < audio_stall_ms` |

常用诊断命令：

```bash
cd /root/paddlejob/workspace/env_run

# 进程是否在跑、端口是否监听
ps -ef | grep inference_tp_folder_ar_stream | grep -v grep
ss -ltnp | grep 8080

# 健康与状态
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/status.json
curl -s http://127.0.0.1:8080/frontend_config.json

# 查看最近一次 job 日志
ls -lt logs/ | head
tail -n 100 logs/<job>.log
```

---

## 8. 已知限制

1. `inference_tp_folder_ar_stream.py` 只支持 TP=1；TP>1 需要各 rank 协调 VAE 解码与帧广播，暂未实现。
2. 增量解码不是逐字节复现串行全量 VAE：窗口边界与 GPU 归约顺序带来约 1/255 的数值差，
   PSNR 约 44 dB，验收请用容差指标。
3. MJPEG 直播不携带音轨，音画同步由浏览器侧完成，精度受 `status_poll_ms` / `sync_interval_ms` 限制。
4. 单 pipeline 生成速度慢于实时：下一段首 chunk 的数秒延迟会导致画面停在上一帧；
   彻底消除需要第二条独立 pipeline 预生成，或缩短首 chunk。
5. `--decode-window-latent-frames <= --ar-video-chunk-size` 时不启用末帧保留，chunk 边界质量下降。
6. 一阶段推理高宽必须为 32 的整数倍；`num_frames` 必须为 `8n+1`。
7. TP 进程数不得超过可见 GPU 数；`TP=3` 不适用（attention heads=32 不能被 3 整除）。
8. `Total including encode` 不含模型加载时间；首段耗时会明显高于稳态。
9. 分段生成与长音频一次性生成**不等价**（整体 PSNR 约 19 dB），跨视频机制追求的是身份/风格一致，
   而不是逐像素复现。
