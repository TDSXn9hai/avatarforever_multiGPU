# 流式入口 YAML 配置与一键启动（2026-10-19）

## 1. 目标

`inference_tp_folder_ar_stream.py` 的启动参数较长，数字人直播场景又需要固定使用
`--watch`、`--max-files 100`、192×352、20 fps 等一组配置。本次改动把常用参数收敛到可编辑的
YAML 文件，并提供一个启动脚本，避免每次手写完整命令。

本次只改参数装载逻辑，不涉及 DiT/VAE、流式解码、前端播放和音视频同步路径。

## 2. 改动

| 文件 | 类型 | 行数 | md5 | 说明 |
|---|---|---:|---|---|
| `inference_tp_folder_ar_stream.py` | 修改 | 833 | `45cdb597910881011e5a74043be0f24f` | 新增 `--config` / `--no-config`；YAML 作为默认值，命令行优先；`--watch` 改为布尔可选（支持 `--no-watch`） |
| `stream_config.yaml` | 新增 | 56 | `09c145a7f581aabfe26a9e9b5ccfaea1` | 默认可编辑配置，对应直播场景 |
| `run_stream.sh` | 新增 | 42 | `fb07671a0ce9fafc233002bc169853ab` | 一键启动：默认 `CUDA_VISIBLE_DEVICES=0,1` + `torchrun --nproc_per_node=1` |
| `USAGE_AND_SETUP_20261016.md` | 修改 | 1334 | `cf9cb361e8ba030b74bc0c65986f72db` | 新增 §12.10 配置说明 |

### 2.1 装载顺序

```text
代码内置默认值  <  stream_config.yaml（或 --config 指定文件）  <  命令行参数
```

实现上先用一个只含 `--config` / `--no-config` 的轻量 parser 预解析，再把 YAML 的每个键
`set_defaults` 到完整 parser 上，最后正常 `parse_args()`。因此：

- `--config` 可以补上 `--distilled-checkpoint-path`、`--gemma-root` 这类原本标记为 required 的参数；
- 命令行里出现的同名参数仍会覆盖 YAML；
- `Path`、`int`、`float`、`bool` 会按 argparse 原有 `type` 转换；
- YAML 顶层必须是 mapping，未知键会直接报错，避免拼写错误被静默忽略。

### 2.2 默认配置

`stream_config.yaml` 的关键默认值：

```yaml
audio-dir: audio_sentence
max-files: 100
watch: true
output-dir: output/sentence_test_0
frame-rate: 20
height: 192
width: 352
decode-device: auto
port: 8080
linger-seconds: 10
```

启动：

```bash
cd /root/paddlejob/workspace/env_run
./run_stream.sh
```

临时覆盖：

```bash
./run_stream.sh --max-files 7 --port 8081
./run_stream.sh --no-watch
./run_stream.sh other.yaml
```

## 3. 测试结果

### 3.1 配置装载与类型转换

用真实 parser（`inference.build_arg_parser` + folder/stream/config 参数组）装载
`stream_config.yaml`，共应用 26 项：

```text
applied: 26
distilled_checkpoint_path = PosixPath('checkpoints/avatarforever-ltx-2.3-22b.safetensors')
gemma_root               = PosixPath('checkpoints/gemma-3-12b-it-qat-q4_0-unquantized')
audio_dir                = PosixPath('audio_sentence')
max_files                = 100 (int)
watch                    = True (bool)
output_dir               = PosixPath('output/sentence_test_0')
frame_rate               = 20 (int)
height                   = 192 (int)
width                    = 352 (int)
fast_infer               = True (bool)
ar_video_chunk_size      = 4 (int)
decode_device            = 'auto' (str)
linger_seconds           = 10 (int)
trim_to_full_chunks      = True (bool)
port                     = 8080 (int)
```

### 3.2 命令行覆盖

```text
argv: --config stream_config.yaml --max-files 7 --no-watch --port 9090
override: max_files=7 watch=False port=9090 audio_dir=audio_sentence
```

说明 YAML 提供默认值、命令行覆盖生效，且 `--no-watch` 可用。

### 3.3 启动脚本

```bash
bash -n run_stream.sh
./run_stream.sh --help
```

输出中先打印：

```text
CUDA_VISIBLE_DEVICES=0,1
配置：/root/paddlejob/workspace/env_run/stream_config.yaml
已加载配置：/root/paddlejob/workspace/env_run/stream_config.yaml（26 项；命令行参数优先）
```

随后正常显示 argparse 帮助，说明脚本、配置加载和入口参数定义可以协同工作。

### 3.4 错误处理

- 未知键 `max_file`：退出码 2，提示
  `配置文件 bad.yaml 中存在未知选项：'max_file'`；
- `./run_stream.sh --no-config --help`：不加载 YAML，恢复内置默认值，required 参数重新出现；
- `--no-config` 且未提供模型路径时：argparse 报
  `the following arguments are required: --distilled-checkpoint-path, --gemma-root`，退出码 2。

### 3.5 未重跑模型推理的原因

本次改动位于 `parse_args()` 之前，配置最终仍落到原来的 `args` 字段；DiT、异步 VAE、MJPEG
和前端代码均未修改。因此没有重复加载 22B 权重做端到端生成，之前的流式推理与前端测试结论
继续适用。若需要完整回归，可直接执行 `./run_stream.sh --max-files 1 --no-watch`。

## 4. `--watch` + `--max-files 100` 的行为

当前 `audio_sentence` 只有 7 个 WAV：

1. 启动后处理已有 7 个文件；
2. 随后进入等待，页面提示“等待下一段音频…”，MJPEG 连接保持；
3. 新文件到来后继续处理，直到累计达到 100 个；
4. `idle-exit-seconds: 0` 表示没有新音频时一直等待；
5. `linger-seconds` 只在监听循环退出后计时。

限制：目录扫描仍在主推理循环内，新文件会在当前 job 完成后被检测到；文件不会丢，但不会在
当前 job 生成期间被提前处理。
