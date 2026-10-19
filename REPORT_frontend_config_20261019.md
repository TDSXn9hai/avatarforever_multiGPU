# 前端参数 YAML 配置（2026-10-19）

## 1. 目标

把原先写死在 `frontend/stream_player.html` 里的播放、音画同步、状态轮询和重连参数，
收敛到可编辑的 `frontend_config.yaml`。入口启动时加载并校验，再通过
`/frontend_config.json` 下发给浏览器。

默认值与改动前完全一致，因此不修改 YAML 时行为不变：

```text
audio_stall_ms=60
audio_drift_seconds=0.20
status_poll_ms=100
sync_interval_ms=50
status_timeout_ms=2500
stream_reconnect_ms=3000
audio_retry_backoff_ms=5000
mjpeg_keepalive_seconds=0.25
sound_enabled=true
```

## 2. 工作方式

```text
frontend_config.yaml
        │ 入口启动时读取 + 类型/范围校验
        ▼
ServerState.frontend_config
        │ GET /frontend_config.json
        ▼
stream_player.html 启动时 fetch
        │ 覆盖默认常量
        ▼
开始轮询 /status.json + 音画同步循环
```

- 入口默认自动读取项目根目录的 `frontend_config.yaml`；
- 可用 `--frontend-config other.yaml` 指定其它文件；
- `--no-frontend-config` 使用内置默认值；
- 网页端请求失败（例如旧版后端没有该接口）时自动回退到内置默认值；
- 未知键、非数字值、超出范围的值在启动时直接报错；
- 修改后重启入口并刷新页面即可，不需要重新构建前端。

## 3. 文件改动

| 文件 | 类型 | 行数 | md5 | 说明 |
|---|---|---:|---|---|
| `inference_tp_folder_ar_stream.py` | 修改 | 966 | `74d86316c2eea5155272186c0eb9bb9d` | 加载/校验前端 YAML；新增 `--frontend-config` / `--no-frontend-config`；keepalive 传入 `FrameBroadcaster` |
| `stream_server.py` | 修改 | 325 | `fb25670e9e1dc90af9a570f24355b48f` | 保存前端配置并提供 `/frontend_config.json` |
| `frontend/stream_player.html` | 修改 | 713 | `152d6df5d24e0ae1afaf76b7b0a17a56` | 常量改为可覆盖变量；启动时先取配置再开始轮询/同步 |
| `frontend_config.yaml` | 新增 | 32 | `d403c177b4433aea17961d65f978b039` | 9 个前端参数及注释 |
| `STREAMING_ASYNC_VAE.md` | 修改 | 202 | `a2e7b696e5fc93667aecf0bb53b2ea6c` | 参数、接口、文件说明 |
| `USAGE_AND_SETUP_20261016.md` | 修改 | 1368 | `88b4e3da341be495f22ee5bd76e96b8a` | §12.3 / §12.5 / §12.10.1 说明 |

## 4. 参数说明

| 配置项 | 默认值 | 作用 |
|---|---:|---|
| `audio_stall_ms` | `60` | 多久没有新视频帧就暂停音频 |
| `audio_drift_seconds` | `0.20` | 音画偏差超过多少秒时 seek 纠正 |
| `status_poll_ms` | `100` | `/status.json` 轮询间隔 |
| `sync_interval_ms` | `50` | 音画同步检查循环间隔 |
| `status_timeout_ms` | `2500` | 单次状态请求超时 |
| `stream_reconnect_ms` | `3000` | MJPEG 断线后的最短重连间隔 |
| `audio_retry_backoff_ms` | `5000` | 音频播放失败后的重试间隔 |
| `mjpeg_keepalive_seconds` | `0.25` | job 间无新帧时重发最后一帧的间隔 |
| `sound_enabled` | `true` | 页面是否默认尝试开启声音 |

范围校验：`audio_stall_ms` 0–5000、`audio_drift_seconds` 0–10、
`status_poll_ms` 20–10000、`sync_interval_ms` 10–5000、`status_timeout_ms` 100–60000、
`stream_reconnect_ms` 0–60000、`audio_retry_backoff_ms` 0–60000、
`mjpeg_keepalive_seconds` 0.05–60。

## 5. 测试结果

### 5.1 配置加载

```text
loaded: /root/paddlejob/workspace/env_run/frontend_config.yaml
items: 9
audio_stall_ms 60 int
audio_drift_seconds 0.2 float
status_poll_ms 100 int
sync_interval_ms 50 int
status_timeout_ms 2500 int
stream_reconnect_ms 3000 int
audio_retry_backoff_ms 5000 int
mjpeg_keepalive_seconds 0.25 float
sound_enabled True bool
```

`--no-frontend-config` 路径返回内置默认值，且不读取文件。

### 5.2 HTTP 下发

用独立 `ServerState(frontend_config=...)` 启动 HTTP 服务：

```text
GET /frontend_config.json
=> {"audio_stall_ms": 80, "audio_drift_seconds": 0.3, "sound_enabled": false}
```

同时确认 `/` 返回的页面包含 `fetch('/frontend_config.json')` 逻辑，且常量已改为可覆盖变量。

### 5.3 错误处理

```text
audio_stall_ms: 99999
=> 前端配置 ... 中的 'audio_stall_ms'=99999 超出范围 [0, 5000]。

unknown_key: 1
=> 前端配置 ... 中存在未知选项：'unknown_key'；可用选项：...

sound_enabled: "yes"
=> 前端配置 ... 中的 'sound_enabled' 必须是 true/false。
```

### 5.4 语法与启动参数

- `python -m py_compile inference_tp_folder_ar_stream.py stream_server.py` 通过；
- HTML 解析通过；
- `./run_stream.sh --help` 能看到 `--frontend-config` / `--no-frontend-config`；
- 此前 `output/no_package_test_20261019` 和 `output/package_video_test_20261019` 的测试结论继续有效。

### 5.5 真实入口冒烟

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=1 \
  inference_tp_folder_ar_stream.py --config stream_config.yaml \
  --no-watch --max-files 1 --audio-dir audio_expand_test \
  --output-dir output/frontend_config_test_20261019 \
  --linger-seconds 15 --port 8093 --no-stream-pacing
```

结果：

- 启动日志包含
  `已加载前端配置：/root/.../frontend_config.yaml（9 项）`；
- `GET /frontend_config.json` 返回全部 9 个参数，值与 YAML 一致；
- 65 帧推理完成，日志为 `完成：job 1 未打包 MP4（41.14s ...）`；
- 输出目录只有 padded WAV 和 `streaming_summary.json`（默认不打包）；
- 日志无 `ERROR`/`Traceback`，进程在 linger 后正常退出。

## 6. 结论

前端参数已从 HTML 中解耦，默认行为不变；调参只需修改 YAML 并重启入口。浏览器在开始轮询和
音画同步前会先读取 `/frontend_config.json`，因此 60 ms / 0.20 s 等参数可以在不重新构建前端
的情况下调整。
