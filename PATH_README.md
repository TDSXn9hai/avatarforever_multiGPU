# 代码/文档快照（2026-10-16）

本目录是 2026-10-16 改动的**代码文件与文档**副本，源文件仍在项目根目录（`/root/paddlejob/workspace/env_run/`），此处仅作留档，二进制内容与源文件逐字节一致（md5 见下表）。

## 快照内容

| 快照文件 | 源文件 | md5 | 行数 | 本次改动 |
|---|---|---|---:|---|
| `inference_tp_interactive.py` | `../inference_tp_interactive.py` | `1b75e4030c97855bc0ec1f2076c540e2` | 566 | 新增批量目录模式 `--audio-dir`（含 `_list_audio_files` / `_make_job` / `_read_batch_jobs`，输出命名按音频名归属）；交互模式行为与命名保持不变 |
| `inference_tp_folder_ar.py` | `../inference_tp_folder_ar.py` | `06b24ecc6304870b786bb878a0ee9e19` | 500 | 2026-10-16 版本：只读文件夹 + 跨视频 AR（历史快照） |
| `cross_video_ar_pipeline.py` | `../cross_video_ar_pipeline.py` | `8813209b7c97c3d6120e98a8e47424ad` | 727 | **新增**：`CrossVideoARPipeline`。sink 作为 `denoise_mask=0` 前缀、携带 chunk 进入 chunk 索引空间（`_cross_video_ar_loop` 是上游 AR 循环的移植，只扩展索引空间），窗口真正滚动；**关键修复**：拼接后按"更长视频"的规范网格重建整张位置表（`_canonical_video_positions` / `_canonical_audio_positions`），修掉"复制表头导致窗口内三块位置重叠 → 第 2 段起 chunk 边界出现影调/颗粒换档"的问题 |
| `USAGE_AND_SETUP_20261015.md` | `../USAGE_AND_SETUP_20261015.md` | `6e3dbd8b3037f026462281a0fc36f6b7` | 983 | 新增 §6.7 批量目录模式、§6.8 跨视频滚动窗口入口（语义/参数/窗口统计/A-B 对照）、§6.8.1「分片 vs 整段对照」（89 帧切分逐段 PSNR 表）与 §6.8.2「已修复的实现 bug：拼接破坏相对位置窗口」（含逐帧视觉复核结论） |
| `stitch_segments.py` | `../stitch_segments.py` | `75dd28977200096727375b7dedab0d73` | 341 | **新增工具**（用户要求，独立于推理入口）：把目录里的分段 mp4 按文件名顺序拼成单个连续视频（视频重编码 h264、音频 AAC），支持段号/帧号角标、段首红框、边界冻结帧，并写出段边界清单便于定位 |
| `verify_stitched_audio.py` | `../verify_stitched_audio.py` | `2ce67461b67d8f38d7b859745f3ab660` | 126 | **新增工具**：拼接音轨回归验证——逐边界搜索与源音频的最优对齐偏移、残差 RMS 与近静音长度（§4.10 的量化证据来自它） |
| `REPORT_segmented_inference_20261016.md` | `../REPORT_segmented_inference_20261016.md` | `17c5a6ac6464f1d4bdf1c308d26fe83e` | 540 | **新增报告**：分段（多音频文件）推理实验汇总——两代入口与代码改动、7 组实验数据（批量模式/`--fast-infer` A/B/静态前缀原型/滚动窗口/分片 vs 整段/连续播放与视觉复核/TP 对比）、9 个踩坑记录（含最关键的位置表被复制破坏相对位置窗口）、与整段推理的差异分析（可对齐部分 + 6 条结构性差异 + 量化差异）、性能显存与建议 |
| `REPORT_interactive_fastinfer_20261016.md` | `../REPORT_interactive_fastinfer_20261016.md` | `0434bd75e64a473103a11d291c2bdd59` | 382 | 新增 §4.5 中文音频多 Job 实验、§4.6 外部监控复测（端到端/峰值显存），以及相关结论与附录 |

校验方式：

```bash
cd /root/paddlejob/workspace/env_run
md5sum inference_tp_interactive.py inference_tp_folder_ar.py cross_video_ar_pipeline.py \
       USAGE_AND_SETUP_20261015.md REPORT_interactive_fastinfer_20261016.md
md5sum output/code_snapshot_20261016/*.py output/code_snapshot_20261016/*.md
```

## 未包含的文件（按约定排除）

以下是为了本次工作临时编写的辅助脚本，按约定未放入快照：

| 文件 | 用途 |
|---|---|
| `split_audio_5s.py` | 把长音频无损切成等长片段（当前 `audio/` 下的 89 帧片段即由 `python split_audio_5s.py cn_test_0.wav audio 4.45` 生成） |
| `timed_multijob_run.py` | 外部计时/显存监控（PTY 驱动交互入口 + `nvidia-smi` 采样，用于 §4.6） |
| `/tmp/probe_prompt_cost.py` | 一次性探针：测提示词编码耗时与文本编码器常驻显存 |

如需一并留档，直接复制这三个文件即可，它们不参与模型计算，也不影响入口行为。

## 相关数据位置

- 音频片段：`/root/paddlejob/workspace/env_run/audio/`（`cn_test_0_seg00.wav` … `cn_test_0_seg11.wav`，原文件 `cn_test_0.wav` 仍在项目根目录）
- 批量模式产物：`output/batch_audio_5s_192x352_fps20/`（12 个视频）及其日志 `logs/cn_test_0_seg*_*.log`
- 交互多 Job 产物：`output/interactive_fastinfer_cn_192x352_fps20/`、`output/interactive_fastinfer_cn_192x352_fps20_timed/`
- 时间/显存原始记录：`output/interactive_fastinfer_cn_192x352_fps20_timed/timed_run_results.json`

## 未改动的文件（供核对）

`inference_tp.py`、`ltx_tp/`、`packages/`（以及虚环境中的 `ltx_pipelines`、`ltx_core`）在 2026-10-16 本轮未修改。
注：`util.py` 已在 2026-10-19 轮次修改（新增 `prepare_padded_audio()`，见下方新表）。

## 2026-10-19 异步 VAE + 流式前端改动

本轮新增异步 VAE 解码入口和浏览器直播前端；完整设计与测试数据见
`REPORT_async_vae_streaming_20261019.md` 和 `STREAMING_ASYNC_VAE.md`。

| 文件（相对项目根目录） | 类型 | md5 | 行数 | 本次改动 |
|---|---|---|---:|---|
| `inference_tp_folder_ar_stream.py` | 新增入口 | `74d86316c2eea5155272186c0eb9bb9d` | 966 | folder/`--watch` + 异步 VAE + HTTP/MJPEG；padded WAV；可选 MP4 打包；前端 YAML |
| `util.py` | 修改 | `a640b3bb52998e9781ed0a88a1a2003a` | 205 | 新增 `prepare_padded_audio()`：用 PyAV 解码 + `wave` 写 16-bit PCM，补静音/截断到视频长度 |
| `inference_tp_folder_ar.py` | 修改 | `e89d832678016bc956618f1504c9a519` | 519 | 非末段改为向上扩展到完整 chunk 边界；使用 padded WAV；summary 增加 expanded/padded 字段 |
| `streaming_vae.py` | 新增模块 | `4aa7a704cabac94354227bdf13cc541c` | 545 | 异步 VAE + 播放链：`start_event` 串行化、`finish_decode`/`finish_playback`、帧广播与 job 间 MJPEG keepalive |
| `stream_server.py` | 新增模块 | `fb25670e9e1dc90af9a570f24355b48f` | 325 | HTTP/MJPEG 服务；IPv4+IPv6 双栈；`/frontend_config.json`；状态含 `package_video` |
| `frontend/stream_player.html` | 新增前端 | `152d6df5d24e0ae1afaf76b7b0a17a56` | 713 | 直播/可选最终视频、配置化 60ms/0.20s、watch 跨 job 不黑屏；未打包时保留最后一帧 |
| `STREAMING_ASYNC_VAE.md` | 新增文档 | `a2e7b696e5fc93667aecf0bb53b2ea6c` | 202 | 入口、参数、60ms/0.20s、`--watch`、扩展/补静音、可选打包、前端 YAML |
| `REPORT_async_vae_streaming_20261019.md` | 新增报告 | `192e061d13f4d402d6f670d3ac18db67` | 318 | 2-job/14 段实验、扩展/补静音验证、持续监听、keepalive、不黑屏 |
| `cross_video_ar_pipeline.py` | 修改 | `955b20c7b7bdf6ce23a9f882ecce4a17` | 772 | 新增向后兼容的 `on_chunk_ready` / `set_streaming_chunk_callback()`；不注册回调时行为不变 |
| `USAGE_AND_SETUP_20261016.md` | 修改 | `88b4e3da341be495f22ee5bd76e96b8a` | 1368 | §12；扩展/补静音、`--watch`、YAML 配置、可选打包、前端参数 |

本文件 `PATH_README.md` 也已复制到 `output/async_vae_stream_20261019/PATH_README.md`。

校验方式：

```bash
cd /root/paddlejob/workspace/env_run
md5sum inference_tp_folder_ar_stream.py streaming_vae.py stream_server.py \
       frontend/stream_player.html STREAMING_ASYNC_VAE.md \
       REPORT_async_vae_streaming_20261019.md cross_video_ar_pipeline.py \
       USAGE_AND_SETUP_20261016.md
md5sum output/async_vae_stream_20261019/*.py \
       output/async_vae_stream_20261019/*.md \
       output/async_vae_stream_20261019/frontend/stream_player.html
```

本轮测试产物（均在 `output/` 下）：

| 路径 | 内容 |
|---|---|
| `output/folder_ar_tp1_stream_2job/` | 2-job 前端验证 + 11:03 全目录 14 段运行；`streaming_summary.json` 为 14 job 汇总 |
| `output/folder_ar_tp1_stream_final_2job/` | 当前最终代码的 2-job headless 回归（89 + 89 帧，PSNR 44.11/43.87 dB） |
| `output/folder_ar_tp1_stream_headless/` | 单音频 `--no-stream-pacing` 冒烟 |
| `output/folder_ar_tp1_stream_smoke/` | 单音频冒烟（含一次修复前的失败记录） |
| `logs/folder_ar_stream_*.log` | 入口级运行日志 |
| `logs/cn_test_0_seg*_20261019_*.log` | 每个 job 的独立日志 |

## 2026-10-19 YAML 配置与一键启动

为 `inference_tp_folder_ar_stream.py` 增加可编辑的 YAML 配置和启动脚本；默认值对应数字人直播
场景（`audio_sentence`、`--watch`、`--max-files 100`、192×352、20 fps、8080 端口）。
详细说明见 `REPORT_stream_config_20261019.md` 与 `USAGE_AND_SETUP_20261016.md` §12.10。

| 新增/修改文件（相对项目根目录） | 类型 | md5 | 行数 | 说明 |
|---|---|---|---:|---|
| `inference_tp_folder_ar_stream.py` | 修改 | `74d86316c2eea5155272186c0eb9bb9d` | 966 | `--config` / `--no-config`；YAML 默认值；`--package-video`；前端 YAML |
| `stream_config.yaml` | 修改 | `780c07281afd426228b7b3924c620ced` | 58 | 可编辑默认配置；`package-video: false` |
| `run_stream.sh` | 新增 | `fb07671a0ce9fafc233002bc169853ab` | 42 | 一键启动（`CUDA_VISIBLE_DEVICES=0,1` + `torchrun --nproc_per_node=1`） |
| `USAGE_AND_SETUP_20261016.md` | 修改 | `88b4e3da341be495f22ee5bd76e96b8a` | 1368 | §12.10 配置说明 + 可选打包 + 前端参数 |
| `REPORT_stream_config_20261019.md` | 新增 | `cef672f6d6245f338ff1be57688d3c2a` | 143 | 改动与测试记录 |

最小启动方式：

```bash
cd /root/paddlejob/workspace/env_run
./run_stream.sh                          # 读取 stream_config.yaml
./run_stream.sh --max-files 7            # 命令行覆盖
./run_stream.sh --no-config --help       # 忽略 YAML
```

快照副本位于 `output/async_vae_stream_20261019/`（含 `stream_config.yaml`、`run_stream.sh`、
入口脚本、报告和文档）。

## 2026-10-19 可选 MP4 打包（`--package-video`）

默认不打包，只输出 MJPEG 实时画面和音轨；需要归档/回看/下载时加 `--package-video`。
详细分析与实测见 `REPORT_package_video_option_20261019.md`。

| 新增/修改文件（相对项目根目录） | 类型 | md5 | 行数 | 说明 |
|---|---|---|---:|---|
| `inference_tp_folder_ar_stream.py` | 修改 | `74d86316c2eea5155272186c0eb9bb9d` | 966 | `--package-video`；默认跳过 MP4 编码与视频注册；前端 YAML |
| `stream_server.py` | 修改 | `fb25670e9e1dc90af9a570f24355b48f` | 325 | `/status.json` 增加 `package_video`；新增 `/frontend_config.json` |
| `frontend/stream_player.html` | 修改 | `152d6df5d24e0ae1afaf76b7b0a17a56` | 713 | 未打包提示；前端参数改为从 YAML 加载 |
| `stream_config.yaml` | 修改 | `780c07281afd426228b7b3924c620ced` | 58 | `package-video: false` |
| `STREAMING_ASYNC_VAE.md` | 修改 | `a2e7b696e5fc93667aecf0bb53b2ea6c` | 202 | 参数/链路/接口说明 |
| `USAGE_AND_SETUP_20261016.md` | 修改 | `88b4e3da341be495f22ee5bd76e96b8a` | 1368 | §12 说明 |
| `REPORT_package_video_option_20261019.md` | 新增 | `868a25cad0d7bcac75562482e5b471c0` | 118 | 改动、影响分析与真实推理测试 |

真实推理测试产物：

| 路径 | 内容 |
|---|---|
| `output/no_package_test_20261019/` | 默认模式：193 帧，只有 padded WAV + summary，`output=null`、`packaged=false` |
| `output/package_video_test_20261019/` | `--package-video`：65 帧，MP4 视频/音频均 3.25 s，编码额外耗时 0.334 s |

## 2026-10-19 前端参数 `frontend_config.yaml`

前端播放、音画同步、状态轮询、重连和 MJPEG keepalive 参数从 HTML 中解耦到独立 YAML。
入口默认自动读取，并通过 `/frontend_config.json` 下发给页面；`--no-frontend-config`
可回退到内置默认值。详细测试见 `REPORT_frontend_config_20261019.md`。

| 新增/修改文件（相对项目根目录） | 类型 | md5 | 行数 | 说明 |
|---|---|---|---:|---|
| `frontend_config.yaml` | 新增 | `d403c177b4433aea17961d65f978b039` | 32 | 9 个前端参数及注释 |
| `inference_tp_folder_ar_stream.py` | 修改 | `74d86316c2eea5155272186c0eb9bb9d` | 966 | 加载/校验前端 YAML；新增 `--frontend-config` / `--no-frontend-config` |
| `stream_server.py` | 修改 | `fb25670e9e1dc90af9a570f24355b48f` | 325 | 新增 `/frontend_config.json` |
| `frontend/stream_player.html` | 修改 | `152d6df5d24e0ae1afaf76b7b0a17a56` | 713 | 启动时读取配置后再轮询/同步 |
| `REPORT_frontend_config_20261019.md` | 新增 | `6d6c95891165d8103cf334e169ee6422` | 149 | 改动与测试记录 |
| `output/frontend_config_test_20261019/` | 测试产物 | - | - | 真实入口冒烟：65 帧、`/frontend_config.json` 9 项、未打包 MP4 |

## 2026-10-19 单卡 / 双卡部署对比测试

同一段 4.45 s 音频（89 帧，192×352，20 fps），比较 DiT+VAE 同卡与 DiT/VAE 分卡；
每配置两轮，取 job 2 稳定值。结论：双卡 AR 时延 4.199 s → 4.090 s（快约 2.6%）；
GPU0 峰值均为 68.54 GiB，双卡在 GPU1 增加约 1.17 GiB VAE 占用。

| 文件/路径 | 类型 | md5 | 说明 |
|---|---|---|---|
| `REPORT_deploy_topology_compare_20261019.md` | 新增 | `0ff5cf02fa9fe1abb5ac71276bcd31f5` | 测试方法、原始数据、均值与结论 |
| `artifacts/output/deploy_compare_single_gpu_20261019/` | 测试产物 | - | 单卡第 1 轮 |
| `artifacts/output/deploy_compare_single_gpu_run2_20261019/` | 测试产物 | - | 单卡第 2 轮 |
| `artifacts/output/deploy_compare_dual_gpu_20261019/` | 测试产物 | - | 双卡第 1 轮 |
| `artifacts/output/deploy_compare_dual_gpu_run2_20261019/` | 测试产物 | - | 双卡第 2 轮 |
