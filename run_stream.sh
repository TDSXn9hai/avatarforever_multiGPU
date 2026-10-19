#!/usr/bin/env bash
# Launch the async-VAE streaming entry with the YAML config.
#
#   ./run_stream.sh                    # stream_config.yaml
#   ./run_stream.sh other.yaml         # another config
#   ./run_stream.sh --max-files 7      # config + CLI override
#   ./run_stream.sh --no-config ...    # ignore stream_config.yaml
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

config_path="${AVATAR_STREAM_CONFIG:-$project_dir/stream_config.yaml}"
if [[ $# -gt 0 && "$1" != -* ]]; then
  config_path="$1"
  shift
fi

has_config_flag=false
for arg in "$@"; do
  case "$arg" in
    --config|--config=*|--no-config) has_config_flag=true ;;
  esac
done

config_args=()
if [[ "$has_config_flag" == false ]]; then
  if [[ ! -f "$config_path" ]]; then
    echo "配置文件不存在：$config_path" >&2
    exit 2
  fi
  config_args=(--config "$config_path")
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
if [[ ${#config_args[@]} -gt 0 ]]; then
  echo "配置：$config_path"
fi
exec torchrun --nproc_per_node=1 inference_tp_folder_ar_stream.py \
  "${config_args[@]}" "$@"
