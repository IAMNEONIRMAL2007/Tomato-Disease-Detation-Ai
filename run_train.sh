# Created By NirmalBorole
#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/nirma/tomato_env/lib/python3.11/site-packages/nvidia/cublas/lib:/home/nirma/tomato_env/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/home/nirma/tomato_env/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/home/nirma/tomato_env/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export TF_ENABLE_ONEDNN_OPTS=1

/home/nirma/tomato_env/bin/python train.py "$@"
