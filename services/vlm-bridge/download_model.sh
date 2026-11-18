#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
exec /home/linux/YoloWebAgentStarter/.venv/bin/python -u -c "
from huggingface_hub import snapshot_download
p = snapshot_download('OpenGVLab/InternVL2-1B', local_dir='/home/linux/models/InternVL2-1B', max_workers=2)
print('DONE', p)
"
