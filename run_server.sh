#!/bin/bash
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
# export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
# export ASCEND_RT_VISIBLE_DEVICES=2,3,10,11,12,13,14,15
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_TORCH_PROFILER_DIR="./vllm_profile"
export VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=False
export VLLM_TORCH_PROFILER_WITH_STACK=False
export CPU_AFFINITY_CONF=2
#export TASK_QUEUE_ENABLE=2
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export DIFFUSION_ATTENTION_BACKEND="FLASH_ATTN"

export ASCEND_LAUNCH_BLOCKING=0

vllm serve "/data2/weight/HunyuanImage-3.0-Instruct/" --omni --port "8031" \
    --log-stats \
    --stage-configs-path "vllm_omni/platforms/npu/stage_configs/hunyuan_image3_t2i.yaml" \
    --additional-config='{
        "multistream_overlap_shared_expert": "True",
        "refresh": "True"
    }'
    # --cache-backend cache_dit