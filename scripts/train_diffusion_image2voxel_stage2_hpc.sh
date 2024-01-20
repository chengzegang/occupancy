export HF_HOME=/scratch/zc2309/.cache/huggingface
export WANDB_MODE=disabled
python -m occupancy.pipelines.train --model panoramic2voxel --data-dir /nuscenes/nuscenes \
    --cache-dir .cache/nuscenes --batch-size 6 --num-workers 16 --device cuda \
    --dtype bf16 --model-config config/diffusion_image2voxel.yml --grad-accum 16 --save-every 100