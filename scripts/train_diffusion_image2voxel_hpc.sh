export TORCHDYNAMO_VERBOSE=1
export WANDB_CACHE_DIR=/scratch/$USER/.cache/wandb
export WANDB_MODE=disabled
python -m occupancy.pipelines.train --model autoencoderkl --data-dir /nuscenes/nuscenes \
    --cache-dir .cache/nuscenes --batch-size 4 --num-workers 16 --device cuda \
    --dtype bf16 --model-config config/diffusion_image2voxel.yml --grad-accum 8 --lr 1.0e-4 --num-classes 1

