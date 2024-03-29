export WANDB_MODE=disabled
python -m occupancy.pipelines.train --model autoencoderkl --data-dir '/mnt/f/datasets/nuscenes/nuScenes-Occupancy-v0.1' \
    --cache-dir .cache/nuscenes --batch-size 1 --num-workers 8 --device cuda \
    --dtype bf16 --model-config config/diffusion_image2voxel.yml --grad-accum 16 \
    --betas 0.9 0.999 --weight-decay 0.1
