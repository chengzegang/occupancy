export WANDB_MODE=disabled
python -m occupancy.pipelines.train --model diffusion3d --data-dir '/mnt/f/datasets/nuscenes/' \
    --cache-dir .cache/nuscenes --batch-size 2 --num-workers 8 --device cuda \
    --dtype bf16 --model-config config/diffusion_image2voxel.yml --grad-accum 8 --save-every 100 --weight-decay 0.1 --warmup-steps 1000
