export WANDB_MODE=disabled
python -m occupancy.pipelines.train --model diffusion3d --data-dir '/mnt/f/datasets/nuscenes/' \
    --cache-dir .cache/nuscenes --batch-size 1 --num-workers 8 --device cuda \
    --dtype bf16 --model-config config/diffusion_image2voxel.yml --grad-accum 1 --save-every 1000 --weight-decay 0.001 \
    --warmup-steps 0 --lr 1.0e-3
