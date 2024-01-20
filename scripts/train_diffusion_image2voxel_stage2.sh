export WANDB_MODE=disabled
python -m occupancy.pipelines.train --model panoramic2voxel --data-dir '/mnt/f/datasets/nuscenes/' \
    --cache-dir .cache/nuscenes --batch-size 1 --num-workers 8 --device cuda \
    --dtype bf16 --model-config config/diffusion_image2voxel.yml --grad-accum 8 --save-every 100 --weight-decay 0.001
    