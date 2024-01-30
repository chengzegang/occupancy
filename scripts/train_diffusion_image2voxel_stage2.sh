export WANDB_MODE=disabled
python -m occupancy.pipelines.train --model panoramic2voxel --data-dir '/mnt/f/datasets/nuscenes/' \
    --cache-dir .cache/nuscenes --batch-size 4 --num-workers 8 --device cuda \
    --dtype bf16 --model-config config/diffusion_image2voxel.yml --grad-accum 32 --save-every 100 --weight-decay 1.0e-1 --lr 1.0e-4 \
    --warmup-steps 10 --num-classes 18
    
