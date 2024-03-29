export WANDB_MODE=disabled
python -m occupancy.pipelines.train --model panoramic2voxel --data-dir '/mnt/f/Dropbox/nuscenes_archive/nuscenes' \
    --cache-dir .cache/nuscenes --batch-size 1 --num-workers 8 --device cuda \
    --dtype bf16 --model-config config/diffusion_image2voxel.yml --grad-accum 16 --save-every 100 --weight-decay 1.0e-5 --lr 2.0e-4 \
    --warmup-steps 10
    
