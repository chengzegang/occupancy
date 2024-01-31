export WANDB_MODE=disabled
python -m occupancy.pipelines.train --model occ_transformer --data-dir '/mnt/f/datasets/nuscenes/' \
    --cache-dir .cache/nuscenes --batch-size 1 --num-workers 8 --device cuda \
    --dtype bf16 --model-config config/diffusion_image2voxel.yml --grad-accum 8 --save-every 100 --weight-decay 1.0e-5 --lr 3.0e-5 \
    --warmup-steps 10 --num-classes 18
    
