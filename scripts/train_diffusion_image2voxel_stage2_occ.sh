export WANDB_MODE=disabled
python -m occupancy.pipelines.train --model occ_transformer --data-dir '/workspace/occupancy/data/nuScenes-Occupancy-v0.1' \
    --cache-dir .cache/nuscenes --batch-size 2 --num-workers 8 --device cuda \
    --dtype bf16 --model-config config/diffusion_image2voxel.yml --grad-accum 32 --save-every 100 --weight-decay 1.0e-5 --lr 1.0e-4 \
    --warmup-steps 10 --num-classes 18
    
