export WANDB_MODE=disabled
python -m occupancy.pipelines.train --model autoencoderkl_2d --data-dir '/mnt/f/datasets/nuscenes/' \
    --cache-dir .cache/nuscenes --batch-size 1 --num-workers 8 --device cuda \
    --dtype fp32 --model-config config/diffusion_image2voxel.yml --grad-accum 1 --num-classes 1 --lr 1.0e-4 --weight-decay 1.0e-5 \
    --total-epochs 1