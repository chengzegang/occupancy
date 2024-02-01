export HF_HOME=/scratch/zc2309/.cache/huggingface
python -m occupancy.pipelines.train --model occ_transformer --data-dir /nuscenes/nuscenes \
    --cache-dir .cache/nuscenes --batch-size 8 --num-workers 16 --device cuda \
    --dtype fp32 --model-config config/diffusion_image2voxel.yml --grad-accum 8 --save-every 100 --weight-decay 1.0e-5 \
    --num-classes 18 --lr 1.0e-5