export HF_HOME=/scratch/zc2309/.cache/huggingface
python -m occupancy.pipelines.train --model panoramic2voxel --data-dir /nuscenes/nuscenes \
    --cache-dir .cache/nuscenes --batch-size 4 --num-workers 16 --device cuda \
    --dtype bf16 --model-config config/diffusion_image2voxel.yml --grad-accum 4 --save-every 100 --weight-decay 1.0e-5 \
    --num-classes 18