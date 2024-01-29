conda create -n occupancy python=3.11
conda install -n occupancy pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda run -n occupancy pip install -e .
mkdir data
cd data
wget https://www.dropbox.com/scl/fi/h9ro2ryc9zb7ffgt7kur8/nuScenes-Occupancy-v0.1.zip?rlkey=bfvy5bhb6mc1a7rlh05xnxa35 -O nuScenes-Occupancy-v0.1.zip
unzip nuScenes-Occupancy-v0.1.zip
cd ..
export WANDB_MODE=disabled
python -m occupancy.pipelines.train --model autoencoderkl --data-dir data/nuScenes-Occupancy-v0.1 \
    --cache-dir .cache/nuscenes --batch-size 1 --num-workers 8 --device cuda \
    --dtype bf16 --model-config config/diffusion_image2voxel.yml --grad-accum 1 --num-classes 18 --lr 1.0e-4 --weight-decay 1.0e-3