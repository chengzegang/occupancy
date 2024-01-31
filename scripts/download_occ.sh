conda create -n occupancy python=3.11 -y --solver libmamba
conda install -n occupancy pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y --solver libmamba
conda run -n occupancy pip install -e .
mkdir data
cd data
wget https://www.dropbox.com/scl/fi/h9ro2ryc9zb7ffgt7kur8/nuScenes-Occupancy-v0.1.zip?rlkey=bfvy5bhb6mc1a7rlh05xnxa35 -O nuScenes-Occupancy-v0.1.zip
unzip nuScenes-Occupancy-v0.1.zip
cd ..