#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --mem=128GB 
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=gen_voxel
#SBATCH --output=gen_voxel.out

singularity exec --nv \
        --overlay /scratch/zc2309/nuscenes.ext3:ro \
	    --overlay /scratch/$USER/containers/overlay.ext3:ro  \
	    /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
	    /bin/bash -c "source /ext3/env.sh; cd /scratch/zc2309/occupancy; bash scripts/gen_voxel.sh"