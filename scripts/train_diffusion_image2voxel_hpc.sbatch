#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=128GB 
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=occ_vae
#SBATCH --output=occ_vae.out

singularity exec --nv \
        --overlay /scratch/zc2309/nuscenes.ext3:ro \
	    --overlay /scratch/$USER/containers/overlay.ext3:ro  \
	    /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
	    /bin/bash -c "source /ext3/env.sh; cd /scratch/zc2309/occupancy; bash scripts/train_diffusion_image2voxel_hpc.sh"