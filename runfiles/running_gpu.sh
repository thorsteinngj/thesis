#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J NNet_train_aug_gpu
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1
### -- specify that the cores must be on the same host -- 
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[gpu32gb]"
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s171945@student.dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Neural2_gpu_%J.out 
#BSUB -e Neural2_Error_gpu_%J.err 

# -- commands you want to execute -- 
# 
nvidia-smi
# Load the CUDA module
#module load cuda/9.1

source ~/environments/gravestones_env/bin/activate
cd ~/environments/gravestones_env/Code/Mask_RCNN/samples/graves 
module load python3/3.6.2 
module load tensorflow/1.5-gpu-python-3.6.2
python3 graves_aug.py train --dataset=/zhome/2e/9/124284/environments/gravestones_env/Code/Mask_RCNN/samples/graves/dataset_new --weights=/work1/s171945/logs/graves20191120T1801/mask_rcnn_graves_0060.h5

