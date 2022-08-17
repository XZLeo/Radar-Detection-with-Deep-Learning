#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 8
#SBATCH --time 168:00:00
#SBATCH --mem-per-gpu 64G
#SBATCH --output /workspaces/%u/thesisdlradardetection/PointNet/Pnet_pytorch/log/hal_logs/trainings/slurm_train-%j-run.out
#SBATCH --partition zprodlow
#

echo "Full Training test"
mkdir -p /workspaces/$USER/training/logs/

echo ""
echo "This job was started as: python3 -u $@"
echo ""
time=$(date "+%Y-%m-%d %H:%M:%S")

# PointNet Training
singularity exec --nv --bind /workspaces/$USER:/workspace \
  --bind /staging/master_theses/thesisradardetection:/RadarScenes \
  --pwd /workspace/thesisdlradardetection/ \
  --env PYTHONPATH=. \
  --cleanenv \
  --no-home \
  /workspaces/$USER/radar_detect.sif \
  python3 -u /workspace/thesisdlradardetection/PointNet/Pnet_pytorch/train_radar_semseg_msg.py \
  	  --model pnet2_radar_semseg_msg \
	  --batch_size 64 \
          --learning_rate 0.0065 \
          --npoint 4096 \
          --step_size 15\
          --lr_decay 0.80\
	  --decay_rate 0.0\
	  --epoch 65\
	  --log_dir new_full_training\
	  --data_aug None \
	  --jitter_data
	  #--debug

#EOF

