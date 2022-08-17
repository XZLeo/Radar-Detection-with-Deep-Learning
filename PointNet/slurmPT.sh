#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 8
#SBATCH --time 00:40:00
#SBATCH --mem-per-gpu 64G
#SBATCH --output /workspaces/%u/thesisdlradardetection/PointNet/Pnet_pytorch/log/hal_logs/testings/slurm_inference-%j-run.out
#SBATCH --partition ztest
#

echo "Full Dataset Testing (DBSCAN + PointNet)"
mkdir -p /workspaces/$USER/training/logs/

echo ""
echo "This job was started as: python3 -u $@"
echo ""
time=$(date "+%Y-%m-%d %H:%M:%S")

# PointNet Testing
singularity exec --nv --bind /workspaces/$USER:/workspace \
  --bind /staging/thesisradardetection:/RadarScenes \
  --pwd /workspace/thesisdlradardetection/ \
  --env PYTHONPATH=. \
  --cleanenv \
  --no-home \
  /workspaces/$USER/radar_detect.sif \
  python3 -u /workspace/thesisdlradardetection/PointNet/Pnet_pytorch/test_radar_semseg.py \
	  --batch_size 32 \
          --num_point 3097 \
	  --log_dir GPU_Jun29_23_jitter_noise\
	  --iou_thresh 0.3\
	  #--debug

