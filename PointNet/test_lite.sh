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
  --bind /workspaces/$USER/thesisdlradardetection/RS_data_sample:/RadarScenes \
  --pwd /workspace/thesisdlradardetection/ \
  --env PYTHONPATH=. \
  --cleanenv \
  --no-home \
  /workspaces/$USER/radar_detect.sif \
  python3 -u /workspace/thesisdlradardetection/PointNet/Pnet_pytorch/test_radar_semseg.py \
	  --test_dataset_path /RadarScenes/test_small \
	  --test_snippet_path ./static/test_small.txt \
    --dbscan_config_file /workspace/thesisdlradardetection/PointNet/Pnet_pytorch/config \
    --batch_size 1 \
    --num_point 4096 \
    --iou_thresh 0.5 \
	  --log_dir sample_training_server\
	  #--debug_size \
    #--idxs_snippets 0\
    #--num_snippets 1\
    #--debug_gt \
    #--plot_cluster \
    #--plot_labels \
    #--show_save_plots show\

