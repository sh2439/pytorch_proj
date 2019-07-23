#!/usr/bin/env bash
#SBATCH -J vgg_face    # Job name
#SBATCH -o vgg_face.o%j    # Name of stdout output file (%j expands to jobId)
#SBATCH -e vgg_face.e%j    # Name of stderr output file
#SBATCH -N 1   # Total number of CPU nodes requested
#SBATCH -n 2   # Total number of CPU cores requrested
#SBATCH --mem=10G    # CPU Memory pool for all cores
#SBATCH -t 120:00:00    # Run time (hh:mm:ss)
#SBATCH --partition=default_gpu --gres=gpu:2   # Which queue to run on, and what resources to use
                                               # --partition=<queue> - Use the `<queue>` queue
                                               # --gres=gpu:1 - Use 1 GPU of any type
                                               # --gres=gpu:1080ti:1 - Use 1 GTX 1080TI GPU

# OPTIONAL: uncomment this if you're using an anaconda environment named <my_env_name>
# . /share/apps/anaconda3/5.2.0/etc/profile.d/conda.sh
#conda activate myenv2
 
echo "detection"

python train.py -c gender_vgg.config

