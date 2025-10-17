#!/bin/bash
# Name: Batch_parallel_processing.sh
# Created by: Jeremy Leitz
# Created on 9/20/2025
#Description: This script is to predict kinase targets using KIPP in parallel with gun parallel
#Usage: sbatch Batch_parallel_processing.sh <input.csv> <output_directory/name> <job number (note: must be multiple of 5; default:30)>

#SBATCH --job-name=Para_KIPP
#SBATCH --error=KIPP_%A_%a.err
#SBATCH --output=KIPP_%A_%a.out
#SBATCH --cpus-per-task=30
#SBATCH --ntasks=1

# Source the conda setup to make the `conda` command available.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kinasepred_JL 

Input="$1"
Output_DIR="$2"

# Make sure output directory exists
mkdir -p $Output_DIR

job_num=$(( $SLURM_CPUS_PER_TASK / 5 ))

# Run in parallel
tail -n +2 $Input | parallel -j $job_num --colsep ',' ~/Software/KIPP/parallel_predict.sh {1} {2} $Output_DIR
