#!/bin/bash
# Name: Batch_parallel_Array_processing.sh
# Created by: Jeremy Leitz
# Created on 9/20/2025
#Description: This script is to predict kinase targets using KIPP in parallel USING A SBATCH JOB ARRAY with gun parallel
#Usage: sbatch Batch_parallel_processing.sh <input.csv> <output_directory/name> <job number (note: must be multiple of 5; default:30)>

#SBATCH --job-name=Para_KIPP
#SBATCH --error=KIPP_%A_%a.err
#SBATCH --output=KIPP_%A_%a.out
#SBATCH --cpus-per-task=30
#SBATCH --ntasks=1

# Source the conda setup to make the `conda` command available.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kinasepred_JL 

Input_DIR="$1"
FINAL_OUTPUT_DIR="$2" #Where results will end up
USE_LOCAL_COPY="${3:-no}"

#Create LOCAL temp directory for the job
LOCAL_OUTPUT="/tmp/KIPP_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$LOCAL_OUTPUT"

# Make sure output directory exists
mkdir -p $FINAL_OUTPUT_DIR

#Calc number of jobs
job_num=$(( $SLURM_CPUS_PER_TASK / 5 ))

#Check job num
if [ "$job_num" -lt 1 ]; then
	job_num=1
fi

chunk_num=$(printf '%04d' $((SLURM_ARRAY_TASK_ID)))
Input="$Input_DIR/chunk_$chunk_num"

# ============== OPTIONAL: Copy to local storage ============="
if [ "$USE_LOCAL_COPY" = "yes" ]; then
	KIPP_LOCAL="/tmp/KIPP_shared"
	LOCK_FILE="/tmp/KIPP_copy.lock"

	#Use flock to ensure

#Debug
echo "================================"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Chunk number: $chunk_num"
echo "Input file: $Input"
echo "Job parallelism: $job_num"
echo "Local output: $LOCAL_OUTPUT"
echo "Final output: $FINAL_OUTPUT_DIR"
echo "================================"

# Run in parallel
cat "$Input" | parallel -j "$job_num" --colsep ',' --timeout 600 --halt soon,fail=10 --joblog "$LOCAL_OUTPUT/joblog_${chunk_num}.txt" ~/Software/KIPP/parallel_predict.sh {1} {2} "$LOCAL_OUTPUT"

parallel_exit=$
exit $parallel_exit
