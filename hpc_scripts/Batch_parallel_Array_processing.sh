#!/bin/bash
# Name: Batch_parallel_Array_processing.sh
# Created by: Jeremy Leitz
# Created on 9/20/2025
#Description: This script is to predict kinase targets using KIPP in parallel USING A SBATCH JOB ARRAY with python multiprocessing parallel
#Usage: sbatch Batch_parallel_processing.sh <input dir contain chunk file> <output_directory/name> <job number (note: must be multiple of 5; default:30)

#SBATCH --job-name=Para_KIPP
#SBATCH --error=KIPP_%A_%a.err
#SBATCH --output=KIPP_%A_%a.out
#SBATCH --cpus-per-task=30
#SBATCH --ntasks=1

#Cleanup function to run on exit
cleanup() {
	local exit_code=$?
	echo ""
	echo "========================"
	echo "Cleanup started at: $(date)"

	#Copy results from local to final destination if using local storage
	if [ "$USE_LOCAL_COPY" == "yes" ] && [ -d "$LOCAL_OUTPUT_DIR" ]; then
		#Added safety so we don't delete wrong folder
                if [ -n "$LOCAL_OUTPUT_DIR" ] && [ "$LOCAL_OUTPUT_DIR" != "/" ] && [[ "$LOCAL_OUTPUT_DIR" == /tmp/* ]]; then
			echo "Copying results from local storage to $Output_DIR..."
			rsync -av --info=progress2 "$LOCAL_OUTPUT_DIR/" "$Output_DIR/"
			rsync_status=$?

			if [ $rsync_status -eq 0 ]; then
				echo "Results copied successfully"
				#Remove local results after successful copy safely
				if [[ "$LOCAL_OUTPUT_DIR" == /tmp/KIPP_results_* ]]; then
					rm -r "$LOCAL_OUTPUT_DIR"
					echo "Local results $LOCAL_OUTPUT_DIR directory cleaned up"
				else
					echo "Warning: Unexpected LOCAL_OUTPUT_DIR path: $LOCAL_OUTPUT_DIR -skipping deletion"
				fi
			else
				echo "WARNING: rsync failed with status $rsync_status"
				echo "Local results remain at: $LOCAL_OUTPUT_DIR"
				exit_code=1
			fi
		else
			echo "WARNING: LOCAL_OUTPUT_DIR not set correctly or unsafe - skipping"
			exit_code=1
		fi
	fi

	#Cleanup KIPP local copy if this appears to be the last job
	if [ "$USE_LOCAL_COPY" == "yes" ] && [ -d "$KIPP_LOCAL" ]; then
		#Double check for safety
		if [ -n "$KIPP_LOCAL" ] && [ "$KIPP_LOCAL" != "/" ] && [[ "$KIPP_LOCAL" == /tmp/* ]]; then
			if [ -d "$KIPP_LOCAL" ]; then

				#Check if any other KIPP jobs are running on this node
				RUNNING_JOBS=$(squeue -h -w $(hostname) -n KIPP_Opt -t RUNNING | wc -l)

				if [ "$RUNNING_JOBS" -le 1 ]; then
					echo "No other KIPP jobs on $(hostname), cleaning up KIPP local copy..."
					if [[ "$KIPP_LOCAL" == /tmp/KIPP_shared ]]; then
						rm -r "$KIPP_LOCAL"
						rm "$LOCK_FILE"
						echo "KIPP local copy cleaned up"
					else
						echo "WARNING: Unexpected KIPP_LOCAL path: $KIPP_LOCAL - skipping cleanup, if this keeps happening, need to modify hard coded /tmp/KIPP_shared path"
					fi
				else
					echo "Other KIPP jobs still running on $(hostname), keeping KIPP local copy"
				fi
			fi
		else
			echo "WARNING: KIPP_LOCAL not set correctly or unsafe - skipping KIPP cleanup"
		fi
	fi

	echo "Cleanup completed at: $(date)"
	echo "==============================="
	exit $exit_code
}

#Set trap to run cleanup on exit
trap cleanup EXIT INT TERM

# Source the conda setup to make the `conda` command available.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kinasepred_JL

# Arguement parsing
Input_DIR="$1"
Output_DIR="$2" #Where results will end up
USE_LOCAL_COPY="${3:-no}" #Default is no

# Make sure output directory exists
mkdir -p "$Output_DIR"

#Calc thread number
THREADS_PER_PROCESS=5
job_num=$(( SLURM_CPUS_PER_TASK / THREADS_PER_PROCESS  ))

#Check job num
if [ "$job_num" -lt 1 ]; then
	job_num=1
fi

#Export for python script to use
export THREADS_PER_PROCESS

#Get chunk file
chunk_num=$(printf '%04d' $((SLURM_ARRAY_TASK_ID)))
Input="$Input_DIR/chunk_$chunk_num"

#Check that Chunk file exists
if [[ ! -f "$Input" ]]; then
	echo "Could not find chunk file to process: $Input"
	exit 1
fi

#Debug
echo "================================"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Chunk number: $chunk_num"
echo "Input file: $Input"
echo "Job parallelism: $job_num"
#echo "KIPP directory used: $KIPP_DIR"
echo "Final output: $Output_DIR"
echo "================================"

# ============== OPTIONAL: Copy to local storage ============="
if [ "$USE_LOCAL_COPY" = "yes" ]; then
	#Create a local copy
	KIPP_LOCAL="/tmp/KIPP_shared"
	LOCK_FILE="/tmp/KIPP_copy.lock"

	#Use flock to ensure only a single copy is copied by a job
	(
		flock -x 200
		if [ ! -d "$KIPP_LOCAL/KinasePredictPro" ]; then
			echo "First job on $(hostname) - copying KIPP to local storage..."
			mkdir -p "$KIPP_LOCAL"
			rsync -a ~/Software/KIPP/KinasePredictPro/ "$KIPP_LOCAL/KinasePredictPro/"
			echo $(ls "$KIPP_LOCAL/KinasePredictPro/models")
			echo "Copy complete at $(date)"
		else
			echo "KIPP already copied to $(hostname), reusing..."
		fi
	) 200>"$LOCK_FILE"

	KIPP_DIR="$KIPP_LOCAL/KinasePredictPro"
else
	KIPP_DIR=~/Software/KIPP/KinasePredictPro
fi
# ============================================================"
#Setup local output directory
if [ "$USE_LOCAL_COPY" = "yes" ]; then
	#Use local temp storage for writing results
	LOCAL_OUTPUT_DIR="/tmp/KIPP_results_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

	#Safety validate path

	if [[ "$LOCAL_OUTPUT_DIR" != /tmp/KIPP_results_* ]] || [ "$LOCAL_OUTPUT_DIR" = "/" ]; then
		echo "ERROR: LOCAL_OUTPUT_DIR has unsafe value: $LOCAL_OUTPUT_DIR"
		exit 1
	fi

	mkdir -p "$LOCAL_OUTPUT_DIR"
	if [ ! -d "$LOCAL_OUTPUT_DIR" ]; then
		echo "ERROR: Failed to create local output directory: $LOCAL_OUTPUT_DIR"
		exit 1
	fi

	echo "Results will be written to local storage: $LOCAL_OUTPUT_DIR"
	echo "Then copied back to: $Output_DIR"
	WORK_OUTPUT_DIR="$LOCAL_OUTPUT_DIR"
else
	#Write directly to final destination (not recommended for hight-throughput)
	echo "Results will be written directly to: $Output_DIR"
	WORK_OUTPUT_DIR="$Output_DIR"
fi

#==============================================================================


#Run batch script from appropriate directory
cd "$KIPP_DIR" || {
	echo "ERROR: Could not change to KIPP directory: $KIPP_DIR" >&2
	exit 1
}

#Count molecules in chunk
NUM_MOLECULES=$(wc -l < "$Input")
echo "Processing $NUM_MOLECULES molecule with $job_num workers"
echo "Expected: Each worker loads models once, then processes ~$((NUM_MOLECULES/job_num)) molecules"
echo ""

echo "Started at :$(date)"
python ~/Software/KIPP/KinasePredictPro/predict_JL.py "$Input" 1 "$WORK_OUTPUT_DIR" "$job_num"
exit_code=$?
echo "Finished at: $(date)"

exit $exit_code
