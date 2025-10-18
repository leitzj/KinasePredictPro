#!/bin/bash
#Script -Batch_Array_Prep.sh
#Written by Jeremy Leitz
#Description: This script prepares a input smiles file for processing in KIPP in an array.

#Usage
usage() {
	cat << EOF
Usage: -i <input CSV> -o <output_DIR> [-n SLURM JOB ARRAY NUMBER] [-c CPUs number to use for multiprocessing]

Arguments:
    -i Input CSV    CSV should be comma separated containing 2 columns: smiles,id
    -o Output DIR   Output directory
    -n Array_number SLURM Array number to use.  Note this is for chunking.  Each array will also multip process the files using cpus-per-task / 5 per process.
    -c CPUs to use  Number of cpus-per-task to use for multiprocessing via python (default: 30)
    -h              Show this message

EOF
	exit 1
}

#Default values
Cpu_num=30
Array_number=0

#Parse arguments:
while getopts i:o:n:c:h flag
do
	case "${flag}" in
		i) Input=${OPTARG};;
		o) Output_DIR=${OPTARG};;
		n) Array_number=${OPTARG};;
		c) Cpu_num=${OPTARG};;
		h) usage;;
		*) usage;;
	esac
done

#Check the input file exits
if [ ! -f "$Input" ]; then
	echo "ERROR: Could not find input file: $Input"
	exit 1
fi

#Check the output directory exits, if not, make it.
if [ ! -d "$Output_DIR" ]; then
	mkdir -p "$Output_DIR"
fi

#Check if the input file is comma delimited, if not change it.
if grep -q ' ' "$Input"; then
	echo "Spaces detected in input file, changing to commas"
	name=$(basename "$Input")
	sed 's/\ /,/g' "$Input" > "$Output_DIR/no_space_$name"
	Use_FILE="$Output_DIR/no_space_$name"
else
	Use_FILE="$Input"
fi

#Count the lines (excluding header)
if [[ "$Array_number" -gt 0 ]]; then
	total_lines=$(tail -n +2 "$Use_FILE" | wc -l)
	lines_per_chunk=$(( (total_lines + Array_number -1) / Array_number ))

	#Split into chunks
	echo "Splitting $Use_FILE: $total_lines lines into $Array_number chunks (~$lines_per_chunk lines each)"
	tail -n +2 "$Use_FILE" | split -l "$lines_per_chunk" -d -a 4 - "$Output_DIR/chunk_"
else
	echo "Removing header, but not spliting input file"
	tail -n +2 "$Use_FILE" > "$Output_DIR/chunk_0000"
fi

#Make final output dir 
mkdir -p "$Output_DIR/results"

#Echo the command to use
echo "=========================="
echo "  Preprocessing Finished  "
echo "=========================="
echo ""
echo "Ready to submit slurm script"
if [[ "$Array_number" -gt 0 ]]; then
	echo "sbatch --array=0-$((Array_number-1)) --cpus-per-task=$Cpu_num ~/Software/KIPP/KinasePredictPro/hpc_scripts/Batch_parallel_Array_processing.sh $Output_DIR $Output_DIR/results yes"
else
	echo "sbatch --cpus-per-task=$Cpu_num ~/Software/KIPP/KinasePredictPro/hpc_scripts/Batch_parallel_Array_processing.sh $Output_DIR $Output_DIR/results"
fi
echo ""
echo "Note: default cpus-per-task is 30, processing is 5 cpus per task, so change at will preferably to a number divisible by 5"
