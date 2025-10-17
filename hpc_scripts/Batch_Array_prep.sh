#!/bin/bash
#Script -Batch_Array_Prep.sh
#Written by Jeremy Leitz
#Description: This script prepares a input smiles file for processing in KIPP in an array.

#Arguments
Input="$1"  #CSV file conatining smiles,id
Output_DIR="$2"
Array_number="$3"

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
	sed 's/\ /,/' "$Input" > "$Output_DIR/no_space_$Input"
	Use_FILE="$Output_DIR/no_space_$Input"
else
	Use_FILE="$Input"
fi

#Count the lines (excluding header)
total_lines=$(tail -n +2 "$Use_FILE" | wc -l)
lines_per_chunk=$(( (total_lines + Array_number -1) / Array_number ))

#Split into chunks
echo "Splitting $Use_FILE: $total_lines lines into $Array_number chunks (~$lines_per_chunk lines each)"
tail -n +2 "$Use_FILE" | split -l "$lines_per_chunk" -d -a 4 - "$Output_DIR/chunk_"

#Echo the command to use
echo "=========================="
echo "  Preprocessing Finished  "
echo "=========================="
echo ""
echo "Ready to submit slurm script"
echo "sbatch --array=0-$((Array_number-1)) --cpus-per-task=30 ~/Software/KIPP/Batch_parallel_Array_processing.sh $Output_DIR"
echo "Note: default cpus-per-task is 30, processing is 5 cpus per task, so change at will preferably to a number divisible by 5"
