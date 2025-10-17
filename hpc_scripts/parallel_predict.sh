#!/bin/bash

#This is a wrapper to run KIPP
SMILES="$1"
ID="$2"
OUTPUT_DIR="${3:-$PWD}" 

out_file="${OUTPUT_DIR}/${ID}_KIPP_Results"

#Safety check: ensure ID is not empty
if [ -z "$ID" ]; then
	echo "ERROR: ID parameter is empty" >&2
	exit 1
fi

if [ -f "$out_file.json" ]; then
	echo "Output for ${ID} already exists, skipping"
	exit 0
fi

#Move into the KIPP directory
cd ~/Software/KIPP/KinasePredictPro || {
	echo "ERROR: could not change to KIPP directory" >&2
	exit 1
}

echo "Processing ${ID}: ${SMILES}"
timeout 600 python predict_JL.py "$SMILES" 1 "$out_file"
exit_code=$?

#Check exit status
if [ $exit_code -eq 124 ]; then
	echo "ERROR: ${ID} timed out after 600 seconds" >&2
	#Clean up partial output if exits
	rm -f "${out_file}.json" "${out_file}"* 2>/dev/null 
	exit 1
elif [ $exit_code -eq 0 ]; then
	#Verify output was actually created
	if [ -f "$out_file.json" ]; then
		echo "Successfully processed ${ID}"
		exit 0
	else
		echo "ERROR: ${ID} completed but output file not found" >&2
		exit 1
	fi
else
	echo "ERROR: failed to process ${ID} (exit code: $exit_code)" >&2
	#Clean up partial output
	if [ -n "$out_file" ]; then
		rm -f "${out_file}.json" "${out_file}"* 2>/dev/null
	fi
	exit 1
fi
