#!/bin/bash
#Here is a script to now run Kipp in a loop

Input=$1
Output=$2

#Make output directory, if needed
mkdir -p $Output

tail -n +2 "$Input" | while IFS=',' read -r smiles ID; do
	output_file="${Output}/${ID}_KIPP_Results.json"
	if [ -f "$output_file" ]; then
		echo "Output file alread exists for ${ID}, ${smiles}, skipping."
		continue
	fi

	echo "Processing ${ID}, smiles: ${smiles} in KIPP."
	~/Software/KIPP/kipp_predict.sh $smiles $ID $Output
	echo "Writing output to ${Output}/${ID}_KIPP_results.json"
done
