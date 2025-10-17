#!/bin/bash
#analyze_kipp_results.sh
#Written by Jeremy Leitz
#Usage: analyze_kipp_results.sh <results_directory> <output.csv> [score_threshold]

Results_Dir="$1"
Output_file="$2"

#Temporary file for all predictions
temp_file=$(mktemp)

#Create CSV header
echo "Compound_ID,Uniprot_ID,Gene_name,Model,Score" > "$Output_file"

#Process each JSON file in directory
for json_file in "$Results_Dir"/*_KIPP_Results.json; do
	compound_id=$(basename "$json_file" _KIPP_Results.json)

	#Parse json, filter by threshold, and output as CSV
	jq -r --arg id "$compound_id" '
		to_entries |  
		.[] | 
		[$id, .key, .value[0], .value[1]] |
		@csv
	' "$json_file" | while IFS=',' read -r compound uniprot model score; do
	#Remove quotes
	uniprot=$(echo "$uniprot" | tr -d '"')
	#Look up the gene name
	gene_name=$(grep $uniprot ~/Software/KIPP/uniprot_kinase_mapping.tsv | awk '{ print $2 }');
	[ -z "$gene_name" ] && gene_name="Unknown"
	
	gene_with_id="${gene_name}(${uniprot})"
	#Output the line
	echo "$compound,$gene_with_id,$score"
	done >> "$temp_file"

done

python3 - "$temp_file" "$Output_file" << 'PYTHON_SCRIPT'
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1], names=['Compound_ID', 'Gene', 'Score'])
pivot_df = df.pivot_table(index='Compound_ID', columns='Gene', values='Score', aggfunc='max')
pivot_df.to_csv(sys.argv[2])

print(f"Results written to {sys.argv[2]}")
print(f"Compounds: {len(pivot_df)}, Kinases: {len(pivot_df.columns)}")
PYTHON_SCRIPT

echo "Results written to $Output_file"

#Cleanup temp files
rm "$temp_file"
