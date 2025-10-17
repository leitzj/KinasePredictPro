#!/bin/bash
#calc_selectivity.sh
# Usage: calc_selectivity.sh <summary.csv> <output.csv> <target_gene_uniprot_id> <sort_metric>
# Sort_metric options: 'S_Score', 'Percentile_Rank', 'Partition_Index', 'Selectivity_Entropy', 'Normalized_Entropy', 'Num_Kinases_Tested'

Input_file="$1"
Output_file="$2"
Target_Kinase="$3"
Sort_Metric="${4:-S_Score}"
display_num="${5:-10}"

python3 ~/Software/KIPP/calc_selectivity.py "$Input_file" "$Output_file" "$Target_Kinase" "$Sort_Metric" "$display_num"
