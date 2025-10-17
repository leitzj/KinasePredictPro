#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from scipy.stats import entropy

def gini_coefficient(scores):
    sorted_scores = np.sort(scores)
    n = len(scores)
    cumsum = np.cumsum(sorted_scores)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0

input_file = sys.argv[1]
output_file = sys.argv[2]
target_kinase = sys.argv[3]
Sort_Metric = sys.argv[4] if len(sys.argv) > 4 else 'S_Score'
display_num = int(sys.argv[5]) if len(sys.argv) > 5 else 10


# Read the summary csv from analyze.csv
df = pd.read_csv(input_file, index_col=0)

# Check if target Kinases exists in columns
if target_kinase not in df.columns:
    print(f"Error: {target_kinase} not found in columns")
    print(f"Available kinase: {', '.join(df.columns[:10])}...")
    sys.exit(1)

# Sort dictionary 
sort_directions = {
        'S_Score': False,
        'Percentile_Rank': False,
        'Partition_Index': False,
        'Selectivity_Entropy': True,
        'Normalized_Entropy': True,
        'Odds_Ratio': False,
        'Gini_Coefficient': False,
        'Num_Kinases_Tested': False
        }
# Validate sort metric
if Sort_Metric not in sort_directions:
    print(f"Warning: Unknown metric '{Sort_Metric}', defaulting to S_Score")
    print(f"Valid metrics: {', '.join(sort_directions.keys())}")
    Sort_Metric = 'S_Score'

results = []

for compound in df.index:
    scores = df.loc[compound].dropna()

    if len(scores) == 0:
        continue

    target_score = scores.get(target_kinase, np.nan)

    if pd.isna(target_score):
        continue

    # Remove target from other scoress for comparison
    other_scores = scores.drop(target_kinase)

    # S-Score: fraction of kinases with score below target
    s_score = (other_scores < target_score).sum() / len(other_scores) if len(other_scores) > 0 else np.nan

    #Percentile rank
    percentile = (other_scores < target_score).sum() / len(scores) * 100

    #Partition Index: target score / sum of all scores
    partition_index = target_score / scores.sum() if scores.sum() > 0 else np.nan

    #Selectivity Entropy: lower = more selective
    #Normalize scores to probs
    prob_scores = scores / scores.sum() if scores.sum() > 0 else scores
    selectivity_entropy = entropy(prob_scores)

    #Max entropy for this number of kinases (uniform distribution)
    max_entropy = np.log(len(scores))
    normalized_entropy = selectivity_entropy / max_entropy if max_entropy > 0 else np.nan
    
    #Calculate odds for target kinase
    target_odds = target_score / (1 - target_score) if target_score < 1 else np.inf

    #Calculate mean odds for other kinases
    other_odds = other_scores / (1 - other_scores)
    median_other_odds = other_odds.replace([np.inf, -np.inf], np.nan).median()

    #Odds ratio
    odds_ratio = target_odds / median_other_odds if median_other_odds > 0 else np.nan
    
    #Gini coefficient for all kinases (including target)
    gini = gini_coefficient(scores.values)

    results.append({
        'Compound_ID': compound,
        f'{target_kinase}_Score': target_score,
        'S_Score': s_score,
        'Percentile_Rank': percentile,
        'Partition_Index': partition_index,
        'Selectivity_Entropy': selectivity_entropy,
        'Normalized_Entropy': normalized_entropy,
        'Odds_Ratio': odds_ratio,
        'Gini_Coefficient': gini,
        'Num_Kinases_Tested': len(scores)
        })

result_df = pd.DataFrame(results)
result_df = result_df.sort_values(Sort_Metric, ascending=sort_directions[Sort_Metric])
result_df.to_csv(output_file, index=False)


print(f"Selectivity metrics written to {output_file}")
print(f"\nTop {display_num} most selective compounds for {target_kinase}:")
print(result_df.head(display_num)[['Compound_ID', f'{target_kinase}_Score', f'{Sort_Metric}', 'S_Score', 'Percentile_Rank', 'Partition_Index', 'Selectivity_Entropy', 'Normalized_Entropy', 'Odds_Ratio', 'Gini_Coefficient']])

