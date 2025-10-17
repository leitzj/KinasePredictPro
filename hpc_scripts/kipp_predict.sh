#!/bin/bash
#Script - Kipp_predict.sh
#Description: This is a wrapper to run a modified KIPP program.  In the original KIPP files can only be written out to the local directory. In predict_JL.py, I added a output destination.
#Written by Jeremy Leitz

SMILES="$1"
ID="$2"
OUTPUT_DIR="${3:-$PWD}"

#Move into the KIPP directory
cd ~/Software/KIPP/KinasePredictPro
out_file="${OUTPUT_DIR}/${ID}_KIPP_Results"
#Process using modified KIPP code
python predict_JL.py "$SMILES" 1 "$out_file" 

