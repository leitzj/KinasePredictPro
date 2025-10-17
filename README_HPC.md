# HPC Batch Processing for KinasePredictPro (KIPP)

This fork adds high-performance computing (HPC) capabilities to KinasePredictPro for processing large libraries in parallel using SLURM clusters.

##Modifications

###Problem
The original code accesses the same models directory causing I/O bottleneck when running hundreds of parallel jobs.

### Solution
**Batch processing mode**: Process chunk of SMILES (Each SLURM array gets 1 chunk)
**Model pre-loading**: Load models once per job, share across workers (Each SLURM job copies models to temp file, deletes after)
**Python muliprocessing**: Switched from parallel to python internal multiprocessing with nesteed parallelism 

## Quickstart

## Original project
For General KIPP usage and citation inforamtion see [README.md](README.md)

#Author
Jeremy Leitz - HPC modifications 2025 
