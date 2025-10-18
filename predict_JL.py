#Modified KIPP by Jeremy Leitz
#This modified version of predict.py allows for an output to a provided desitnation
#This version loads models once per worker.

import multiprocessing
import pandas as pd
import numpy as np
import json
import time
import sys
import warnings
from pathlib import Path
import os
from functools import partial

# Set thread counts BEFORE importing ML libraries
# This ensures each process uses the correct number of threads
THREADS_PER_PROCESS = int(os.environ.get('THREADS_PER_PROCESS', '5'))
os.environ['OMP_NUM_THREADS'] = str(THREADS_PER_PROCESS)
os.environ['OPENBLAS_NUM_THREADS'] = str(THREADS_PER_PROCESS)
os.environ['MKL_NUM_THREADS'] = str(THREADS_PER_PROCESS)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(THREADS_PER_PROCESS)
os.environ['NUMEXPR_NUM_THREADS'] = str(THREADS_PER_PROCESS)
warnings.filterwarnings('ignore')

# PyTorch and TensorFlow thread settings
try:
    import torch
    torch.set_num_threads(THREADS_PER_PROCESS)
    torch.set_num_interop_threads(1) # Only use threads within operations
except ImportError:
    pass

try:
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(THREADS_PER_PROCESS)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except ImportError:
    pass

from chemprop.train import predict
from chemprop.data import MoleculeDataLoader, MoleculeDatapoint, MoleculeDataset
from deepchem.data import NumpyDataset
from voting_fusion_use import get_predict_helper
from ofa_smi_eval import featurizer_ofa, load_model_ofa, predict_ofa

# ============== HARD CODED PATHS =================
DATA_DIR = "./KIPP_data"
VOTING_CSV = f"{DATA_DIR}/voting_accor.csv"
INFO_CSV = f"{DATA_DIR}/info_best.csv"
# =================================================

#Global variables for worker processes (loaded once per worker)
_worker_models = None
_worker_model_type = None
_worker_model_info = None

def init_worker(model_type, model_info_data):
    """
    Initialze worker process by loading all models only one time.
    This runs once per worker at startup.
    """
    global _worker_models, _worker_model_type, _worker_model_info

    _worker_model_type = model_type
    _worker_model_info = model_info_data
    _worker_models = {}

    print(f"[Worker {os.getpid()}] Loading models... this may take a minute or two")
    start_time = time.time()

    try:
        if model_type == 0:
            #Use voting model
            for idx, (target_name, seed) in enumerate(model_info_data):
                try:
                    #Preload models if possible (depeneds on get_predict_helper implementation)
                    #For now load on-demand but cache
                    _worker_models[target_name] = { 'seed' : seed, 'model': None}
                except Exception as e:
                    print(f"[Worker {os.getpid()}] Warning: Could not pre-load {target_name}: {e}")
        else:
            #Best models - load all best models
            for uniprot_id, method_name, seed, hyper in model_info_data:
                try:
                    algorithm_name = method_name.split('_', 1)[0]
                    algorithm_name = method_name.split('_', 1)[1] if algorithm_name == 'Graph' else algorithm_name
                    featurizer_name = method_name.split('_',1)[1] if method_name not in ['FPGNN', 'Chemprop'] else None

                    featurizer = featurizer_ofa(featurizer_name) if featurizer_name else None
                    model = load_model_ofa(uniprot_id, algorithm_name, featurizer_name, hyper, seed)

                    _worker_models[uniprot_id] = {
                            'model': model,
                            'method_name': method_name,
                            'featurizer': featurizer,
                            'algorithm_name': algorithm_name,
                            'featurizer_name': featurizer_name,
                            'seed': seed
                            }
                except Exception as e:
                    print(f"[Worker {os.getpid()}] Warning: Could not load module for {uniprot_id}: {e}")
        elapsed = time.time() - start_time
        print(f"[Worker {os.getpid()}] Models loaded in {elapsed:.1f}s")

    except Exception as e:
        print(f"[Worker {os.getpid()}] Error during initialization: {e}")
        raise

def predict_with_preloaded_models(smiles, uniprot_id, model_info):
    """Use pre-loaded models from worker to make predictions"""

    method_name = model_info['method_name']
    model = model_info['model']
    featurizer = model_info['featurizer']
    algorithm_name = model_info['algorithm_name']
    featurizer_name = model_info['featurizer_name']
    seed = model_info['seed']

    if method_name == 'FPGNN':
        predict_result = float(model([smiles])[0][0])
    elif method_name == 'Chemprop':
        predict_result = float(
                np.array(
                    predict(
                        model,
                        MoleculeDataLoader(MoleculeDataset([MoleculeDatapoint(smiles=[smiles])]), 
                            batch_size=1,
                            num_workers=0
                            )
                        )
                ).squeeze()
        )
    else:
        if isinstance(featurizer, list):
            test_dataset = NumpyDataset(X=np.concatenate([
                NumpyDataset(X=featurizer[0].featurize([smiles])).X,
                NumpyDataset(X=featurizer[1].featurize([smiles])).X
                ], axis=1))
        else:
            test_dataset = NumpyDataset(X=featurizer.featurize([smiles]))

        predict_result = float(
                predict_ofa(
                    uniprot_id,
                    algorithm_name,
                    featurizer_name,
                    model,
                    test_dataset=test_dataset,
                    seed2=seed
                ).flatten()
         )

    return {uniprot_id: [method_name, predict_result]}



def process_single_smiles(args):
    """
    Worker function for processing one SMILES
    Uses preloaded models from global _worker_models.
    """
    smiles, compound_id, output_dir = args
    
    #Check if the output already exists
    output_file = f"{output_dir}/{compound_id}_KIPP_Results.json"
    if os.path.exists(output_file):
        print(f"Output for {compound_id} already exists, skipping")
        return {
                'id': compound_id,
                'smiles': smiles,
                'status': 'skipped'
                }

    try:
        result = {}

        if _worker_model_type == 0:
            #Voting models
            for target_name, model_data in _worker_models.items():
                try:
                    seed = model_data['seed']
                    pred = get_predict_helper(smiles, target_name, seed)
                    result.update(pred)
                except Exception as e:
                    print(f"Error predicting {target_name} for {compound_id}: {e}")
        else:
            #Best models -use pre-loaded models
            for uniprot_id, model_info in _worker_models.items():
                try:
                    pred = predict_with_preloaded_models(smiles, uniprot_id, model_info)
                    result.update(pred)
                except Exception as e:
                    print(f"Error predicting {uniprot_id} for {compound_id}: {e}")

        return {
                'id': compound_id,
                'smiles': smiles,
                'predictions': result,
                'status': 'success'
                }

    except Exception as e:
        return {
                'id': compound_id,
                'smiles': smiles,
                'error': str(e),
                'status': 'failed'
                }

def kipp_predict_batch(input_csv, model_type, output_dir, num_workers=None):
    """Process a batch of SMILES from CSV file
    Args:
        input_csv: Path to CSV with SMILES,ID (no header)
        model_type: 0(voting) or 1(best models)
        output_dir: Directory to write result JSON files
        num_workers: Number of parallel workers (default: cpu_count)
    """
    
    #Read input CSV
    df = pd.read_csv(input_csv, header=None, names=['smiles', 'id'])
    
    if model_type == 0:
        voting_df = pd.read_csv(VOTING_CSV, encoding='utf-8', low_memory=False)
        model_info_data = list(zip(voting_df['name'].tolist(), voting_df['seed'].tolist()))
        print(f"Loaded metadata for {len(model_info_data)} voting models")
    else:
        info_df = pd.read_csv(INFO_CSV).set_index('uniprot_id')
        model_info_data = [
                (index, row[0], row[1], eval(row[2]) if pd.notna(row[2]) else None)
                    for index, row in info_df.iterrows()
                    ]
        print(f"Loaded metadata for {len(model_info_data)} best models")
    
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    print(f"Processing {len(df)} compounds with {num_workers} workers...")
    print(f"Each worker will load models once, then process ~{len(df)//num_workers} compounds")
    
    #Prepare arguments for parallel processing
    args_list = [
        (row['smiles'], row['id'], output_dir)
            for _, row in df.iterrows()
            ]
    
    #reate output directory
    os.makedirs(output_dir, exist_ok=True)
    
    #Create worker pool with initialization
    start_time = time.time()
    
    with multiprocessing.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(model_type, model_info_data)
        ) as pool:
        #Process with progress tracking
        results = []
        for i, result in enumerate(pool.imap_unordered(process_single_smiles, args_list), 1):
            results.append(result)
            if i % 10 == 0 or i == len(args_list):
                elapsed = time.time() - start_time
                rate = i /elapsed if elapsed > 0 else 0
                eta = (len(args_list) - i)/ rate if rate > 0 else 0
                print(f"Progress: {i}/{len(args_list)} ({(100*i/len(args_list)):.1f}%) - "
                    f"{rate:.2f} compounds/sec - ETA: {(eta/60):.1f} min")

    #Save individual results
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for result in results:
        output_file = f"{output_dir}/{result['id']}_KIPP_Results.json"
        with open(output_file, 'w') as f:
            if result['status'] == 'success':
                json.dump(result['predictions'], f, indent = 2)
                success_count += 1
            elif result['status'] == 'skipped':
                skipped_count += 1
            else:
                json.dump(result, f, indent=2)
                failed_count += 1
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Completed in {total_time/60:.1f} minutes ({total_time:.1f}s)")
    print(f"Success: {success_count}, Failed: {failed_count}, Skipped: {skipped_count}")
    print(f"Average: {total_time/len(df):.2f} seconds per compound")
    print(f"{'='*60}")

    if failed_count > 0:
        print("\nFailed compounds:")
        for result in results:
            if result['status'] == 'failed':
                print(f"  {result['id']}: {result.get('error', 'unknown error')}")
    
    return results


#Currently not used
def specific_predict(smiles, uniprot_id, method_name, seed=None, hyper=None):
    """Make prediction for a single SMILES using a specific model"""
    algorithm_name = method_name.split('_', 1)[0]
    algorithm_name = method_name.split(
        '_', 1)[1] if algorithm_name == 'Graph' else algorithm_name
    featurizer_name = method_name.split(
        '_',
        1)[1] if method_name != 'FPGNN' and method_name != 'Chemprop' else None
    featurizer = featurizer_ofa(featurizer_name) if featurizer_name else None
    model = load_model_ofa(uniprot_id, algorithm_name, featurizer_name, hyper,
                           seed)
    if method_name == 'FPGNN':
        predict_result = float(model([smiles])[0][0])
    elif method_name == 'Chemprop':
        predict_result = float(
            np.array(
                predict(
                    model,
                    MoleculeDataLoader(MoleculeDataset(
                        [MoleculeDatapoint(smiles=[smiles])]),
                                       batch_size=1,
                                       num_workers=0))).squeeze())
    else:
        if isinstance(featurizer, list):
            test_dataset = NumpyDataset(X=np.concatenate([
                NumpyDataset(X=featurizer[0].featurize([smiles])).X,
                NumpyDataset(X=featurizer[1].featurize([smiles])).X
            ],
                                                         axis=1))
        else:
            test_dataset = NumpyDataset(X=featurizer.featurize([smiles]))
        predict_result = float(
            predict_ofa(uniprot_id,
                        algorithm_name,
                        featurizer_name,
                        model,
                        test_dataset=test_dataset,
                        seed2=seed).flatten())
    return {uniprot_id: [method_name, predict_result]}

#Currently not used
def kipp_predict(smiles: str, model_type: int):
    """
    Run KIPP prediction for a single SMILES
    This loads models fresh for each SMILES (original behavior)
    """
    assert model_type in range(0, 2)
    #pool = multiprocessing.Pool()
    result = {}
    if model_type == 0:
        voting_df = pd.read_csv(VOTING_CSV,
                                encoding='utf-8',
                                low_memory=False)
        for target_name, seed in zip(voting_df['name'].tolist(),
                                     voting_df['seed'].tolist()):
            try:
                pred = get_predict_helper(smiles, target_name, seed)
                result.update(pred)
            except Exception as e:
                print(f"Error predicting {target_name}: {e}")
                    #pool.apply_async(get_predict_helper,
                             #args=(smiles, target_name, seed),
                             #callback=lambda x: result.update(x),
                             #error_callback=lambda x: print(x))
    else:
        info_df = pd.read_csv(INFO_CSV).set_index('uniprot_id')
        for index, row in info_df.iterrows():
            try:
                pred = specific_predict(smiles, index, row[0], row[1], eval(row[2]) if pd.notna(row[2]) else None)
                result.update(pred)
            except Exception as e:
                print(f"Error predicting {index}: {e}")
            #pool.apply_async(specific_predict,
             #                args=(smiles, index, row[0], row[1],
              #                     eval(row[2]) if row[2] else None),
               #              callback=lambda x: result.update(x),
                #             error_callback=lambda x: print(x))
   # pool.close()
   # pool.join()
    return result


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python predict_JL_batch.py <input_csv> <model_type> <output_directory> [num_workers]")
        print("\nArguments:")
        print("   input_csv    : CSV file with SMILES,ID (no header)")
        print("   model_type   : 0 (voting) or 1 (best models)")
        print("   output_dir   : Directory for output JSON files")
        print("   num_workers  : Number of parallel workers (optional)")
        sys.exit(1)

    input_csv = sys.argv[1]
    model_type = int(sys.argv[2])
    output_dir = sys.argv[3]
    num_workers = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    results = kipp_predict_batch(input_csv, model_type, output_dir, num_workers)
