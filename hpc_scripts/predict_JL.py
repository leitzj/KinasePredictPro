#Modified KIPP by Jeremy Leitz
#This modified version of predict.py allows for an output to a provided desitnation
import multiprocessing
import pandas as pd
import numpy as np
from chemprop.train import predict
from chemprop.data import MoleculeDataLoader, MoleculeDatapoint, MoleculeDataset
from deepchem.data import NumpyDataset
from voting_fusion_use import get_predict_helper
from ofa_smi_eval import featurizer_ofa, load_model_ofa, predict_ofa
import json
import time
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ============== HARD CODED PATHS =================
KIPP_BASE_DIR = Path("/home/jleitz/Software/KIPP/KinasePredictPro")
DATA_DIR = KIPP_BASE_DIR / "data"
VOTING_CSV = DATA_DIR / "voting_accor.csv"
INFO_CSV = DATA_DIR / "info_best.csv"
# =================================================


def process_single_smiles(args):
    """Worker function for processing one SMILES"""
    smiles, compound_id, model_type = args
    try:
        result = kipp_predict(smiles, model_type)
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
    """Process a batch of SMILES from CSV file"""

    #Read input CSV
    df = pd.read_csv(input_csv, header=None, names=['smiles', 'id'])

    print(f"Processing {len(df)} compounds with {num_workers} workers...")

    #Prepare arguments for parallel processing
    args_list = [(row['smiles'], row['id'], model_type)
            for _, row in df.iterrows()]

    #Process in parallel using multiprocessing
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(process_single_smiels, args_list)

    #Save individual results files
    os.makedirs(output_dir, exist_ok=True)
    for result in results:
        output_file = f"{output_dir}/{result['id']}_KIPP_Results.json"
        with open(output_file, 'w') as f:
            json.dump(result['predictions'] if result['status' == 'success' else result, f)
    print(f"Completed processing {len(results)} compounds")
    return results

def specific_predict(smiles, uniprot_id, method_name, seed=None, hyper=None):
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


def kipp_predict(smiles: str, model_type: int):
    assert model_type in range(0, 2)
    pool = multiprocessing.Pool()
    result = {}
    if model_type == 0:
        voting_df = pd.read_csv(VOTING_CSV,
                                encoding='utf-8',
                                low_memory=False)
        for target_name, seed in zip(voting_df['name'].tolist(),
                                     voting_df['seed'].tolist()):
            pool.apply_async(get_predict_helper,
                             args=(smiles, target_name, seed),
                             callback=lambda x: result.update(x),
                             error_callback=lambda x: print(x))
    else:
        info_df = pd.read_csv(INFO_CSV).set_index('uniprot_id')
        for index, row in info_df.iterrows():
            pool.apply_async(specific_predict,
                             args=(smiles, index, row[0], row[1],
                                   eval(row[2]) if row[2] else None),
                             callback=lambda x: result.update(x),
                             error_callback=lambda x: print(x))
    pool.close()
    pool.join()
    return result


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python predict_JL_batch.py <input_csv> <model_type> <output_directory> [num_workers]")
    input_csv = sys.argv[1]
    model_type = int(sys.argv[2])
    output_dir = sys.argv[3]
    num_workers = int(sys.argv[4]) if len(sys.argv) > 4 else None
    #Change to script directory to handle relative paths
