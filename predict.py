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

warnings.filterwarnings('ignore')
#Global to hold pre-loaded models
LOADED_MODELS = {}
    
def load_all_models_once(model_type):
    """Load all models once at startup before forking workers"""
    print(f"Loading all models for model_type={model_type}...")
    models = {}

    if model_type == 0:
        #Voting model approach
        voting_df = pd.read_csv('./KIPP_data/voting_accor.csv', encoding='utf-8', low_memory=False)
        print(f"Model type 0: {len(voting_df)} targets to process")
        # Note: voting models are loaded per-prediction in get_predict_helper
        return {'type': 0, 'voting_df': voting_df}
    else:
        info_df = pd.read_csv('./KIPP_data/info_best.csv').set_index('uniprot_id')

        for uniprot_id, row in info_df.iterrows():
            method_name = row[0]
            seed = row[1]
            hyper = eval(row[2]) if pd.notna(row[2]) else None

            #Parse method name
            algorithm_name = method_name.split('_', 1)[0]
            algorithm_name = method_name.split('_', 1)[1] if algorithm_name == 'Graph' else algorithm_name
            featurizer_name = method_name.split('_', 1)[1] if method_name not in ['FPGNN', 'Chemprop'] else None
            #Load model and featurizier
            featurizer = featurizer_ofa(featurizer_name) if featurizer_name else None
            model = load_model_ofa(uniprot_id, algorithm_name, featurizer_name, hyper, seed)

            models[uniprot_id] = {
                    'model': model,
                    'method_name': method_name,
                    'algorithm': algorithm_name,
                    'featurizer_name': featurizer_name,
                    'featurizer': featurizer,
                    'seed': seed
                    }
        print(f"Loaded {len(models)} models into memory")
        return {'type': 1, 'models': models}



def specific_predict(smiles, uniprot_id, method_name, seed=None, hyper=None):
    """Make prediction using pre-loaded model"""
    model = model_info['model']
    method_name = mode_info['method_name']
    featurizer = model_info['featurizer']
    algorithm = model_info['algorithm']
    featurizer_name = model_info['featurizer_name']
    seed = model_info['seed']

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

def process_single_smiels(args):
    """Worker function - processes one SMILES using pre-loaded models"""
    smiles, compound_id, output_dir = args

    try:
        result{}

        if LOADED_MODELS['type'] == 0:
            #Voting approach - uses get_predict_helper which loads models internally
            voting_df = LOADED_MODELS['voting_df']
            for target_name, seed in zip(voting_df['name'].tolist(), voting_df['seed'].tolist()):
                pred = get_predicct_helper(smiles, target_name, seed)
                result.update(pred)
            else:
                # Model type 1 -use pre-loaded models
                for uniprot_id, model_info in LOADED_MODELS['models'].items():
                    pred = specific_predict_preloaded(smiles, uniprot_id, model_info)
                    result.update(pred)

            #Write output file
            output_file = f"{output_dir}/{compound_id}_KIPP_Results.json"
            with open(output_file, 'w') as f:
                json.dump(result, f)

            return {'id': compound_id, 'status': 'success'}

        except Exception as e:
            return {'id': compound_id, 'status': 'failed', 'error':str(e)}

def kipp_predict_batch(input_csv, model_type, output_dir, num_workers=None):
    """ Process a batch of SMILES from CSV file

    Args:
        input_csv: Path to CSV with clumns: SMILES, ID (no header)
        model_type: 0 or 1 (voting or best models)
        output_dir: Directory to write results
        num_workers: Number of worker processes (default: cpu_count)
    """
    global LOADED_MODELS
    
    #Load all models ONCE before forking workers
    LOADED_MODELS = load_all_models_once(model_type)

    #Read input CSV
    df = pd.read_csv(input_csv, header=None, name=['smiles', 'id'])
    print(f"Processing {len(df)} compounds with {num_workers} workers...")

    #Create output directory
    os.makedirs(output_dir, exist_ok=True)

    #Prepare arguments for workers
    args_list = [(row['smiles'], row['id'], output_dir) for _, row in df.iterrows()]

    #Set worker count
    if num_workers is None:
        num_workers = 10

    #Process in parallel -workers inherit LOADED_MODELS via copy-on-write
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(process_single_smiles, args_list)

    #Summary
    success = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')

    print(f"\nCompleted: {success} successful, {failed} failed")

    if failed > 0:
        print(f"\nFailed compounds:")
        for r in results:
            if r['status'] == 'failed':
                print(f" {r['id']}: {r.get('error', 'unknown error')}")

    return results

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python predict_JL_batch.py <input.csv> <model_type> <output_directory> [num_workers]")
        print("     input_csv    : CSV file with SMILES,ID (no header)")
        print("     model_type   : 0 (voting) or 1 (best models)")
        print("     output_dir   : Directory for output JSON files")
        print("     num_workers  : Number of parallel workers (optional, default: 10")
        sys.exit(1)

    input_csv = sys.argv[1]
    model_type = int(sys.argv[2])
    output_dir = sys.argv[3]
    num_workers = int(sys.argv[4]) if len(sys.argv) > 4 else None

    results = kipp_predict_batch(input_csv, model_type, output_dir, num_workers)
