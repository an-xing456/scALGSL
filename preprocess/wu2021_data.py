'''
    process data from single-cell portral 
    1. mtx expression file (log normalized data)
    2. gene name file
    3. cell name
    4. cell type
    into a h5ad file
'''
import pandas as pd
from scipy.io import mmread
import anndata as ad
import os
import argparse
import numpy as np
import random

def random_stratify(cell_types, adata, size, selected_idx=[]):
    classes = np.unique(cell_types)
    all_select_idxs = []
    for t in list(classes):
        idx = list(np.where(cell_types==t)[0])
        # delete seleted idxs
        idx = list(set(idx) - set(selected_idx))
        random.shuffle(idx)        
        s_idx = idx[:int(size * len(idx))]
        all_select_idxs += s_idx
    new_adata = adata[all_select_idxs, :]
    return new_adata, all_select_idxs
        
    


parser = argparse.ArgumentParser()

parser.add_argument('--dir_name', type=str, 
                    default='../raw_data/Wu2021')
parser.add_argument('--save_path', type=str, 
                    default='../raw_data/Wu2021')

args = parser.parse_args()
dir_name = args.dir_name
old_save_path = args.save_path


gene_names = pd.read_csv(os.path.join(dir_name, 'gene.tsv'), delimiter='\t', header=None)
meta_data = pd.read_csv(os.path.join(dir_name, 'meta_data.txt'), delimiter='\t')
cell_names = np.array(meta_data['NAME'].tolist())
cell_types = np.array(meta_data['CellType'].tolist())
bio_samples = np.array(meta_data['biosample_id'].tolist())
csr_data = mmread(os.path.join(dir_name, 'data.mtx')).tocsr().transpose() # cell * gene

group_list = ['BC-P1', 'BC-P2', 'BC-P3', 'PC-P1', ' M-P1'] #Note that there is an extra TAB character before "M-P1" in the source file
donor_id = ['CID4471', 'CID44971', 'CID4513', 'PID17267', 'SCC180161']

for i, group in enumerate(group_list):      

      
    key1 = group
    key2 = donor_id[i]    
    idx = list(np.where(np.isin(bio_samples, [key1, key2]))[0])          
        
    data = csr_data[idx, :]
    types = cell_types[idx]
    types = np.where(types == 'Cancer/Epithelial', 'Cancer', types)
    types = np.where(types == 'Cancer/Epithelial Cycling', 'Cancer', types)

        
    names = list(cell_names[idx])
    
    adata = ad.AnnData(data, dtype=float)
    adata.obs_names = names
    adata.var_names = gene_names.iloc[:, 0].tolist()
    adata.obs['cell_type'] = types
    
    new_data, _ = random_stratify(types, adata, 6000 / adata.n_obs)    
    save_path = os.path.join(old_save_path, group+'_6000')
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
    new_data.write(os.path.join(save_path, 'data.h5ad'), compression='gzip')
