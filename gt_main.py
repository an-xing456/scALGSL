import os
from utils import load_data
import numpy as np
import torch
from model import GTModel
import json
import argparse
from model.prognn import ProGNN
import pandas as pd
import copy
from utils import setup_seed
from scipy.sparse import save_npz

setup_seed()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
# data config
parser.add_argument('--data_dir', type=str, 
                             default='../experiments/bcp2_6000-bcp3_6000-exp0050/data',
                             help='data directory')
parser.add_argument('--output', type=str, default='al_acc.csv', help='output acc file')                

'''
    Active Learning
'''
parser.add_argument('--basef', type=float, 
                             default=0.8, 
                             help='base factor for active learning')
parser.add_argument('--k_select', type=int, 
                             default=1, 
                             help='num of nodes to select for every iteration')

parser.add_argument('--init_num_per_class', type=int, 
                             default=100, 
                             help='for active learning, we will pick some initial nodes for every class')
parser.add_argument('--max_per_class', type=int, 
                             default=150, 
                             help='max number of nodes for each class')

'''
    GT model
'''
parser.add_argument('--auxilary_epochs', type=int, 
                             default=50, 
                             help='epochs for training')

parser.add_argument('--epochs', type=int, 
                             default=50, 
                             help='epochs for training')


parser.add_argument('--debug', action='store_true', 
                             default=True, 
                             help='debug mode')

parser.add_argument('--hidden_dim', type=int,
                             default=256, 
                             help='hidden dim for graph transformer')
parser.add_argument('--n_heads', type=int,
                             default=2, 
                             help='num of heads for GTModel')
parser.add_argument('--n_layers', type=int, 
                             default=3, 
                             help='num of layers for GTModel')
parser.add_argument('--pos_enc_dim', type=int,
                             default=8, 
                             help='positional encoding dim')
parser.add_argument('--dropout', type=float,
                             default=0, 
                             help='dropout rate')

'''
    Graph Structure Learning
'''
parser.add_argument('--gt_interval', type=int, 
                             default=5 , #Train every other interval.
                             help='interval for GL')

parser.add_argument('--alpha', type=float, 
                    default=0, # LDS
                    help='weight of l1 norm')
parser.add_argument('--beta', type=float, 
                    default=0, # LDS
                    help='weight of nuclear norm')
parser.add_argument('--gamma', type=float, 
                    default=1, 
                    help='weight of GNN loss')
parser.add_argument('--lambda_', type=float, 
                    default=0, # LDS
                    help='weight of feature smoothing')

'''
    auxilary settings
'''
parser.add_argument('--auxilary_num', type=int,
                    default=-1,
                    help='num of nodes for auxilary model')

'''
    Switches
'''
parser.add_argument('--layer_norm', action='store_true',
                             default=False, 
                             help='layer norm for GTModel')
parser.add_argument('--batch_norm', action='store_true',
                             default=True, 
                             help='Batch norm for GTModel')
parser.add_argument('--residual', action='store_true',
                             default=True, 
                             help='residual for GTModel')
parser.add_argument('--symmetric', action='store_true', 
                            default=True,
                            help='whether use symmetric matrix')
parser.add_argument('--gsl', action='store_true',
                    default=False,
                    help='whether use graph structure learning')
parser.add_argument('--add_pos_enc', action='store_true',
                             default=True, 
                             help='whether adding postional encoding to node feature')
parser.add_argument('--al', action='store_true', 
                             default=False, 
                             help='active learning mode')
parser.add_argument('--is_auxilary', action='store_true',
                    default=True,
                    help='is auxilari model?')
parser.add_argument('--use_auxilary', action='store_true',
                    default=False,
                    help='for GTModel, whether use auxilary model')
parser.add_argument('--graph_method', type=str,
                    default='knn',
                    help='graph contruction method: knn or mnn')
parser.add_argument('--best_save', action='store_true',
                    default=False, 
                    help='save best model or not')
parser.add_argument('--bias', action='store_true',
                    default=False,
                    help='whether use bias in GTModel')
parser.add_argument('--config', type=str,
                    default=None,
                    help='hyperparameter setting')

                    
args = parser.parse_args()
    

proj = args.data_dir.split('/')[-2]

# load data
g_data, auxilary_g_data, adata, data_info = load_data(args=args, use_auxilary=args.use_auxilary)
max_nodes_num = data_info['class_num'] * args.max_per_class
data_info['max_nodes_num'] = max_nodes_num

# For debug information
print("data path is {:}, \n ref_data num: {:}, \nquery_data num :{:}, \n auxilary data num:{:}".format(args.data_dir, adata.uns['n_ref'], adata.uns['n_query'], auxilary_g_data.num_nodes() if args.use_auxilary else 0))



if args.use_auxilary:        
    # auxilary model no need: AL and GL
    auxilary_args = copy.copy(args)
    auxilary_args.epochs = 200
    auxilary_args.al = False
    auxilary_args.updated_adj = False
    auxilary_model = GTModel(args=auxilary_args,                    
                    class_num=data_info['auxilary_class_num'],
                    in_dim=auxilary_g_data.ndata['x'].shape[1],
                    pos_enc=auxilary_g_data.ndata['PE'].to(device) if args.add_pos_enc else None).to(device)

    # use Pro-GNN to train the GT
    auxilary_model_prognn = ProGNN(auxilary_model, data_info=data_info, args=auxilary_args, device=device)
    auxilary_model_prognn.fit(g_data=auxilary_g_data)
    torch.cuda.empty_cache() # release memory


'''
 ========= For cell type prediction ========= 
'''
args.is_auxilary = False
type_model = GTModel(args=args,                
                class_num=data_info['class_num'],
                in_dim=g_data.ndata['x'].shape[1],
                pos_enc=g_data.ndata['PE'].to(device) if args.add_pos_enc else None).to(device)

if args.use_auxilary:
    auxilary_embeddings = auxilary_model.get_embeddings(g_data=g_data, args=args)
    auxilary_output = auxilary_model.pred_cellstates(g_data=g_data, args=args).detach().cpu().numpy()
        
    print("auxilary label distribution is:")
    label = np.argmax(auxilary_output, axis=1)
    # 打印label的分布
    print(np.unique(label, return_counts=True))

    type_model.set_state_embeddings(auxilary_embeddings)


prognn = ProGNN(type_model, data_info=data_info, args=args, device=device)
prognn.fit(g_data=g_data)


test_res = prognn.test(features=g_data.ndata['x'].to(device), 
                       idx_test=data_info['test_idx'], 
                       labels=g_data.ndata['y_true'].to(device))



old_adj = g_data.adj_external(scipy_fmt='csr')
new_adj = test_res[3]

'''
 ======== save =========
'''

acc_file = args.output

ref_proj = proj.split('-')[0]
query_proj = proj.split('-')[1]
if args.use_auxilary:
    auxilary_proj = proj.split('-')[2]
    if args.auxilary_num != -1:
        auxilary_proj += '_{:}'.format(args.auxilary_num)
else:
    auxilary_proj = ''

with open('config/{:}-{:}-{:}_acc_{:.3f}_f1_{:.3f}.json'.format(
    ref_proj, query_proj, auxilary_proj, test_res[0], test_res[1]), 'w') as f:
    json.dump(vars(args), f)
    
second_key = 'GT'

if args.al:
    second_key += ' + AL'

if args.gsl:
    second_key += ' + GL'
    
first_key = ref_proj + '-' + query_proj
if args.use_auxilary:
    first_key += ('-' + auxilary_proj)

print("experiment {:}_{:} finished".format(first_key, second_key))

acc_file_path = os.path.join('result/acc', acc_file)
os.makedirs(os.path.dirname(acc_file_path), exist_ok=True)
columns = ["GT + AL + GL", "GT + AL", "GT + GL", "GT"]

if not os.path.exists(acc_file_path):
    acc_data = pd.DataFrame(columns=columns)
    acc_data.to_csv(acc_file_path, index=False)
else:
    acc_data = pd.read_csv(acc_file_path, index_col=0)

if first_key not in acc_data.index.tolist():
    new_row = {col: '' for col in acc_data.columns}
    acc_data.loc[first_key] = new_row

acc_data.loc[first_key][second_key] = test_res[0]
acc_data.to_csv(acc_file_path)




print("acc is {:.3f}".format(test_res[0]))

# save query_true.csv, query_predict.csv
ref_true = data_info['label_encoder'].inverse_transform(g_data.ndata['y_true'].numpy()[:adata.uns['n_ref']])
query_true = data_info['label_encoder'].inverse_transform(g_data.ndata['y_true'].numpy()[adata.uns['n_ref']:])
query_predict = data_info['label_encoder'].inverse_transform(test_res[2])

ref_true_df = pd.DataFrame(ref_true, columns=['cell_type'])
query_true_df = pd.DataFrame(query_true, columns=['cell_type'])
query_predict_df = pd.DataFrame(query_predict, columns=['cell_type'])

exp_save_path = os.path.join('result', first_key + '_' + second_key)
if not os.path.exists(exp_save_path):
    os.makedirs(exp_save_path)

save_npz(os.path.join(exp_save_path, "old_graph.npz"), old_adj)  
save_npz(os.path.join(exp_save_path, "new_graph.npz"), new_adj)  


ref_true_df.to_csv(os.path.join(exp_save_path, 'ref_true.csv'), index=False)
query_true_df.to_csv(os.path.join(exp_save_path, 'query_true.csv'), index=False)
query_predict_df.to_csv(os.path.join(exp_save_path, 'query_pred.csv'), index=False)

# save ref embeedings and query embeedings and auxilary embeddings
if args.use_auxilary:
    np.save(os.path.join(exp_save_path, 'auxilary_embeddings.npy'), auxilary_embeddings.detach().cpu().numpy())    
    cell_states = ['Angiogenesis', 'Apoptosis', 'CellCycle', 'Differentiation', 'DNAdamage', 'DNArepair', 'EMT', 'Hypoxia', 'Inflammation', 'Invasion', 'Metastasis', 'Proliferation', 'Quiescence', 'Stemness']    
    cell_states_score = pd.DataFrame(auxilary_output, columns=cell_states)
    print("auxilary label distribution is:")
    label = np.argmax(auxilary_output, axis=1)
    print(np.unique(label, return_counts=True))    
    cell_states_score.to_csv(os.path.join(exp_save_path, 'cell_states_score.csv'), index=False)
    
ref_query_embeddings = type_model.get_classifier_embeddings(g_data=g_data, args=args).detach().cpu().numpy()
ref_embeddings = ref_query_embeddings[:adata.uns['n_ref']]
query_embeddings = ref_query_embeddings[adata.uns['n_ref']:]

np.save(os.path.join(exp_save_path, 'ref_embeddings.npy'), ref_embeddings)
np.save(os.path.join(exp_save_path, 'query_embeddings.npy'), query_embeddings)

if args.al:
    np.save(os.path.join(exp_save_path, 'selected_idx.npy'), np.array(data_info['selected_idx']))
