import os
import random
import numpy.random
import scipy.stats
import torch
import dgl
import scipy.sparse as sp
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import anndata as ad
import networkx as nx
from scipy.io import mmread
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import accuracy_score
from model import GTModel
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_seed(seed=32):   
    dgl.seed(seed)        
    dgl.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)       
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) #所有GPU
        torch.cuda.manual_seed(seed)     # 当前GPU    
        # CUDA有些算法是non deterministic, 需要限制    
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # CUDA >= 10.2版本会提示设置这个环境变量
        torch.use_deterministic_algorithms(True)        
    print("set up seed!")

def normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    # 计算行和
    rowsum = adj.sum(1)    
    # 计算每行的倒数的平方，并将无穷大值替换为 0
    r_inv_sqrt = np.where(rowsum != 0, 1 / np.sqrt(rowsum), 0)    
    # 构建对角矩阵
    r_mat_inv_sqrt = np.diag(r_inv_sqrt)    
    # 对称归一化
    norm_adj = r_mat_inv_sqrt @ adj @ r_mat_inv_sqrt    
    return norm_adj


def capsule_pd_data_to_anndata(data, label, edge_index):
    '''
    :param data: pandas frame, including cell names, gene names
    :param label: cell labels (pandas)
    :param edge_index: COO format [[row...], [col...]]
    :return:
    '''    
    adata = ad.AnnData(data.to_numpy(), dtype=float)
    adata.obs_names = data.index.tolist()
    adata.var_names = data.columns.tolist()        
    adata.obs['cell_type'] = label.to_numpy().reshape(-1)
    adata.uns['edge_index'] = edge_index
    return adata


def median_normalization(data):
    '''
    :param data: matrix of (cells * genes)
    :return: median normalized data
    '''
    row_sum = np.sum(data, axis=1)
    mean_transcript = np.mean(row_sum)
    print("细胞平均表达量是 {:.3f}".format(mean_transcript))
    row_sum[np.where(row_sum == 0)] = 1
    # scale_factor = 1e4
    # data_norm = np.log1p((data / row_sum.reshape(-1 ,1))*scale_factor)
    data_norm = (data / row_sum.reshape(-1, 1)) * mean_transcript
    return data_norm

def combine_inter_intra_graph(inter_graph_path, intra_graph_path, n_nodes_ref, n_nodes_query):
    '''
    combine inter-graph and intra-graph to get hybrid graoh and convert it to coo format(edge_index)
    :param inter_graph_path:
    :param intra_graph_path:
    :return:
    '''
    inter_graph = pd.read_csv(inter_graph_path, index_col=0)
    intra_graph = pd.read_csv(intra_graph_path, index_col=0)

    
    # 先对query的数目做一个映射, 变成 ref_num + idx
    inter_graph['V2'] += n_nodes_ref
    intra_graph['V1'] += n_nodes_ref
    intra_graph['V2'] += n_nodes_ref
    
    # 获取row和col
    row = inter_graph['V1'].tolist() + intra_graph['V1'].tolist()
    col = inter_graph['V2'].tolist() + intra_graph['V2'].tolist()
    
    
    
    # 构建一个adj矩阵（保证是对称矩阵，是无向图)
    adj = np.identity(n_nodes_ref+n_nodes_query)
    adj[row, col] = 1
    adj[col, row] = 1

    # 再转成COO format
    row, col = adj.nonzero()
    row = list(row)
    col = list(col)

    return np.array([row, col])

def get_auxilary_graph(auxilary_graph_path, n_nodes):
    auxilary_graph = pd.read_csv(auxilary_graph_path, index_col=0)
    row = auxilary_graph['V1'].tolist()
    col = auxilary_graph['V2'].tolist()    
    # 构建一个adj矩阵（保证是对称矩阵，是无向图)
    adj = np.identity(n_nodes)
    adj[row, col] = 1
    adj[col, row] = 1
    
    # 再转成COO format
    row, col = adj.nonzero()
    row = list(row)
    col = list(col)

    return np.array([row, col])
    
    

#graph central
def centralissimo(G):
    centralities = []
    centralities.append(nx.pagerank(G))
    L = len(centralities[0])
    Nc = len(centralities)
    cenarray = np.zeros((Nc,L))
    for i in range(Nc):
        cenarray[i][list(centralities[i].keys())]=list(centralities[i].values())
    normcen = (cenarray.astype(float)-np.min(cenarray,axis=1)[:,None])/(np.max(cenarray,axis=1)-np.min(cenarray,axis=1))[:,None]
    return normcen


# calculate the percentage of elements smaller than the k-th element
def perc_for_entropy(input, k):
    return sum([1 if i else 0 for i in input < input[k]]) / float(len(input))


# calculate the percentage of elements larger than the k-th element
def perc_for_density(input, k): return sum([1 if i else 0 for i in input > input[k]]) / float(len(input))


def random_stratify_sample(ref_labels, train_size):
    # 对每个类都进行随机采样，分成train, val
    # 这地方要保证train的数据是从0开始计数的,
    # print(ref_labels.squeeze())
    
    label_set = set(list(ref_labels.squeeze()))
    train_idx = []
    val_idx = []
    for c in label_set:
        idx = np.where(ref_labels == c)[0]
        np.random.seed(20)
        np.random.shuffle(idx)
        train_num = int(train_size * len(idx))
        train_idx += list(idx[:train_num])
        val_idx += list(idx[train_num:])

    return train_idx, val_idx


def random_stratify_sample_with_train_idx(ref_labels, train_idx, init_num_per_class):
    '''
    paramters:
        train_idx: 训练集下标
        init_num_per_class: 针对每一个类选取一定初始数目节点
    '''
    label_set = list(set(list(ref_labels.squeeze())))
    label_set.sort()
    new_train_idx = []
    for c in label_set:
        idx = np.array(train_idx)[np.where(ref_labels[train_idx] == c)[0]]
        np.random.seed(20)
        if len(idx) < init_num_per_class:
            random_nodes = list(np.random.choice(idx, len(idx), replace=False))
        else:
            random_nodes = list(np.random.choice(idx, init_num_per_class, replace=False))
        new_train_idx += random_nodes
    return new_train_idx


def get_anndata(args):
    data_dir = args.data_dir                
    
    ref_data_h5 = ad.read_h5ad(os.path.join(data_dir, 'ref_data.h5ad'))
    query_data_h5 = ad.read_h5ad(os.path.join(data_dir, 'query_data.h5ad'))

    if args.use_auxilary:
        auxilary_data_h5 = ad.read_h5ad(os.path.join(data_dir, 'auxilary_data.h5ad'))
    
    ref_data = ref_data_h5.X.toarray()
    query_data = query_data_h5.X.toarray()
    
    data = np.concatenate([ref_data, query_data], axis=0)
    label = np.concatenate([ref_data_h5.obs['cell_type'].to_numpy(), query_data_h5.obs['cell_type'].to_numpy()], axis=0)                
    
    edge_index = combine_inter_intra_graph(inter_graph_path=os.path.join(data_dir, 'inter_graph.csv'),
                                            intra_graph_path=os.path.join(data_dir, 'intra_graph.csv'),
                                            n_nodes_ref=ref_data.shape[0],
                                            n_nodes_query=query_data.shape[0])
    
    if args.use_auxilary:
        auxilary_edge_index = get_auxilary_graph(auxilary_graph_path=os.path.join(data_dir, 'auxilary_graph.csv'), n_nodes=auxilary_data_h5.n_obs)            

    adata = ad.AnnData(csr_matrix(data, dtype=float), dtype=float)
    
    adata.obs_names = list(ref_data_h5.obs_names) + list(query_data_h5.obs_names)
    adata.var_names = ref_data_h5.var_names
    adata.obs['cell_type'] = label        
    adata.uns['edge_index'] = edge_index
    if args.use_auxilary:
    # take auxilary data all into all_data        
        adata.uns['auxilary_data'] = auxilary_data_h5.X.toarray()
        adata.uns['auxilary_label'] = auxilary_data_h5.obsm['label']
        adata.uns['auxilary_edge_index'] = auxilary_edge_index                                        
    adata.uns['n_ref'] = ref_data_h5.n_obs
    adata.uns['n_query'] = query_data_h5.n_obs        
    
    return adata, adata.uns['n_ref'], adata.uns['n_query']
    


def load_data(args, use_auxilary=True):
    if os.path.exists(os.path.join(args.data_dir, 'all_data.h5ad')):
        adata = ad.read_h5ad(os.path.join(args.data_dir, 'all_data.h5ad'))
        n_ref = adata.uns['n_ref']
        n_query = adata.uns['n_query']
    else:
    # 数据准备
        adata, n_ref, n_query = get_anndata(args=args)    
    
    # if not 'edge_index_knn' in adata.uns:
    if not 'edge_index_knn' in adata.uns:  
        adata.uns['edge_index_knn'] = construct_graph_with_knn(adata.X.toarray())
    
    adata.write(os.path.join(args.data_dir, 'all_data.h5ad'))     
    # take ref_query data into dgl data
    if args.graph_method == 'knn':
        src, dst = adata.uns['edge_index_knn'][0], adata.uns['edge_index_knn'][1]
    elif args.graph_method == 'mnn':
        src, dst = adata.uns['edge_index'][0], adata.uns['edge_index'][1]
    
    g_data = dgl.graph((src, dst), num_nodes=adata.n_obs)                
    y_true = adata.obs['cell_type'].to_numpy()    
    
    label_encoder = LabelEncoder()
    y_true = label_encoder.fit_transform(y_true)        
    
    g_data.ndata['x'] = torch.tensor(adata.X.toarray(), dtype=torch.float)              
    g_data.ndata['y_true'] = torch.tensor(y_true, dtype=torch.long)
    g_data.ndata['y_predict'] = torch.tensor(y_true, dtype=torch.long)
    
    # get data info
    data_info = get_data_info(args=args, adata=adata, n_ref=n_ref, n_query=n_query)
    data_info['label_encoder'] = label_encoder
    
    if not 'PE' in adata.uns:            
        pe_tensor = dgl.lap_pe(g_data, k=args.pos_enc_dim, padding=True)
        adata.uns['PE'] = pe_tensor.numpy()
        adata.write(os.path.join(args.data_dir, 'all_data.h5ad')) 
        g_data.ndata['PE'] = pe_tensor            
    else:
        g_data.ndata['PE'] = torch.FloatTensor(adata.uns['PE'])

    auxilary_g_data = None
    if use_auxilary:
        # if not 'auxilary_edge_index_knn' in adata.uns:
        adata.uns['auxilary_edge_index_knn'] = construct_graph_with_knn(adata.uns['auxilary_data'])
        adata.write(os.path.join(args.data_dir, 'all_data.h5ad')) 
        
        auxilary_g_data = get_auxilary_g_data(adata=adata)        
        
        if not 'auxilary_PE' in adata.uns:            
            pe_tensor = dgl.lap_pe(auxilary_g_data, k=args.pos_enc_dim, padding=True)
            adata.uns['auxilary_PE'] = pe_tensor.numpy()
            adata.write(os.path.join(args.data_dir, 'all_data.h5ad')) 
            auxilary_g_data.ndata['PE'] = pe_tensor            
        else:
            auxilary_g_data.ndata['PE'] = torch.FloatTensor(adata.uns['auxilary_PE'])
    
    

    return g_data, None if auxilary_g_data is None else auxilary_g_data, adata, data_info

def get_data_info(args, adata, n_ref, n_query):
    '''
        get train_idx, val_idx, test_idx etc. information about the data        
    '''
    data_info = {}
    
    ref_label = adata.obs['cell_type'].to_numpy()[:n_ref]        
    # if the task is cell type prediction, we can use random stratify sample
    train_idx, val_idx = random_stratify_sample(ref_label, train_size=0.8)
                            
    print("initial train idx:{:}".format(len(train_idx)))
    print("class num is {:}".format(len(np.unique(ref_label))))
    
    train_idx_for_active_learning = random_stratify_sample_with_train_idx(ref_label,
                                                                train_idx=train_idx,
                                                                init_num_per_class=args.init_num_per_class)        
    
    if args.use_auxilary:
        auxilary_label = adata.uns['auxilary_label']
        idxs = [i for i in range(auxilary_label.shape[0])]
        random.seed(32)
        random.shuffle(idxs)
        if args.auxilary_num != -1:
            idxs = idxs[:args.auxilary_num]                   
        auxilary_train_idx = idxs        
        data_info['auxilary_train_idx'] = auxilary_train_idx
        

    random.seed(32)
    # 对val_idx再做一个split
    random.shuffle(val_idx)
    ration = 0.5
    data_info['val_idx'] = val_idx[:int(len(val_idx) * 0.5)]
    data_info['gt_idx'] = val_idx[int(len(val_idx) * 0.5):]
    data_info['test_idx'] = [i + n_ref for i in range(n_query)]
    # 记录主动学习选取的节点
    data_info['selected_idx'] = []
    if args.train_idx:
        data_info['train_idx'] = np.load(args.train_idx).tolist()
    else:
        if args.al:
            data_info['train_idx'] = train_idx_for_active_learning        
        else:
            data_info['train_idx'] = train_idx     
       
    
    data_info['class_num'] = len(np.unique(adata.obs['cell_type'].to_numpy()))
    # 回归任务也需要知道label的dim
    if args.use_auxilary:
        data_info['auxilary_class_num'] = adata.uns['auxilary_label'].shape[1]
    
    return data_info            

def get_auxilary_g_data(adata):
    src, dst = adata.uns['auxilary_edge_index_knn'][0], adata.uns['auxilary_edge_index_knn'][1]
    g_data = dgl.graph((src, dst), num_nodes=adata.uns['auxilary_data'].shape[0])
    g_data.ndata['x'] = torch.tensor(adata.uns['auxilary_data'], dtype=torch.float)
    if adata.uns['auxilary_label'].shape[1] > 1:
        # 获取到最大的作为下标
        g_data.ndata['y_true'] = torch.tensor(np.argmax(adata.uns['auxilary_label'], axis=1), dtype=torch.long)        
    return g_data


def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def active_learning(g_data, epoch, out_prob, norm_centrality, args, data_info):    
    np.random.seed(32)
    gamma = np.random.beta(1, 1.005 - args.basef ** epoch)
    alpha = beta = (1 - gamma) / 2
    prob = out_prob
    if args.al and len(data_info['train_idx']) < data_info['max_nodes_num']:
        print("Active learning!")
        entropy = scipy.stats.entropy(prob.T)
        # kmeans = KMeans(n_clusters=data_info['class_num'], random_state=0).fit(prob)
        cluster_centers = []
        all_idx = [i for i in range(g_data.ndata['x'].shape[0])]       
        all_idx = np.array(all_idx) 
        ref_data_idx = np.setdiff1d(all_idx, data_info['test_idx'])
        prob = prob[ref_data_idx]        
        y_true = g_data.ndata['y_true'].numpy()[ref_data_idx]        
        for i in set(y_true):
            cluster_centers.append(np.mean(prob[np.where(y_true == i)], axis=0))
        cluster_centers = np.array(cluster_centers)            
        ed_score = euclidean_distances(prob, cluster_centers)        
        # ed_score = euclidean_distances(prob, kmeans.cluster_centers_)
        density = np.min(ed_score, axis=1)
        # entropy和density的norm: 计算样本中的百分位数（因为只需要比较样本之间的分数即可）
        norm_entropy = np.array([perc_for_entropy(entropy, i) for i in range(len(entropy))])
        norm_density = np.array([perc_for_density(density, i) for i in range(len(density))])
        norm_centrality = norm_centrality.squeeze()
        finalweight = alpha * norm_entropy[ref_data_idx] + beta * norm_density + gamma * norm_centrality[ref_data_idx]

        # 把train, val, test的数据排除, 从剩余的label budget里面获取节点
        finalweight[data_info['train_idx'] + data_info['val_idx']] = -100
        select_arr = np.argpartition(finalweight, -args.k_select)[-args.k_select:]
        for node_idx in select_arr:
            if node_idx not in data_info['train_idx']:
                data_info['train_idx'].append(node_idx)
                data_info['selected_idx'].append(node_idx)
                if (args.debug):
                    print("Epoch {:}: pick up {:} node to the training set!".format(epoch, args.k_select))


def construct_graph_with_knn(data, k=5):
    A = kneighbors_graph(data, k, mode='connectivity', include_self='auto')     
    # turn A into undirecitonal adjcent matrix        
    G = nx.from_numpy_array(A.toarray())
    edges = []    
    for (u, v) in G.edges():
        edges.append([u, v])
        if u!=v:
            edges.append([v, u])
    edges = np.array(edges).T
    # edges = torch.tensor(edges, dtype=torch.long)
    # [row, col] 2 * n
    return edges