import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.pgd import PGD, prox_operators
import warnings
import networkx as nx
from utils import centralissimo, accuracy, active_learning
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score

class ProGNN:
    """ ProGNN (Properties Graph Neural Network). See more details in Graph Structure Learning for Robust Graph Neural Networks, KDD 2020, https://arxiv.org/abs/2005.10203.

    Parameters
    ----------
    model:
        model: The backbone GNN model in ProGNN
    args:
        model configs
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
    See details in https://github.com/ChandlerBang/Pro-GNN.

    """

    def __init__(self, model, args, data_info, device):
        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_graph = None        
        self.weights = None
        self.estimator = None
        self.model = model.to(device)
        self.data_info = data_info       
         

    def fit(self, g_data):
        """Train Pro-GNN.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices
        """
                
        args = self.args    
        self.model_optimizer = optim.Adam(self.model.parameters(),
                               lr=1e-3)
                                       
        # 不需要转为numpy
        # save_adj = g_data.adj_external(scipy_fmt='csr')
        # save_adj = normalize_adj(save_adj)   
        # save_adj = csr_matrix(save_adj)     
        # save_eidx = torch.stack(g_data.edges()).cpu().numpy()
        # np.savetxt('old_graph.csv', save_eidx, delimiter=',')  
        # save_npz(os.path.join(, "old_graph.npz"), save_adj)      
        
        adj = g_data.adjacency_matrix().to_dense().to(self.device)
        estimator = EstimateAdj(adj, symmetric=args.symmetric, device=self.device).to(self.device)
        self.estimator = estimator
        self.model_optimizer_adj = optim.SGD(estimator.parameters(),
                              momentum=0.9, lr=1e-2)

        self.model_optimizer_l1 = PGD(estimator.parameters(),
                        proxs=[prox_operators.prox_l1],
                        lr=1e-2, alphas=[args.alpha])

        warnings.warn("If you find the nuclear proximal operator runs too slow, you can modify line 77 to use prox_operators.prox_nuclear_cuda instead of prox_operators.prox_nuclear to perform the proximal on GPU. See details in https://github.com/ChandlerBang/Pro-GNN/issues/1")
        self.model_optimizer_nuclear = PGD(estimator.parameters(),
                  proxs=[prox_operators.prox_nuclear_cuda],
                  lr=1e-2, alphas=[args.beta])


        node_x = g_data.ndata['x'].to(self.device)
        labels = g_data.ndata['y_true'].to(self.device)

        if args.is_auxilary:
            train_idx = self.data_info['auxilary_train_idx']
            val_idx = []
        else:
            train_idx = self.data_info['train_idx']
            val_idx = self.data_info['val_idx']
        
        
        # if args.is_auxilary:
        #     criterion = torch.nn.MSELoss()
        # else:            
        criterion = torch.nn.CrossEntropyLoss()
            
        # Train model
        t_total = time.time()
        if args.al and not args.is_auxilary:
            graph = nx.Graph(adj.detach().cpu().numpy())
            norm_centrality = centralissimo(graph)
        if args.is_auxilary:
            epochs = args.auxilary_epochs
        else:
            epochs = args.epochs
            
        for epoch in range(epochs):                                              
            new_adj = self.estimator.sample()
            prob = self.train_gnn(adj=new_adj, 
                                features=node_x,                               
                                labels=labels,
                                epoch=epoch,
                                criterion=criterion)
                            
            # 预留20个epoch去学习
            if args.al and not args.is_auxilary and epoch < args.epochs - 20:
                # will change outer data_info (the parameter is reference)
                active_learning(
                                g_data=g_data,
                                epoch=epoch,
                                out_prob=prob,
                                norm_centrality=norm_centrality,
                                args=self.args,
                                data_info=self.data_info)
                print("train set now has {:}".format(len(self.data_info['train_idx'])))
                
            # auxilary model不需要GL
            if args.gsl and not args.is_auxilary:
                if epoch % args.gt_interval == 0 or args.gt_interval == 0:
                    # Update S       
                    train_idx = self.data_info['gt_idx']                    
                    self.train_adj(epoch, node_x, adj, labels,
                            train_idx, val_idx)
                                     
                                        

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))                

        # Testing
        print("picking the best model according to validation performance")        
        
        if args.best_save:
            self.model.load_state_dict(self.weights)

    def train_gnn(self, adj, features, labels, epoch, criterion):
        args = self.args
        labels = labels.to(self.device)
        if args.debug:
            print("\n=== train_gnn ===")                
        
        
        t = time.time()
        self.model.train()
        self.model_optimizer.zero_grad()
        # GTModel        
        output = self.model(adj, features)
        
        if args.is_auxilary:
            train_idx = self.data_info['auxilary_train_idx']
            val_idx = []            
        else:
            train_idx = self.data_info['train_idx']
            val_idx = self.data_info['val_idx']
            
        test_idx = self.data_info['test_idx']
        
        loss_train = criterion(output[train_idx], labels[train_idx])
        if not args.is_auxilary:
            # main model
            acc_train = accuracy(output[train_idx], labels[train_idx])
        
        loss_train.backward()
        self.model_optimizer.step()

        prob = F.softmax(output.detach(), dim=1).cpu().numpy()                        
                
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        output = self.model(adj, features)        
        
        loss_val = criterion(output[val_idx], labels[val_idx])

        if args.is_auxilary:
            acc_train = accuracy(output[train_idx], labels[train_idx])
            if loss_val < self.best_val_loss:
                self.best_val_loss = loss_val
                self.best_graph = adj
                self.weights = deepcopy(self.model.state_dict())
                if self.args.debug:
                    print(f'saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())
        else:            
            acc_val = accuracy(output[val_idx], labels[val_idx])
            acc_test = accuracy(output[test_idx], labels[test_idx])
            f1_test = f1_score(torch.argmax(output[test_idx], dim=1).cpu().numpy(), labels[test_idx].detach().cpu().numpy(), average='macro')            
            if acc_val > self.best_val_acc:
                self.best_val_acc = acc_val
                self.best_graph = adj
                self.weights = deepcopy(self.model.state_dict())
                if self.args.debug:
                    print(f'saving current model and graph, best_val_acc: %s' % self.best_val_acc.item())
        if self.args.debug:  
            if args.is_auxilary:          
                print('Epoch: {:04d}'.format(epoch+1),
                        'loss_train: {:.4f}'.format(loss_train.item()),
                        'acc_train: {:.4f}'.format(acc_train.item()),
                        'time: {:.4f}s'.format(time.time() - t))
            else:
                print('Epoch: {:04d}'.format(epoch+1),
                        'loss_train: {:.4f}'.format(loss_train.item()),
                        'acc_train: {:.4f}'.format(acc_train.item()),
                        'loss_val: {:.4f}'.format(loss_val.item()),
                        'acc_val: {:.4f}'.format(acc_val.item()),
                        'acc_test：{:.4f}'.format(acc_test.item()),
                        'acc_f1:{:.4f}'.format(f1_test),
                        'time: {:.4f}s'.format(time.time() - t))
                
        self.model_optimizer.zero_grad() # 清除缓存
        return prob

    def train_adj(self, epoch, features, original_adj, labels, idx_train, idx_val):        
        estimator = self.estimator
        args = self.args        
        
        if args.debug:
            print("\n=== train_adj ===")
        t = time.time()
        estimator.train()
        self.model_optimizer_adj.zero_grad()
        
        loss_l1 = torch.norm(estimator.estimated_adj, 1)
        loss_fro = torch.norm(estimator.estimated_adj - original_adj, p='fro')        
        norm_adj = estimator.sample() # 其实norm_adj和estimated_adj在这边没有什么差别
        
        if args.lambda_:
            loss_smooth_feat = self.feature_smoothing(estimator.estimated_adj, features)
        else:
            loss_smooth_feat = 0 * loss_l1

        
        # edge_index = norm_adj.nonzero().T        
        
        if args.is_auxilary:
            criterion = torch.nn.MSELoss()            
        else:
            criterion = torch.nn.CrossEntropyLoss()    
            
        output = self.model(norm_adj, features)        
        loss_gcn = criterion(output[idx_train], labels[idx_train])
        
        if not args.is_auxilary:
            # if is main model
            acc_train = accuracy(output[idx_train], labels[idx_train])

        # loss_symmetric = torch.norm(estimator.estimated_adj \
        #                 - estimator.estimated_adj.t(), p="fro")
        
        # for loss that are diffiential
        # loss_fro不需要
        loss_diffiential =  0 * loss_fro + args.gamma * loss_gcn + args.lambda_ * loss_smooth_feat
        loss_diffiential.backward()               
        self.model_optimizer_adj.step()  # 更新adj的参数, 这部分是可微分的参数
        
        # 这部分不重要 loss_nuclear 和 loss_l1，这部分的更新直接看ProxOperator部分
        loss_nuclear =  0 * loss_fro
        if args.beta != 0:
            self.model_optimizer_nuclear.zero_grad()
            self.model_optimizer_nuclear.step()
            loss_nuclear = prox_operators.nuclear_norm # 内部计算好了nulcear_norm, 这里只是为了展示

        self.model_optimizer_l1.zero_grad()
        self.model_optimizer_l1.step()                
        total_loss = loss_fro \
                    + args.gamma * loss_gcn \
                    + args.alpha * loss_l1 \
                    + args.beta * loss_nuclear                    
        
        estimator.estimated_adj.data.copy_(torch.clamp(
                  estimator.estimated_adj.data, min=0, max=1))
        
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        with torch.no_grad():
            # 进行五次采样
            loss_val = 0
            for i in range(5):
                norm_adj = estimator.sample()
                # edge_index = norm_adj.nonzero().T           
                output = self.model(norm_adj, features)
                loss_val += criterion(output[idx_val], labels[idx_val])
            loss_val /= 5
            
        
        acc_val = accuracy(output[idx_val], labels[idx_val])                
        print('Epoch: {:04d}'.format(epoch+1),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'time: {:.4f}s'.format(time.time() - t))
        
        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = norm_adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())            

        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_fro: {:.4f}'.format(loss_fro.item()),
                      'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                      'loss_feat: {:.4f}'.format(loss_smooth_feat.item()),                      
                      'delta_l1_norm: {:.4f}'.format(torch.norm(estimator.estimated_adj-original_adj, 1).item()),
                      'loss_l1: {:.4f}'.format(loss_l1.item()),
                      'loss_total: {:.4f}'.format(total_loss.item()),
                      'loss_nuclear: {:.4f}'.format(loss_nuclear.item()))
                
        
    def test(self, features, idx_test, labels):
        """
            Evaluate the performance of ProGNN on test set
        """
        print("\t=== testing ===")
        
        if self.args.is_auxilary:
            criterion = torch.nn.MSELoss()        
        else:
            criterion = torch.nn.CrossEntropyLoss()                    
                
        self.model.eval()        

        adj = self.best_graph
        if self.best_graph is None:
            # 一般就是GSL没有使用的时候
            adj = self.estimator.normalize()
        
        # edge_index = adj.nonzero().T                 
        # 采样5次
        loss_test = 0
        acc_test = 0
        best_acc = 0
        best_macro_f1 = 0
        best_output = None
        macrof1_test = 0
        new_adj = None
        with torch.no_grad():
            for i in range(5):
                adj = self.estimator.sample()
                output = self.model(adj, features)                
                
                # save_npz("new_graph.npz", save_adj)            
                # save_eidx = edge_index.detach().cpu().numpy()
                # np.savetxt('new_graph.csv', save_eidx, delimiter=',')
                loss_test += criterion(output[idx_test], labels[idx_test])                              
                acc_tmp = accuracy(output[idx_test], labels[idx_test])          
                acc_test += acc_tmp            
                if acc_tmp > best_acc:
                    best_acc = acc_tmp
                    best_output = output
                    print("save new adj")
                    new_adj = csr_matrix(adj.detach().cpu().numpy())
                
                macrof1_test += f1_score(torch.argmax(output[idx_test], dim=1).cpu().numpy(), labels[idx_test].detach().cpu().numpy(), average='macro')
            
        acc_test /= 5
        macrof1_test /= 5
        loss_test /= 5            
                
        print("\tTest set results:",
                "loss= {:.4f}".format(loss_test.item()),
                "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item(), macrof1_test, torch.argmax(best_output[idx_test], dim=1).detach().cpu().numpy(), new_adj
        

    def feature_smoothing(self, adj, X):
        adj = (adj.t() + adj)/2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv  + 1e-3
        r_inv = r_inv.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        # L = r_mat_inv @ L
        L = r_mat_inv @ L @ r_mat_inv

        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat


class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, adj, symmetric=False, device='cpu'):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self._init_estimation(adj)
        self.symmetric = symmetric
        self.device = device

    def _init_estimation(self, adj):
        with torch.no_grad():
            n = len(adj)
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):
        if self.symmetric:
            adj = (self.estimated_adj + self.estimated_adj.t())/2
        else:
            adj = self.estimated_adj
        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx
    
    def sample(self):
        '''
            采用伯努利采样来进行0-1映射
        '''        
        edge_probs = self.estimated_adj
        torch.manual_seed(32)
        adj = torch.distributions.Bernoulli(edge_probs).sample()                
        # STE                    
        adj = (adj - edge_probs).detach() + edge_probs
        if self.symmetric:
            adj = (adj + adj.t())/2
        # Normalize要比不norm稍微好点
        return self._normalize(adj + torch.eye(adj.shape[0]).to(self.device))