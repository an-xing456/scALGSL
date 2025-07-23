import torch
import torch.nn.functional as F
import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SparseMHA(nn.Module):
    """Sparse Multi-head Attention Module"""

    def __init__(self, hidden_dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scaling = hidden_dim ** -0.5
        

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

    def forward(self, A, h):
        N = len(h)
        # [N, dh, nh]
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        # [N, dh, nh]
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        # [N, dh, nh]
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)        
        # bmm: (b, n, m) * (b, m, q) = (m, n, q)
        # q.transpose(0, 2).transpose(1, 2) = (nh, N, dh)
        # k.transpose(0, 2) = (nh, dh, N)
        # attn = (nh, N, N)
        attn = torch.mul(A, torch.bmm(q.transpose(0, 2).transpose(1, 2), k.transpose(0, 2)))        
        # attn = dglsp.bsddmm(A, q, k.transpose(1, 0))  # (sparse) [N, N, nh]
        # Sparse softmax by default applies on the last sparse dimension.
        # attn = attn.softmax()  # (sparse) [N, N, nh]
        attn = F.softmax(attn, dim=2)
        # v.transpose(0, 2).transpose(1, 2) = [nh,N,dh]
        # attn = [nh, N, N]
        # out = [nh, N, dh]
        out = torch.bmm(attn, v.transpose(0, 2).transpose(1, 2))        
        # out = [N, dh, nh]
        out = out.transpose(0, 1).transpose(1, 2)
        # out = dglsp.bspmm(attn, v)  # [N, dh, nh]
        out = self.out_proj(out.reshape(N, -1))
        return out
