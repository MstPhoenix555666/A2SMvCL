'''
Description: A2SMvCL model file
version: 
'''
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class GCNCLBertClassifier(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn_model = GCNAbsaModel(bert, opt=opt)
        self.classifier = nn.Linear(opt.bert_dim * 2, opt.polarities_dim)

    def forward(self, inputs):
        outputs1, outputs2, pooled_output, ret = self.gcn_model(inputs)
        final_outputs = torch.cat((outputs1, outputs2, pooled_output), dim=-1)
        logits = self.classifier(final_outputs)

        loss_cl = ret.mean()
        return logits, loss_cl


class GCNAbsaModel(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn = GCNBert(bert, opt, opt.num_layers)
        self.mem_dim = opt.bert_dim // 2
        self.fc1 = torch.nn.Linear(self.mem_dim, 32)
        self.fc2 = torch.nn.Linear(32, self.mem_dim)
        self.tau = opt.tau

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim1(self, z1, z2):
        z1 = F.normalize(z1, dim=2)
        z2 = F.normalize(z2, dim =2)
        return torch.bmm(z1,z2.transpose(1,2))

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):  # cal cosine simility
        z1 = F.normalize(z1, dim=2)
        z2 = F.normalize(z2, dim=2)
        return torch.bmm(z1, z2.transpose(1,2))
      
    def scope_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, s_mask, a_mask):
        f = lambda x: torch.exp(x / self.tau)  # f: e^(f(z1,z2)/t)
        Ba,Seq,Dim = z1.shape
        
        #aspect-mask
        a_mask = a_mask.unsqueeze(-1) #[B,S,1]
        asp_m = a_mask.expand(Ba,Seq,Dim) #[B,S,D]
        a_z1 = z1 * asp_m
        m_between_sim = f(self.sim(a_z1, z2)) # f(ma_h1, h2)
        m_refl_sim = f(self.sim(a_z1, z1)) # f(ma_h1, h1)
        
        #span-mask
        s_mask = s_mask.unsqueeze(-1) #[B,S,1]
        span_m = s_mask.expand(Ba,Seq,Dim) #[B,S,D]
        s_z1 = z1 * span_m
        s_z2 = z2 * span_m    
        as_refl_sim = f(self.sim(a_z1, s_z1)) # f(ma_h1, ms_h1)
        as_between_sim = f(self.sim(a_z1, s_z2)) # f(ma_h1, ms_h2)

        # weighted f()
        weighted_between_sim = f(torch.mul(self.sim(a_z1, s_z2), self.sim(a_z1, s_z2).diagonal(dim1=-2,dim2=-1).unsqueeze(dim=-1)))

        #Scope-asisted MvGCL
        pos = as_between_sim.diagonal(dim1=-2,dim2=-1) + (as_refl_sim.sum(2) - as_refl_sim.diagonal(dim1=-2,dim2=-1)) + (weighted_between_sim.sum(2) - weighted_between_sim.diagonal(dim1=-2,dim2=-1)) # 3
        alle = m_refl_sim.sum(2) + m_between_sim.sum(2) - m_refl_sim.diagonal(dim1=-2,dim2=-1)
        cl_logit = pos / alle

        return -torch.log(cl_logit) 


    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask, asp, asp_end, adj_dep, src_mask, aspect_mask, len, span = inputs
        h1, h2, pooled_output = self.gcn(adj_dep, inputs)
        
        # span-masked graphCL
        h1 = self.projection(h1)
        h2 = self.projection(h2)  #[B,s,D/2]

        # scope-asisted GCL
        l1 = self.scope_semi_loss(h1, h2, span, aspect_mask) #[B,S] 
        l2 = self.scope_semi_loss(h2, h1, span, aspect_mask)  

        ret = (l1 + l2) * 0.5
        ret = ret.mean(dim=1, keepdim=True) #if mean else ret.sum()
        
        # avg pooling asp feature
        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.bert_dim // 2) 
        outputs1 = (h1*aspect_mask).sum(dim=1) / asp_wn
        outputs2 = (h2*aspect_mask).sum(dim=1) / asp_wn

        return outputs1, outputs2, pooled_output, ret


class GCNBert(nn.Module):
    def __init__(self, bert, opt, num_layers):
        super(GCNBert, self).__init__()
        self.bert = bert
        self.opt = opt
        self.layers = num_layers
        self.mem_dim = opt.bert_dim // 2
        self.attention_heads = opt.attention_heads
        self.bert_dim = opt.bert_dim
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        self.layernorm = LayerNorm(opt.bert_dim)
        
        # keyward 
        self.attn = MultiHeadAttention(opt.attention_heads, self.bert_dim)
        
        # gcn layer
        self.depW = nn.ModuleList() # DepGCN
        for layer in range(self.layers):
            input_dim = self.bert_dim if layer == 0 else self.mem_dim
            self.depW.append(nn.Linear(input_dim, self.mem_dim))
            
        self.semW = nn.ModuleList()  # SemGCN
        for j in range(self.layers):
            input_dim = self.bert_dim if j == 0 else self.mem_dim
            self.semW.append(nn.Linear(input_dim, self.mem_dim))
            
        self.fc3 = nn.Linear(self.mem_dim, self.mem_dim)
        self.fc2 = nn.Linear(self.bert_dim, self.mem_dim)

    def forward(self, adj, inputs):
        text_bert_indices, bert_segments_ids, attention_mask, asp, asp_len, adj_dep, src_mask, aspect_mask, len, span = inputs
        src_mask = src_mask.unsqueeze(-2)
        
        sequence_output, pooled_output = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids)
        sequence_output = self.layernorm(sequence_output)
        gcn_inputs = self.bert_drop(sequence_output)
        pooled_output = self.pooled_drop(pooled_output)
        
        attn_tensor = self.attn(gcn_inputs, gcn_inputs, src_mask)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        adj_ag = None
       
        # Attention matrix
        for i in range(self.attention_heads):
            if adj_ag is None:
                adj_ag = attn_adj_list[i]
            else:
                adj_ag += attn_adj_list[i]
        adj_ag = adj_ag / self.attention_heads

        for j in range(adj_ag.size(0)):
            adj_ag[j] -= torch.diag(torch.diag(adj_ag[j]))
            adj_ag[j] += torch.eye(adj_ag[j].size(0)).cuda()
        adj_ag = src_mask.transpose(1, 2) * adj_ag

        H_l = gcn_inputs

        for l in range(self.layers):
            si = nn.Sigmoid()
            # **********GCN*********
            AH_sem = adj_ag.bmm(H_l)
            I_sem = self.semW[l](AH_sem) #SemGCN
            AH_dep = adj.bmm(H_l)
            I_dep = self.depW[l](AH_dep) #depGCN
            g = si(I_dep)
            lam_g = self.opt.lam * g # [16, 100, 768/2]
            I_com = torch.mul((1-lam_g),I_sem) + torch.mul(lam_g,I_dep) # adaptive fusion
            relu = nn.ReLU()
            H_out = relu(self.fc3(I_com))
            
            if l == 0:
                H_l = self.fc2(H_l)
            g_l = si(H_l)
            H_l = torch.mul(g_l, H_out) + torch.mul((1 - g_l),H_l)

        return H_l, I_sem, pooled_output

def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn
