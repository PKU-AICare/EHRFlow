import torch
from torch import nn


class MHAGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim: int=32, feat_dim: int=8, act_layer=nn.GELU, drop=0.1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        self.num_heads = 4
        self.act = act_layer()
        self.grus = nn.ModuleList(
            [
                nn.GRU(1, feat_dim, num_layers=1, batch_first=True)
                for _ in range(input_dim)
            ]
        )
        self.mha = nn.MultiheadAttention(input_dim * feat_dim, self.num_heads, dropout=drop, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            self.act,
            nn.Dropout(drop),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.out_proj = nn.Linear(input_dim * feat_dim, hidden_dim)
        
        self.input_attn = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            self.act,
            nn.Linear(input_dim // 4, input_dim),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(drop)
    
    def forward(self, x, **kwargs):
        # x: [bs, time_steps, input_dim]
        bs, time_steps, input_dim = x.shape

        x_unrolled = x.contiguous().view(-1, input_dim)    # [bs * t, h]
        input_attn = self.input_attn(x_unrolled).view(bs, time_steps, input_dim) # [bs, t, d_input]

        x = x * input_attn
        out = torch.zeros(bs, time_steps, self.input_dim, self.feat_dim).to(x.device)
        # attention = torch.zeros(bs, time_steps, self.input_dim, time_steps).to(x.device)
        for i, gru in enumerate(self.grus):
            cur_feat = x[:, :, i].unsqueeze(-1)     # [bs, time_steps, 1]
            cur_feat = gru(cur_feat)[0]             # [bs, time_steps, feat_dim]
            out[:, :, i] = cur_feat
            # attention[:, :, i] = attn_weight

        out = out.flatten(2)        # [bs, time, input, feat] -> [bs, time, input * feat]
        out = self.dropout(out)
        out = self.mha(out, out, out)[0]
        out = self.out_proj(out)    # [bs, time, input * feat] -> [bs, time, hidden_dim]
        out = out + self.ffn(out)   # [bs, time, hidden_dim] -> [bs, time, hidden_dim]
        
        feature_importance = input_attn.sum(-1).squeeze(-1) # [bs, input_dim]
        # time_step_feature_importance = input_attn  # [bs, time_steps, input_dim]
        
        scores = {
            'feature_importance': feature_importance,
            'time_step_feature_importance': input_attn
        }
        return out, scores
