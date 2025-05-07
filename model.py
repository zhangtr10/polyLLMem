from torch import nn
import torch
import math
## LORA
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=8):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling + self.bias
    
## FUSION
class GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # A linear layer mapping from 2*hidden_size -> hidden_size
        self.gate_layer = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, gpt_embed, conf_embed):
        # gpt_embed: (batch_size, hidden_size)
        # conf_embed: (batch_size, hidden_size)
        # Concatenate
        cat = torch.cat([gpt_embed, conf_embed], dim=-1)  # (batch, 2*hidden_size)
        # Compute gate
        gate = torch.sigmoid(self.gate_layer(cat))       # (batch, hidden_size)
        # Weighted combination
        combined = gate * gpt_embed + (1 - gate) * conf_embed   # (batch, hidden_size)
        return combined
    
## MODEL
class PolymerModel_cat(nn.Module):
    def __init__(self, gpt_model, unimol_model, hidden_size=512, rank=4, alpha=32, dropout=0.1):
        super().__init__()
        # Extract hidden sizes from models
        gpt_hidden_size = gpt_model.config.hidden_size if gpt_model else 4096
        conf_input_size = unimol_model.config.hidden_size if unimol_model else 1536

        # Projection layers for embeddings
        self.gpt_projection = nn.Sequential(
            nn.Linear(gpt_hidden_size, hidden_size), nn.GELU(), nn.BatchNorm1d(hidden_size)
        )
        self.conf_projection = nn.Sequential(
            nn.Linear(conf_input_size, hidden_size), nn.GELU(), nn.BatchNorm1d(hidden_size)
        )

        # LoRA layers for GPT and conf embeddings
        self.lora_gpt = LoRALayer(hidden_size, hidden_size, rank=rank, alpha=alpha)
        self.lora_conf = LoRALayer(hidden_size, hidden_size, rank=rank, alpha=alpha)
        
        # Gated fusion layer for combining GPT and conf embeddings
        self.Fusion_gpt_conf = GatedFusion(hidden_size)
        
        # Residual block
        self.residual_block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size),
        )

        # Regression network
        self.reg_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            #nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout),

            nn.Linear(hidden_size // 2, 64),
            nn.GELU(),
            #nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
                # Regression network
        # self.reg_net = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size // 2),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_size//2, hidden_size // 4),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_size//4, 64),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(64, 1),
        # )
        
    def forward(self, gpt_embed, conf_embed):
        # Project embeddings to hidden size
        gpt_feat = self.gpt_projection(gpt_embed)
        conf_feat = self.conf_projection(conf_embed)
        gpt_lora = self.lora_gpt(gpt_feat)
        conf_lora = self.lora_conf(conf_feat)        

        # Apply LoRA updates
        gpt_feat = gpt_feat + gpt_lora
        conf_feat = conf_feat + conf_lora
        # Fuse features and pass through the residual block
        combined = self.Fusion_gpt_conf(gpt_feat, conf_feat)
        residual = self.residual_block(combined)
        pred = self.reg_net(residual) 
        return {
            'prediction': pred,
            'enhanced': residual,
            'gpt_lora': gpt_lora,    # Return LoRA outputs for potential analysis
            'conf_lora': conf_lora
        }