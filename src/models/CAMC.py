import torch
import torch.nn as nn

from src.models.encoders import ImageEncoder, TextEncoder

class ContradictionAwareLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_t1 = nn.LayerNorm(hidden_dim)
        self.ffn_t = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm_t2 = nn.LayerNorm(hidden_dim)

    def forward(self, img_emb, txt_emb, img_mask=None):
       
        cross_out, attn_weights = self.cross_attn(
            query=txt_emb,
            key=img_emb,
            value=img_emb,
            key_padding_mask=img_mask
        )  
        txt_emb = self.norm_t1(txt_emb + cross_out)

        ffn_out = self.ffn_t(txt_emb)
        txt_emb = self.norm_t2(txt_emb + ffn_out)

        return txt_emb, attn_weights
    
class ContradictionAwareEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, num_layers=6):
        super().__init__()
        self.img_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layers = nn.ModuleList([
            ContradictionAwareLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, img_emb, txt_emb):
        img_emb = self.img_proj(img_emb)   # ablation: img_proj

        all_attn_weights = []
        for layer in self.layers:
            txt_emb, attn_weight = layer(img_emb, txt_emb)
            all_attn_weights.append(attn_weight)   

        text_pool = txt_emb[:, 0]   
        img_pool = img_emb[:, 0]    

        fusion_feature = torch.cat([
            img_pool,
            text_pool,
            torch.abs(text_pool - img_pool),
            text_pool * img_pool
        ], dim=-1)                  # [B, 4*hidden]

        return fusion_feature, all_attn_weights
    
class CAMC(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8,
                 num_layers=6, num_classes=3):
        super().__init__()
        self.ie = ImageEncoder()
        self.te = TextEncoder()
        self.encoder = ContradictionAwareEncoder(
            hidden_dim, num_heads, num_layers
        )


        for p in self.ie.parameters():
            p.requires_grad = False
        for p in self.te.parameters():
            p.requires_grad = False
        self.ie.eval()
        self.te.eval()

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, pixel_values, input_ids, attention_mask=None):
        with torch.no_grad():  
            img_emb = self.ie(pixel_values)
            txt_emb = self.te(input_ids, attention_mask)

        fusion_feature, all_attn_weights = self.encoder(img_emb, txt_emb)
        logits = self.classifier(fusion_feature)

        return logits, all_attn_weights

    def train(self, mode=True):
        super().train(mode)   
        self.ie.eval()        
        self.te.eval()
        return self