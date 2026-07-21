import torch
import torch.nn as nn

from src.models.encoders import ImageEncoder, TextEncoder


class LateFusion(nn.Module):
    """Late fusion baseline: frozen ViT + frozen BERT, CLS pooling per
    modality, then the same 4-way fusion vector and classifier head as CAMC.

    Ablation contrast with CAMC: no cross-attention layers, so the two
    modalities never interact at token level. The image-side linear
    projection and the [img, txt, |img-txt|, img*txt] fusion mirror CAMC
    exactly, isolating the contribution of the ContradictionAwareEncoder.
    """

    def __init__(self, hidden_dim=768, num_classes=3, dropout=0.1):
        super().__init__()
        self.ie = ImageEncoder()
        self.te = TextEncoder()
        for p in self.ie.parameters():
            p.requires_grad = False
        for p in self.te.parameters():
            p.requires_grad = False
        self.ie.eval()
        self.te.eval()

        self.img_proj = nn.Linear(hidden_dim, hidden_dim)  # mirrors CAMC img_proj

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, pixel_values, input_ids, attention_mask=None):
        with torch.no_grad():
            img_emb = self.ie(pixel_values)              # [B, 1+N_patch, 768]
            txt_emb = self.te(input_ids, attention_mask)  # [B, N_txt, 768]

        img_pool = self.img_proj(img_emb[:, 0])  # CLS pooling per modality
        text_pool = txt_emb[:, 0]

        fusion_feature = torch.cat([
            img_pool,
            text_pool,
            torch.abs(text_pool - img_pool),
            text_pool * img_pool
        ], dim=-1)                               # [B, 4*hidden]

        logits = self.classifier(fusion_feature)
        return logits, None

    def train(self, mode=True):
        super().train(mode)
        self.ie.eval()
        self.te.eval()
        return self
