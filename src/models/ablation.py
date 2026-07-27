import torch
import torch.nn as nn

from src.models.CAMC import ContradictionAwareLayer
from src.models.encoders import ImageEncoder, TextEncoder


class CAMCAblation(nn.Module):
    """Exp 4 architecture ablation variants of CAMC (guide §5.2).

    Two axes, both fixed per experiment config:
      use_cross_attn  False = frozen encoders + CLS pooling only
      fusion          which of CAMC's fusion parts feed the classifier, as
                      a subset of "itdp" in CAMC's concat order:
                      i=img CLS, t=text CLS, d=|t-i|, p=t*i

    A1 = no attn + "it"; A3 = attn + "it"; A4 = attn + "itd".
    A2 (no attn + "itdp") is the LateFusion baseline and A6 (attn + "itdp")
    is CAMC itself — both already trained, not rebuilt here.
    """

    def __init__(self, hidden_dim=768, num_heads=8, num_layers=6,
                 num_classes=3, dropout=0.1, use_cross_attn=True,
                 fusion="itdp"):
        super().__init__()
        assert fusion and all(c in "itdp" for c in fusion), fusion
        self.fusion = fusion
        self.use_cross_attn = use_cross_attn

        self.ie = ImageEncoder()
        self.te = TextEncoder()
        for p in self.ie.parameters():
            p.requires_grad = False
        for p in self.te.parameters():
            p.requires_grad = False
        self.ie.eval()
        self.te.eval()

        self.img_proj = nn.Linear(hidden_dim, hidden_dim)  # mirrors CAMC
        if use_cross_attn:
            self.layers = nn.ModuleList([
                ContradictionAwareLayer(hidden_dim, num_heads,
                                        dropout=dropout)
                for _ in range(num_layers)
            ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * len(fusion), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, pixel_values, input_ids, attention_mask=None):
        with torch.no_grad():
            img_emb = self.ie(pixel_values)
            txt_emb = self.te(input_ids, attention_mask)

        img_emb = self.img_proj(img_emb)

        all_attn_weights = None
        if self.use_cross_attn:
            all_attn_weights = []
            for layer in self.layers:
                txt_emb, attn_weight = layer(img_emb, txt_emb)
                all_attn_weights.append(attn_weight)

        img_pool = img_emb[:, 0]
        text_pool = txt_emb[:, 0]
        parts = {"i": img_pool, "t": text_pool,
                 "d": torch.abs(text_pool - img_pool),
                 "p": text_pool * img_pool}
        fusion_feature = torch.cat([parts[c] for c in self.fusion], dim=-1)

        logits = self.classifier(fusion_feature)
        return logits, all_attn_weights

    def train(self, mode=True):
        super().train(mode)
        self.ie.eval()
        self.te.eval()
        return self
