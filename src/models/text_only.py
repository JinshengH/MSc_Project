import torch
import torch.nn as nn

from src.models.encoders import TextEncoder


class TextOnly(nn.Module):
    """Text-only baseline: frozen BERT -> CLS pooling -> MLP -> 3 logits.

    A standalone control model (language-bias check), not a variant of CAMC.
    Accepts the same forward signature as CAMC so the training loop stays
    generic; pixel_values is ignored.
    """

    def __init__(self, hidden_dim=768, num_classes=3, dropout=0.1):
        super().__init__()
        self.te = TextEncoder()
        for p in self.te.parameters():
            p.requires_grad = False
        self.te.eval()

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, pixel_values, input_ids, attention_mask=None):
        with torch.no_grad():
            txt_emb = self.te(input_ids, attention_mask)   # [B, N_txt, 768]
        logits = self.classifier(txt_emb[:, 0])            # CLS pooling
        return logits, None

    def train(self, mode=True):
        super().train(mode)
        self.te.eval()
        return self
