import torch
import torch.nn as nn

from src.models.encoders import ImageEncoder


class ImageOnly(nn.Module):
    """Image-only baseline: frozen ViT -> CLS pooling -> MLP -> 3 logits.

    A standalone control model (image-bias check), mirroring TextOnly.
    Accepts the same forward signature as CAMC so the training loop stays
    generic; input_ids/attention_mask are ignored.
    """

    def __init__(self, hidden_dim=768, num_classes=3, dropout=0.1):
        super().__init__()
        self.ie = ImageEncoder()
        for p in self.ie.parameters():
            p.requires_grad = False
        self.ie.eval()

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, pixel_values, input_ids, attention_mask=None):
        with torch.no_grad():
            img_emb = self.ie(pixel_values)      # [B, 1+N_patch, 768]
        logits = self.classifier(img_emb[:, 0])  # CLS pooling
        return logits, None

    def train(self, mode=True):
        super().train(mode)
        self.ie.eval()
        return self
