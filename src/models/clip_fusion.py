import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


class CLIPFusion(nn.Module):
    """CLIP fusion baseline: frozen CLIP (ViT-B/16) projected embeddings,
    then the same 4-way fusion vector and classifier head as late_fusion.

    Ablation contrast with late_fusion: the image tower is the same ViT-B/16
    architecture, so the isolated variable is the pretraining objective --
    contrastively *aligned* image/text spaces (CLIP) vs independently
    pretrained ViT/BERT. Both embeddings come from CLIP's projection heads
    (512-d, the space its alignment was trained in) and are L2-normalised,
    as in CLIP's own similarity computation. No extra projection layer:
    re-projecting would distort the alignment under test.
    """

    def __init__(self, model, embed_dim=512,
                 hidden_dim=768, num_classes=3, dropout=0.1):
        super().__init__()
        # Local safetensors copy produced by scripts/convert_clip_to_safetensors.py;
        # the upstream repo ships only .bin, which transformers refuses to load
        # under torch < 2.6 (CVE-2025-32434).
        self.clip = CLIPModel.from_pretrained(str(model))
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, pixel_values, input_ids, attention_mask=None):
        # Explicit tower + projection path rather than get_image_features /
        # get_text_features: those changed return type across transformers
        # major versions (5.x hands back an output object of unprojected
        # hidden states), while vision_model/text_model + visual_projection/
        # text_projection are stable. Normalised, this reproduces CLIPModel's
        # own image_embeds/text_embeds exactly (verified to ~6e-8).
        with torch.no_grad():
            vision_out = self.clip.vision_model(pixel_values=pixel_values)
            text_out = self.clip.text_model(
                input_ids=input_ids, attention_mask=attention_mask)
            img_pool = self.clip.visual_projection(vision_out.pooler_output)
            text_pool = self.clip.text_projection(text_out.pooler_output)
        img_pool = F.normalize(img_pool, dim=-1)
        text_pool = F.normalize(text_pool, dim=-1)

        fusion_feature = torch.cat([
            img_pool,
            text_pool,
            torch.abs(text_pool - img_pool),
            text_pool * img_pool
        ], dim=-1)                               # [B, 4*embed_dim]

        logits = self.classifier(fusion_feature)
        return logits, None

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()
        return self
