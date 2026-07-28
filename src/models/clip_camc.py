import torch
import torch.nn as nn
from transformers import CLIPModel

from src.models.CAMC import ContradictionAwareLayer


class CLIPCAMC(nn.Module):
    """camc-on-CLIP: CAMC's cross-attention stack on frozen CLIP towers.

    Completes the 2x2 factorial {BERT+ViT, CLIP} x {pooled fusion,
    token-level interaction}: alignment is held literally identical to
    clip_fusion (same frozen checkpoint), so vs clip_fusion the only
    variable is the interaction head, and vs camc the only variable is
    the encoder family.

    Token features come from each tower's last hidden state, mapped
    through CLIP's own projection heads token-wise so both streams live
    in the 512-d joint space the alignment was trained in. Vision tokens
    get post_layernorm first, text tokens are already final-layernormed
    inside the tower; at the pooled positions this exactly reproduces
    CLIPModel's image_embeds/text_embeds pre-normalisation. No L2
    normalisation: unit-scaling every token is not part of CLIP's
    contract, and the interaction stack's LayerNorms handle scale.

    Two deliberate differences from CAMC forced by CLIP's text tower:
    the text pool is the EOS position (CLIP's pooled token; BERT pools
    [CLS] at position 0), located as the last non-padded token, and text
    token features are causal (each position only sees its prefix).
    The classifier input/hidden sizes match clip_fusion's head exactly
    (4*512 -> hidden_dim -> classes).
    """

    def __init__(self, model, embed_dim=512, hidden_dim=768, num_heads=8,
                 num_layers=6, num_classes=3, dropout=0.1):
        super().__init__()
        # Local safetensors copy, same as clip_fusion (see that model and
        # scripts/convert_clip_to_safetensors.py for why).
        self.clip = CLIPModel.from_pretrained(str(model))
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()

        self.img_proj = nn.Linear(embed_dim, embed_dim)  # mirrors CAMC
        self.layers = nn.ModuleList([
            ContradictionAwareLayer(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, pixel_values, input_ids, attention_mask=None):
        with torch.no_grad():
            vision_out = self.clip.vision_model(pixel_values=pixel_values)
            text_out = self.clip.text_model(
                input_ids=input_ids, attention_mask=attention_mask)
            img_emb = self.clip.visual_projection(
                self.clip.vision_model.post_layernorm(
                    vision_out.last_hidden_state))
            txt_emb = self.clip.text_projection(text_out.last_hidden_state)

        img_emb = self.img_proj(img_emb)

        all_attn_weights = []
        for layer in self.layers:
            txt_emb, attn_weight = layer(img_emb, txt_emb)
            all_attn_weights.append(attn_weight)

        # CLIP tokenizer always appends EOS, and DataCollatorWithPadding
        # right-pads, so EOS is the last attended position.
        if attention_mask is not None:
            eos_idx = attention_mask.sum(dim=-1) - 1
        else:
            eos_idx = torch.full((input_ids.size(0),),
                                 input_ids.size(1) - 1,
                                 device=input_ids.device)
        text_pool = txt_emb[torch.arange(txt_emb.size(0),
                                         device=txt_emb.device), eos_idx]
        img_pool = img_emb[:, 0]

        fusion_feature = torch.cat([
            img_pool,
            text_pool,
            torch.abs(text_pool - img_pool),
            text_pool * img_pool
        ], dim=-1)                               # [B, 4*embed_dim]

        logits = self.classifier(fusion_feature)
        return logits, all_attn_weights

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()
        return self
