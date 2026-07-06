import torch.nn as nn
from transformers import ViTModel, BertModel

class ImageEncoder(nn.Module):
    def __init__(self, model="google/vit-base-patch16-224"):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
       
        img_emb = outputs.last_hidden_state   # [B, 1+N_patch, 768]
        return img_emb


class TextEncoder(nn.Module):
    def __init__(self, model="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model)

    def forward(self, input_ids, attention_mask=None):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        txt_emb = output.last_hidden_state    # [B, N_txt, 768]
        return txt_emb