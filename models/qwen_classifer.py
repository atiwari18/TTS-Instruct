import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class QwenBinaryClassifier(nn.Module):
    def __init__(self, model_name="Qwen/Qwen3-8B-Base", freeze_base=True):
        super().__init__()

        self.freeze_base = freeze_base

        #Load the base model WITHOUT its language modeling head. 
        #AutoModel gives us the raw hidden states isntead of next-token prediction logit.
        self.base_model = AutoModel.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )

        #Freeze LLM weights
        if freeze_base:
            self.base_model.requires_grad_(False)
            self.base_model.eval()

        #Hidden deminasions from the model config
        hidden_size = self.base_model.config.hidden_size

        #Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1), 
            nn.Linear(hidden_size, hidden_size // 2), 
            nn.ReLU(), 
            nn.Linear(hidden_size // 2, 2)
        ).to(device="cuda", dtype=torch.float32)

        #Start the GRPRO with equal scores for each candidate
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_base:
            self.base_model.eval()
        return self

    def forward(self, input_ids, attention_mask=None):
        track_base_grad = torch.is_grad_enabled() and not self.freeze_base

        with torch.set_grad_enabled(track_base_grad):
            #Pass token_ids through base LLM
            hidden = self.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        if attention_mask is None:
            pooled = hidden[:, -1]
        else:
            mask = attention_mask.to(device=hidden.device, dtype=torch.bool)
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            last_indices = positions.masked_fill(~mask, -1).amax(dim=1)
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            pooled = hidden[rows, last_indices]

        pooled = pooled.to(self.classifier[1].weight)
        
        return {"logits": self.classifier(pooled)}
