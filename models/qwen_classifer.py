import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, TaskType, get_peft_model

class QwenBinaryClassifier(nn.Module):
    def __init__(self, model_name="Qwen/Qwen3-8B-Base", use_lora: bool = True, rank: int = 8,
                 alpha: int = 16, dropout: float = 0.05):
        super().__init__()

        self.model_name = model_name
        self.use_lora = use_lora

        #Load the base model WITHOUT its language modeling head. 
        #AutoModel gives us the raw hidden states isntead of next-token prediction logit.
        self.base_model = AutoModel.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )

        #Hidden deminasions from the model config
        hidden_size = self.base_model.config.hidden_size

        #Generate LoRA configuration
        if self.use_lora:
            config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=["q_proj", "v_proj"], bias="none", task_type=TaskType.FEATURE_EXTRACTION)
            self.backbone = get_peft_model(self.base_model, config)
            self.backbone.print_trainable_parameters()

        #Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1), 
            nn.Linear(hidden_size, hidden_size // 2), 
            nn.ReLU(), 
            nn.Linear(hidden_size // 2, 2)
        ).to(device="cuda", dtype=torch.float32)

        #Start the GRPRO with equal scores for each candidate
        # nn.init.zeros_(self.classifier[-1].weight)
        # nn.init.zeros_(self.classifier[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        return self

    def forward(self, input_ids, attention_mask=None):
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        hidden = output.last_hidden_state

        if attention_mask is None:
            pooled = hidden[:, -1, :]
        else:
            mask = attention_mask.to(device=hidden.device, dtype=torch.bool)
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            last_indices = positions.masked_fill(~mask, -1).amax(dim=1)
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            pooled = hidden[rows, last_indices]

        pooled = pooled.to(self.classifier[1].weight)
        
        return {"logits": self.classifier(pooled)}
