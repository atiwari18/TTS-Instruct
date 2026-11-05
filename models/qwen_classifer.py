import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class QwenBinaryClassifier(nn.Module):
    def __init__(self, model_name="Qwen/Qwen3-8B-Base", freeze_base=True):
        super().__init__()

        #Load the base model WITHOUT its language modeling head. 
        #AutoModel gives us the raw hidden states isntead of next-token prediction logit.
        self.base_model = AutoModel.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )

        #Freeze LLM weights
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        #Hidden deminasions from the model config
        hidden_size = self.base_model.config.hidden_size

        #Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1), 
            nn.Linear(hidden_size, hidden_size // 2), 
            nn.ReLU(), 
            nn.Linear(hidden_size // 2, 2)
        )

        self.classifier = self.classifier.to(self.base_model.device)

    def forward(self, input_ids, attention_mask=None):
        #Pass token_ids through base LLM
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        #hidden_state: (batch, sequence_len, hidden_size)
        hidden_states = outputs.last_hidden_state

        batch_size = input_ids.shape[0]

        #Find the index of last token for each sequence in the batch.
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
        else: 
            sequence_lengths = torch.tensor( [input_ids.shape[1] - 1] * batch_size, device=input_ids.device)

        #Extract onlt the hidden state of the last token
        last_hidden_states = hidden_states[torch.arange(batch_size, device=input_ids.device), sequence_lengths]

        #Pass the extracted hidden state through the classifier
        logits = self.classifier(last_hidden_states)

        return {"logits": logits}
