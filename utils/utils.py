import re
import torch 
from pathlib import Path

def load_script(path: str):
    text = Path(path).read_text(encoding="utf-8").strip()

    #Split on punctuation (. ? !)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    num_gaps = len(sentences) - 1

    return {
        "text" : text, 
        "sentences" : sentences, 
        "num_gaps" : num_gaps, 
        "budget_K" : max(1, num_gaps // 3)
    }

def save_policy(model, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    #Save the classification head
    torch.save(model.classifier.state_dict(), save_dir / "classifier.pt")

    print(f"Saved classification head to {save_dir}")

    return    