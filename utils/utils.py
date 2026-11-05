import re 
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