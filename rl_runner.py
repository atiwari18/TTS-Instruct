from transformers import AutoTokenizer
from models.qwen_classifer import QwenBinaryClassifier
from RL.grpo import GRPO
from utils.utils import load_script
import argparse

def parse_args():
    parser = argparse.ArgumentParser("GRPO RL Synthetic RUN")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--save_dir", type=str, default="saved_policy/nov-6-2026-run")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = QwenBinaryClassifier(model_name, freeze_base=True)
    grpo = GRPO()

    #Load Scipts
    script1 = load_script("scripts\\script1.txt")
    script2 = load_script("scripts\\script2.txt")
    scripts = [script1, script2]

    #grpo.train_synthetic(model, tokenizer, scripts, args.steps, args.group_size, args.lr, args.save_dir)

    grpo.evaluate_policy(model, tokenizer, script2, 30, args.save_dir, device="cuda")