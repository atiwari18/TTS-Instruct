from transformers import AutoTokenizer
from models.qwen_classifer import QwenBinaryClassifier
from RL.grpo import GRPO
from utils.utils import load_script
import argparse

def parse_args():
    parser = argparse.ArgumentParser("GRPO RL Synthetic RUN")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--save_dir", type=str, default="saved_policy/nov-19-2026-run-1")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--K", type=int, default=2)

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
    script1 = load_script("./scripts//script1.txt", args.K)
    script2 = load_script("./scripts//script2.txt", args.K)
    script3 = load_script("./scripts//script3.txt", args.K)
    script4 = load_script("./scripts//script4.txt", args.K)
    script5 = load_script("./scripts//script5.txt", args.K)
    script6 = load_script("./scripts//script6.txt", args.K)
    script7 = load_script("./scripts//script7.txt", args.K)
    script8 = load_script("./scripts//script8.txt", args.K)
    script9 = load_script("./scripts//script9.txt", args.K)
    script10 = load_script("./scripts//script10.txt", args.K)
    script11 = load_script("./scripts//script11.txt", args.K)
    scripts = [script1, script2, script3, script4, script5, script6, script7, script8, script9, script10]

    grpo.train_synthetic(model, tokenizer, scripts, args.steps, args.group_size, args.lr, args.save_dir)

    grpo.evaluate_policy(model, tokenizer, script11, 30, args.save_dir)