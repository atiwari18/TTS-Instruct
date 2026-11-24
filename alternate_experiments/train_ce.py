import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer

from utils.utils import load_script


class QwenPausePolicy(nn.Module):
    def __init__(self, model_name: str, use_lora: bool = True, rank: int = 8,
                 alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.model_name = model_name
        self.use_lora = use_lora
        base = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        hidden_size = base.config.hidden_size
        if use_lora:
            config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.backbone = get_peft_model(base, config)
            self.backbone.print_trainable_parameters()
        else:
            self.backbone = base
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        device = self.backbone.get_input_embeddings().weight.device
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),
        ).to(device=device, dtype=torch.float32)
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.use_lora:
            self.backbone.eval()
        return self

    def forward(self, input_ids, attention_mask=None):
        if self.use_lora:
            output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state
        if attention_mask is None:
            pooled = hidden[:, -1, :]
        else:
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            last = positions.masked_fill(~attention_mask.to(hidden.device).bool(), -1).amax(1)
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        pooled = pooled.to(device=self.classifier[1].weight.device, dtype=torch.float32)
        return self.classifier(pooled)


def optimal_labels(sentences, budget):
    n = len(sentences)
    if not 0 <= budget <= n:
        raise ValueError(f"Budget {budget} must be between zero and {n}")
    # Stable tie-breaking: earlier positions first.
    best = set(sorted(range(n), key=lambda i: -len(sentences[i]))[:budget])
    return [int(i in best) for i in range(n)]


def make_prompt(sentences, boundary, selected, pauses_left, explain_rule=False):
    numbered_script = "\n".join(
        f"[{i}] {sentence}" for i, sentence in enumerate(sentences, 1)
    )
    instruction = (
        "Task: Place exactly the allowed number of pauses after the longest "
        "sentences, measured by character count; earlier sentences win ties.\n\n"
        if explain_rule else ""
    )
    return (
        instruction
        + f"Full script:\n{numbered_script}\n\n"
        + f"Consider a pause immediately after sentence [{boundary}].\n"
        + f"Pauses already placed after sentences: {selected}.\n"
        + f"Pauses left, including this boundary: {pauses_left}.\n"
        + "Should a pause go here? Answer 0 (no) or 1 (yes):"
    )


def get_logits(model, tokenizer, sentences, boundary, selected, pauses_left,
               max_prompt_tokens, explain_rule):
    prompt = make_prompt(sentences, boundary, selected, pauses_left, explain_rule)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False,
                       return_token_type_ids=False)
    if inputs["input_ids"].shape[1] > max_prompt_tokens:
        raise ValueError(f"Prompt longer than {max_prompt_tokens} tokens")
    device = model.backbone.get_input_embeddings().weight.device
    inputs = {key: val.to(device) for key, val in inputs.items()}
    return model(**inputs).float()


def train_one_script(model, tokenizer, optimizer, script, positive_weight,
                     max_prompt_tokens, explain_rule):
    sentences, budget = script["sentences"], script["budget_K"]
    labels = optimal_labels(sentences, budget)
    selected = []
    optimizer.zero_grad(set_to_none=True)
    total_ce = 0.0
    total_weight = 0.0
    trained = 0
    positive_count = 0

    for i, target_action in enumerate(labels):
        boundary = i + 1
        pauses_left = budget - len(selected)
        positions_left = len(labels) - i
        # Forced decisions have no trainable policy choice in our sampler.
        if pauses_left == 0:
            assert target_action == 0
        elif pauses_left == positions_left:
            assert target_action == 1
        else:
            logits = get_logits(model, tokenizer, sentences, boundary,
                                selected, pauses_left, max_prompt_tokens, explain_rule)
            target = torch.tensor([target_action], dtype=torch.long, device=logits.device)
            ce = F.cross_entropy(logits, target)
            weight = positive_weight if target_action == 1 else 1.0
            # Backprop once per decision to avoid retaining all Qwen graphs.
            (ce * weight).backward()
            total_ce += ce.detach().item()
            total_weight += weight
            trained += 1
            positive_count += target_action
        # Teacher forcing: update history using the ground-truth action.
        if target_action:
            selected.append(boundary)

    grad_norm = 0.0
    lora_grad_tensors = 0
    if total_weight:
        trainable = [p for p in model.parameters() if p.requires_grad]
        for param in trainable:
            if param.grad is not None:
                param.grad.div_(total_weight)
        for name, param in model.backbone.named_parameters():
            if "lora_" in name and param.grad is not None and param.grad.abs().sum().item() > 0:
                lora_grad_tensors += 1
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).item())
        optimizer.step()

    return {
        "ce": total_ce / max(trained, 1),
        "trainable_decisions": trained,
        "positive_decisions": positive_count,
        "grad_norm": grad_norm,
        "lora_grad_tensors": lora_grad_tensors,
    }


@torch.no_grad()
def greedy_rollout(model, tokenizer, script, max_prompt_tokens, explain_rule):
    model.eval()
    sentences, budget = script["sentences"], script["budget_K"]
    selected, decisions = [], []
    for i in range(len(sentences)):
        boundary = i + 1
        pauses_left = budget - len(selected)
        positions_left = len(sentences) - i
        if pauses_left == 0:
            action = 0
        elif pauses_left == positions_left:
            action = 1
        else:
            logits = get_logits(model, tokenizer, sentences, boundary, selected,
                                pauses_left, max_prompt_tokens, explain_rule)
            action = int(logits.argmax(dim=-1).item())
        decisions.append(action)
        if action:
            selected.append(boundary)
    return decisions


@torch.no_grad()
def teacher_forced_accuracy(model, tokenizer, script, max_prompt_tokens, explain_rule):
    model.eval()
    sentences, budget = script["sentences"], script["budget_K"]
    labels = optimal_labels(sentences, budget)
    selected = []
    correct = total = 0
    for i, target in enumerate(labels):
        left = budget - len(selected)
        positions_left = len(labels) - i
        if 0 < left < positions_left:
            logits = get_logits(model, tokenizer, sentences, i + 1,
                                selected, left, max_prompt_tokens, explain_rule)
            correct += int(int(logits.argmax(-1).item()) == target)
            total += 1
        if target:
            selected.append(i + 1)
    return correct / total if total else 1.0


def reward(decisions, labels, budget):
    return sum(bool(a) and bool(b) for a, b in zip(decisions, labels)) / budget if budget else 1.0


def evaluate(model, tokenizer, scripts, max_prompt_tokens, explain_rule):
    return sum(
        reward(greedy_rollout(model, tokenizer, s, max_prompt_tokens, explain_rule),
               optimal_labels(s["sentences"], s["budget_K"]), s["budget_K"])
        for s in scripts
    ) / len(scripts)


def inspect_policy(model, tokenizer, script, max_prompt_tokens, explain_rule):
    sentences, budget = script["sentences"], script["budget_K"]
    targets = optimal_labels(sentences, budget)
    selected = []
    model.eval()
    with torch.no_grad():
        for i, target in enumerate(targets):
            left = budget - len(selected)
            positions_left = len(targets) - i
            if left == 0:
                action, prob = 0, None
            elif left == positions_left:
                action, prob = 1, None
            else:
                logits = get_logits(model, tokenizer, sentences, i + 1,
                                    selected, left, max_prompt_tokens, explain_rule)
                prob = float(logits.softmax(-1)[0, 1].item())
                action = int(logits.argmax(-1).item())
            prob_str = "forced" if prob is None else f"{prob:.6f}"
            print(f"Sentence {i+1:2d} | Length {len(sentences[i]):3d} | "
                  f"Target {target} | P(pause) {prob_str:>8} | Action {action}")
            if action:
                selected.append(i + 1)


def save_checkpoint(model, tokenizer, args):
    directory = Path(args.save_dir)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.classifier.state_dict(), directory / "classifier.pt")
    if model.use_lora:
        model.backbone.save_pretrained(directory / "lora_adapter")
    tokenizer.save_pretrained(directory / "tokenizer")
    with (directory / "training_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved classifier and {'LoRA adapter' if model.use_lora else 'frozen model configuration'} to {directory}")


def parse_args():
    parser = argparse.ArgumentParser("LoRA supervised pause-policy diagnostic")
    parser.add_argument("--model_name", default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--train_scripts", type=int, nargs="+", default=[1],
                        help="Indices of training script files; defaults to one-script diagnostic")
    parser.add_argument("--held_out_script", type=int, default=11)
    parser.add_argument("--scripts_dir", default="scripts")
    parser.add_argument("--save_dir", default="saved_policy/ce-lora")
    parser.add_argument("--lora_lr", type=float, default=2e-4)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--positive_weight", type=float, default=6.0)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_prompt_tokens", type=int, default=4096)
    parser.add_argument("--explain_rule", action="store_true",
                        help="Ablation: explicitly tell the model the longest-sentence rule")
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.positive_weight <= 0:
        raise ValueError("--positive_weight must be positive")
    if args.held_out_script in args.train_scripts:
        raise ValueError("Held-out script must not also be a training script")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = QwenPausePolicy(args.model_name, use_lora=not args.no_lora,
                            rank=args.rank, alpha=args.alpha,
                            dropout=args.lora_dropout)
    groups = [{"params": list(model.classifier.parameters()), "lr": args.head_lr}]
    if model.use_lora:
        lora_params = [p for p in model.backbone.parameters() if p.requires_grad]
        assert lora_params, "No trainable LoRA parameters were found"
        groups.insert(0, {"params": lora_params, "lr": args.lora_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)

    path = Path(args.scripts_dir)
    training = [load_script(str(path / f"script{i}.txt"), args.K)
                for i in args.train_scripts]
    held_out = [load_script(str(path / f"script{args.held_out_script}.txt"), args.K)]
    n = len(held_out[0]["sentences"])
    print(f"Uniform-random held-out expected reward: {args.K / n:.4f}")
    print("Untrained greedy train reward:",
          f"{evaluate(model, tokenizer, training, args.max_prompt_tokens, args.explain_rule):.3f}")
    print("Untrained greedy held-out reward:",
          f"{evaluate(model, tokenizer, held_out, args.max_prompt_tokens, args.explain_rule):.3f}")

    for epoch in range(1, args.epochs + 1):
        random.shuffle(training)
        epoch_results = []
        for script in training:
            model.train()
            epoch_results.append(train_one_script(
                model, tokenizer, optimizer, script,
                args.positive_weight, args.max_prompt_tokens, args.explain_rule))
        avg_ce = sum(x["ce"] for x in epoch_results) / len(epoch_results)
        train_reward = evaluate(model, tokenizer, training,
                                args.max_prompt_tokens, args.explain_rule)
        held_reward = evaluate(model, tokenizer, held_out,
                               args.max_prompt_tokens, args.explain_rule)
        tf_train = sum(teacher_forced_accuracy(model, tokenizer, s,
                        args.max_prompt_tokens, args.explain_rule)
                       for s in training) / len(training)
        mean_norm = sum(x["grad_norm"] for x in epoch_results) / len(epoch_results)
        lora_nonzero = sum(x["lora_grad_tensors"] for x in epoch_results)
        print(f"Epoch {epoch:3d}/{args.epochs} | CE {avg_ce:.4f} | "
              f"train reward {train_reward:.3f} | held-out {held_reward:.3f} | "
              f"teacher-forced acc {tf_train:.3f} | grad norm {mean_norm:.3f} | "
              f"nonzero LoRA grad tensors {lora_nonzero}", flush=True)

    print("\nTraining-script inspection:")
    inspect_policy(model, tokenizer, training[0], args.max_prompt_tokens, args.explain_rule)
    print("Training greedy actions:", greedy_rollout(model, tokenizer, training[0],
                                                      args.max_prompt_tokens, args.explain_rule))
    print("Training optimal actions:", optimal_labels(training[0]["sentences"], training[0]["budget_K"]))
    print("\nHeld-out inspection:")
    inspect_policy(model, tokenizer, held_out[0], args.max_prompt_tokens, args.explain_rule)
    print("Held-out greedy actions:", greedy_rollout(model, tokenizer, held_out[0],
                                                      args.max_prompt_tokens, args.explain_rule))
    print("Held-out optimal actions:", optimal_labels(held_out[0]["sentences"], held_out[0]["budget_K"]))
    save_checkpoint(model, tokenizer, args)


if __name__ == "__main__":
    main()