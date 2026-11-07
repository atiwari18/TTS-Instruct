import torch
import torch.nn.functional as F
from torch.optim import AdamW
from utils.utils import save_policy
import numpy as np
from pathlib import Path

class GRPO():
    def sample_pause_sequence(self, model, tokenizer, script_text, num_gaps, budget_K, keep_grad=False, decisions=None, device="cuda"):
        is_sampling = decisions is None
        sampled_decisions = [] if is_sampling else None
        log_probs, free_mask = [], []
        remaining = budget_K
        prev_decisions = []

        for gap_idx in range(num_gaps):
            forced = (remaining == 0) or (remaining == num_gaps - gap_idx)
            prompt = (
                f"Script:\n{script_text}\n\n"
                f"Gap {gap_idx + 1}/{num_gaps}. "
                f"Pauses left to place: {remaining}. "
                f"Previous decisions: {prev_decisions}\n"
                f"Should we place a pause here? Answer with 0 (no) or 1 (yes)."
            )

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if is_sampling:
                with torch.no_grad():
                    out = model(**inputs)
                    # Critical: move to float32 before softmax
                    logits = out["logits"].float()
                    probs = F.softmax(logits, dim=-1)

                    # Hard safety against numerical issues
                    probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
                    probs = torch.clamp(probs, min=1e-8, max=1.0)
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                # Hard constraints
                if remaining == 0:
                    action = 0
                elif remaining == (num_gaps - gap_idx):
                    action = 1
                else:
                    # Final safety check before multinomial
                    if torch.isnan(probs).any() or (probs < 0).any() or (probs > 1).any():
                        print(f"[WARNING] Invalid probs at gap {gap_idx}: {probs}")
                        probs = torch.tensor([[0.5, 0.5]], device=device)
                    action = torch.multinomial(probs, num_samples=1).item()

                sampled_decisions.append(action)
            else:
                # Recompute path (for gradients)
                action = decisions[gap_idx]
                if keep_grad:
                    out = model(**inputs)
                else:
                    with torch.no_grad():
                        out = model(**inputs)

                logits = out["logits"].float()
                probs = F.softmax(logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
                probs = torch.clamp(probs, min=1e-8, max=1.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            # Record log-prob of the taken action
            log_prob = torch.log(probs[0, action] + 1e-10)
            log_probs.append(log_prob)
            free_mask.append(0.0 if forced else 1.0)

            # Update state
            if action == 1:
                remaining -= 1
            prev_decisions.append(action)

        if is_sampling:
            return sampled_decisions, None
        else:
            return decisions, log_probs, free_mask

    def synthetic_reward(self, decisions, K):
        positions = []

        for i, d in enumerate(decisions):
            if d == 1:
                positions.append(i)

        if len(positions) != K:
            return 0.0

        avg_pos = sum(positions) / K

        return avg_pos / (len(decisions) - 1)

    def grpo_steps(self, model, tokenizer, script_text, num_gaps, budget_K, group_size, device="cuda"):
        model.train()
        all_decisions = []
        rewards = []

        #Sampling
        for _ in range(group_size):
            decisions, _ = self.sample_pause_sequence(
                model, tokenizer, script_text, num_gaps, budget_K,
                keep_grad=False, decisions=None, device=device
            )
            all_decisions.append(decisions)
            rewards.append(self.synthetic_reward(decisions, budget_K))

        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

        #Advantage with group normalization
        mean_r = rewards.mean()
        std_r = rewards.std(unbiased=False) + 1e-8
        advantages = (rewards - mean_r) / std_r

        #Policy gradient phase
        policy_loss = 0.0
        for i in range(group_size):
            _, log_probs, free_mask = self.sample_pause_sequence(
                model, tokenizer, script_text, num_gaps, budget_K,
                keep_grad=True, decisions=all_decisions[i], device=device
            )

            lp = torch.stack(log_probs)
            mask = torch.tensor(free_mask, device=device)
            if mask.sum() == 0:
                continue

            seq_logprob = (lp * mask).sum()
            adv = torch.clamp(advantages[i], -1.5, 1.5)
            policy_loss = policy_loss - (seq_logprob * adv)

        policy_loss = policy_loss / group_size
        return policy_loss, rewards.mean().item()

    def evaluate_policy(self, model, tokenizer, script, num_samples, save_dir, device="cuda"):
        model.eval()
        rewards = []
        all_decisions = []

        with torch.no_grad():
            for _ in range(num_samples):
                decisions, _ = self.sample_pause_sequence(
                            model, tokenizer, script["text"], 
                            script["num_gaps"], script["budget_K"], device
                        )

                r = self.synthetic_reward(decisions, script["budget_K"])
                rewards.append(r)
                all_decisions.append(decisions)

        avg_reward = sum(rewards) / len(rewards)

        save_dir = save_dir + "/policy_evaluation.txt"
        Path(save_dir).parent.mkdir(parents=True, exist_ok=True)

        with open(save_dir, "w") as file:
            print(f"Average synthetic reward on novel script: {avg_reward:.3f}")
            file.write(f"Average synthetic reward on novel script: {avg_reward:.3f}\n\n")

            for script, pause_duration in zip(script["sentences"], all_decisions[20]):
                if pause_duration == 1:
                    text = script + " " + "[PAUSE]" + " \n"
                    file.write(text)
                else:
                    text = script + " " + "[NO PAUSE]" + " \n"
                    file.write(text)

        return

    def train_synthetic(self, model, tokenizer, scripts, steps, group_size, lr, save_dir):
        
        device = next(model.parameters()).device
        optimizer = AdamW(model.classifier.parameters(), lr=lr)

        for step in range(steps):
            #pick one script 
            script = scripts[step % len(scripts)]

            loss, avg_reward = self.grpo_steps(
                model, tokenizer, script["text"], script["num_gaps"], script["budget_K"], 
                group_size=group_size, device=device
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 5 == 0:
                print(f"Step {step:4d} | loss {loss.item():.4f} | avg reward {avg_reward:.3f}")

        save_policy(model, save_dir)

        self.evaluate_policy(model, tokenizer, script, 30, save_dir, device="cuda")