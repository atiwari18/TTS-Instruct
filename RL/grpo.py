import torch
import torch.nn.functional as F
from torch.optim import AdamW
from utils.utils import save_policy
import numpy as np

class GRPO():
    def __init__(self, model, tokenizer, scripts, num_gaps, budget_K, device):
        self.model = model
        self.tokenizer = tokenizer
        self.scripts = scripts

    #Sample pause sequence 
    def sample_pause_sequence(self, model, tokenizer, script_text, num_gaps, budget_K, device):
        decisions = []
        log_probs = []
        remaining = budget_K

        for gap_idx in range(num_gaps):
            #build a simple prompt for the model
            prompt = (
                f"Script:\n{script_text}\n\n"
                f"Gap {gap_idx+1}/{num_gaps}. "
                f"Pauses left to place: {remaining}. "
                f"Previous decisions: {decisions}\n"
                f"Should we place a pause here?"
            )

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model(**inputs)
                logits = out["logits"]
                probs = F.softmax(logits, dim=-1)

            #If not budget left then "no pause"
            if remaining == 0:
                action = 0

            #If we must place all remaining pauses we force pause
            elif remaining == (num_gaps - gap_idx):
                action = 1

            #Sample from the probabilites
            else:
                action = torch.multinomial(probs, num_samples=1).item()

            #recod logprov of the chosen action
            log_prob = torch.log(probs[0, action] + 1e-10)
            log_probs.append(log_prob)

            decisions.append(action)
            if action == 1:
                remaining -= 1

        return decisions, log_probs

    def synthetic_reward(self, decisions, K):
        positions = []

        for i, d in enumerate(decisions):
            if d == 1:
                positions.append(i)

        if len(positions) != K:
            return 0.0

        avg_pos = sum(positions) / K

        return avg_pos / (len(decisions) - 1)

    def grpo_steps(self, model, tokenizer, script_text, num_gaps, budget_K, group_size, clip_eps=0.1, device="cuda"):
        model.train()
        all_decisions = []
        all_logprobs = []
        rewards = []

        #Sampling phase
        for _ in range(group_size):
            decisions, log_probs = self.sample_pause_sequence(
                model, tokenizer, script_text, num_gaps, budget_K, device
            )

            all_decisions.append(decisions)
            all_logprobs.append(log_probs)

            #Add rewards
            rewards.append(self.synthetic_reward(decisions, budget_K))

        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

        #CALCULATING ADVANTAGE GRPO CORE
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r

        #Compute Loss
        # We need the log-probability of each full sequence under the *current* policy
        # because we already sampled with torch.no_grad, we re-compute the log-probs
        # with gradients enabled (this is the standard way).

        policy_loss = 0.0
        for i in range(group_size):
            decisions = all_decisions[i]

            #Sum of log-probabilites
            seq_logprob = torch.stack(all_logprobs[i]).sum()

            #Importance ratio
            ratio = torch.exp(seq_logprob = seq_logprob.detach())

            adv = advantages[i]
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv
            policy_loss = policy_loss - torch.min(unclipped, clipped)

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
        with open(save_dir, "w") as file:
            print(f"Average synthetic reward on novel script: {avg_reward:.3f}")
            file.write(f"Average synthetic reward on novel script: {avg_reward:.3f}\n\n")

            
            for script, pause_duration in zip(script["text"], decisions[20]):

                if pause_duration == 1:
                    text = script + " " + "[PAUSE]" + " "
                    file.write(text)
                else:
                    text = script + " " + "[NO PAUSE]" + " "
                    file.write(text)

        return

    def train_synthetic(self, model, tokenizer, scripts, steps, group_size, lr, save_dir):
        
        device = next(model.parameters()).device
        optimizer = AdamW(model.classifier.parameters(), lr=lr)

        for step in range(steps):
            #pick one script 
            script = scripts[step % len(scripts)]

            loss, avg_reward = self.grpo_step(
                model, tokenizer, script["text"], script["num_gaps"], script["budget_K"], 
                group_size=group_size, device=device
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 20 == 0:
                print(f"Step {step:4d} | loss {loss.item():.4f} | avg reward {avg_reward:.3f}")

        save_policy(model, save_dir)

        self.evaluate_policy(model, tokenizer, script, 30, save_dir, device="cuda")