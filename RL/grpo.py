import math
from pathlib import Path
import torch
from torch.optim import AdamW
from utils.utils import save_policy

class GRPO():
    def __init__(self, exploration=0.1):
        if not 0 <= exploration < 1:
            raise ValueError("exploration must be in [0, 1)")
        
        self.exploration = exploration
        self.max_prompt_tokens=4096

    def sample_pause_sequence(self, model, tokenizer, sentences, budget):
        numbered_script = "\n".join(
            f"[{i}] {sentence}"
            for i, sentence in enumerate(sentences, start=1)
        )
        decisions = []
        selected = []
        rollout_log_prob = torch.zeros((), device="cuda")

        for boundary in range(1, len(sentences) + 1):
            pauses_left = budget - len(selected)
            boundaries_left = len(sentences) - boundary + 1

            if pauses_left == 0:
                action = 0
            elif pauses_left == boundaries_left:
                action = 1
            else:
                prompt = (
                    f"Full script:\n{numbered_script}\n\n"
                    f"Consider a pause immediately after sentence [{boundary}].\n"
                    f"Pauses already placed after sentences: {selected}.\n"
                    f"Pauses left, including this boundary: {pauses_left}.\n"
                    "Should a pause go here? Answer 0 (no) or 1 (yes):"
                )
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=False,
                    return_token_type_ids=False,
                )
                if inputs["input_ids"].shape[1] > self.max_prompt_tokens:
                    raise ValueError("Full script prompt exceeds the token limit")

                inputs = {
                    key: value.to(torch.device("cuda"))
                    for key, value in inputs.items()
                }

                logits = model(**inputs)["logits"][0].float()
                log_probs = logits.log_softmax(dim=-1)

                if self.exploration:
                    q_yes = pauses_left / boundaries_left
                    baseline = logits.new_tensor([1 - q_yes, q_yes])
                    log_probs = torch.logaddexp(
                        log_probs + math.log1p(-self.exploration),
                        baseline.log() + math.log(self.exploration),
                    )

                action = torch.multinomial(log_probs.detach().exp(), 1).item()
                rollout_log_prob = rollout_log_prob + log_probs[action]

            decisions.append(action)
            if action == 1:
                selected.append(boundary)

        return decisions, rollout_log_prob

    def longest_reward(self, decisions, sentences, budget):
        # Length is measured in characters. Earlier sentences win ties.
        target = set(sorted(range(len(sentences)), key=lambda i: -len(sentences[i]))[:budget])

        chosen = {i for i, action in enumerate(decisions) if action == 1}

        return len(chosen & target) / budget


    def grpo_steps(self, model, tokenizer, script, group_size):
        sentences, budget = script["sentences"], script["budget_K"]

        if not sentences or not 0 <= budget <= len(sentences):
            raise ValueError("Expected a nonempty script and a valid pause budget")
        if group_size < 2:
            raise ValueError("Need at least two rollouts per group")
        if budget == 0 or budget == len(sentences):
            return None, 1.0

        model.eval()  # Gradients still flow through the classifier head.

        rewards = []
        rollout_log_probs = []
        for _ in range(group_size):
            decisions, log_prob = self.sample_pause_sequence(
                model, tokenizer, sentences, budget
            )

            reward = self.longest_reward(decisions, sentences, budget)

            rewards.append(reward)
            rollout_log_probs.append(log_prob)

        log_probs = torch.stack(rollout_log_probs)
        rewards = log_probs.new_tensor(rewards)
        mean_reward = rewards.mean()
        std_reward = rewards.std(unbiased=False)

        if std_reward.item() == 0:
            return None, mean_reward.item()

        advantages = (rewards - mean_reward) / std_reward
        loss = -(log_probs * advantages).mean()
        return loss, mean_reward.item()

    def evaluate_policy(self, model, tokenizer, script, num_samples, save_dir):
        model.eval()
        rewards = []
        all_decisions = []

        with torch.no_grad():
            for _ in range(num_samples):
                decisions, _ = self.sample_pause_sequence(
                            model, tokenizer, script["sentences"], script["budget_K"])

                r = self.longest_reward(decisions, script["sentences"], script["budget_K"])
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
        if not scripts or steps < 1 or group_size < 2:
            raise ValueError("Training needs scripts, steps >= 1, and group_size >= 2")

        optimizer = AdamW(model.classifier.parameters(), lr=lr)
        updates = 0
        total_reward = 0.0

        for step in range(steps):
            script = scripts[step % len(scripts)]
            optimizer.zero_grad(set_to_none=True)
            loss, avg_reward = self.grpo_steps(model, tokenizer, script, group_size)
            total_reward += avg_reward

            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                updates += 1

            if (step + 1) % 5 == 0 or step == 0 or step + 1 == steps:
                print(
                    f"Step {step + 1:4d}/{steps} | mean reward "
                    f"{total_reward / (step + 1):.3f} | updates {updates}/{step + 1}"
                )

        save_policy(model, save_dir)
        self.evaluate_policy(model, tokenizer, script, 30, save_dir)