import math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from utils.utils import save_policy

class GRPO():
    def __init__(self, exploration=0.1):
        if not 0 <= exploration < 1:
            raise ValueError("exploration must be in [0, 1)")
        
        self.exploration = exploration
        self.max_prompt_tokens=4096
    def _get_action_log_probs(self, model, tokenizer, sentences, boundary, selected, budget):
        numbered_script = "\n".join(
            f"[{i}] {sentence}"
            for i, sentence in enumerate(sentences, start=1)
        )

        pauses_left = budget - len(selected)
        boundaries_left = len(sentences) - boundary + 1

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

        device = next(model.parameters()).device

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        logits = model(**inputs)["logits"][0].float()

        pauses_left = budget - len(selected)
        boundaries_left = len(sentences) - boundary + 1

        q = pauses_left / boundaries_left

        # This helper is called only for free decisions.
        assert 0 < q < 1

        # Difference between the two learned logits.
        learned_logit = logits[1] - logits[0]

        # Convert budget probability to log-odds.
        budget_logit = torch.logit(logits.new_tensor(q))

        # Combine learned preference with budget preference.
        adjusted_logit = learned_logit + budget_logit

        # Convert into a two-action log-probability distribution.
        log_probs = F.log_softmax(
            torch.stack([
                torch.zeros_like(adjusted_logit),
                adjusted_logit
            ]),
            dim=-1
        )

        return log_probs

    @torch.no_grad()
    def sample_pause_sequence(self, model, tokenizer, sentences, budget):
        decisions = []
        selected = []
        free_decisions = []

        for boundary in range(1, len(sentences) + 1):
            pauses_left = budget - len(selected)
            boundaries_left = len(sentences) - boundary + 1

            if pauses_left == 0:
                action = 0
            elif pauses_left == boundaries_left:
                action = 1
            else:
                log_probs = self._get_action_log_probs(model, tokenizer, sentences, boundary, selected, budget)

                action = torch.multinomial(log_probs.exp(), 1).item()

                free_decisions.append({
                    "boundary" : boundary, 
                    "selected_before" : selected.copy(), 
                    "action": action
                })

            decisions.append(action)

            if action == 1:
                selected.append(boundary)

        return {
            "decisions" : decisions, 
            "free_decisions" : free_decisions
        }

    # def longest_reward(self, decisions, sentences, budget):
    #     # Length is measured in characters. Earlier sentences win ties.
    #     target = set(sorted(range(len(sentences)), key=lambda i: -len(sentences[i]))[:budget])

    #     chosen = {i for i, action in enumerate(decisions) if action == 1}

    #     return len(chosen & target) / budget


    def longest_reward(
        self, decisions, sentences, budget,
        exact_weight=0.5
    ):
        if budget == 0:
            return 1.0

        lengths = [len(s) for s in sentences]

        target = set(
            sorted(
                range(len(sentences)),
                key=lambda i: (-lengths[i], i)
            )[:budget]
        )

        chosen = {
            i for i, action in enumerate(decisions)
            if action == 1
        }

        # Original reward
        exact_reward = len(chosen & target) / budget

        # Length-sensitive reward
        selected_length = sum(lengths[i] for i in chosen)
        optimal_length = sum(lengths[i] for i in target)

        length_reward = (
            selected_length / optimal_length
            if optimal_length > 0 else 1.0
        )

        return (
            exact_weight * exact_reward
            + (1 - exact_weight) * length_reward
        )

    def grpo_steps(self, model, tokenizer, script, group_size):
        sentences, budget = script["sentences"], script["budget_K"]

        if not sentences or not 0 <= budget <= len(sentences):
            raise ValueError("Expected a nonempty script and a valid pause budget")
        if group_size < 2:
            raise ValueError("G >= 2")
        if budget == 0 or budget == len(sentences):
            return None, 1.0

        model.eval()  # Gradients still flow through the classifier head.

        rewards = []
        decisions = []

        #Collecting decisions
        for _ in range(group_size):
            pause_sequence = self.sample_pause_sequence(
                model, tokenizer, sentences, budget
            )

            reward = self.longest_reward(pause_sequence["decisions"], sentences, budget)

            rewards.append(reward)
            decisions.append(pause_sequence)

        #Calculating advantages
        rewards = torch.tensor(rewards, dtype=torch.float32)
        mean_reward = rewards.mean().item()
        std_reward = rewards.std(unbiased=False).item()

        print(
            f"Rollout rewards: {rewards.tolist()} | "
            f"Mean: {mean_reward:.3f} | "
            f"Std: {std_reward:.3f}"
        )

        positive_reward_rate = (
            (rewards > 0).float().mean().item()
        )

        print(f"Positive reward rate: {positive_reward_rate:.3f}")

        if std_reward < 1e-8:
            return None, mean_reward

        advantages = (
            (rewards - rewards.mean()) / std_reward
        ).tolist()

        #Replay with gradients
        total_loss = 0.0
        for pause_sequence, advantage in zip(decisions, advantages):
            for d in pause_sequence["free_decisions"]:
                #Reconstruct original decision 
                log_probs = self._get_action_log_probs(model, tokenizer, sentences, d["boundary"], d["selected_before"], budget)

                action = d["action"]

                #Original group-normalized loss
                loss = ( -advantage * log_probs[action] / group_size )

                #Store only a numerical value not the computational graph
                total_loss += loss.detach().item()

                #Accumulate gradients and release forward computation graph
                loss.backward()

        return total_loss, mean_reward


    def evaluate_policy(self, model, tokenizer, script, num_samples, save_dir):
        model.eval()
        rewards = []
        all_decisions = []

        with torch.no_grad():
            for _ in range(num_samples):
                decision = self.sample_pause_sequence(
                            model, tokenizer, script["sentences"], script["budget_K"])

                decisions = decision["decisions"]

                r = self.longest_reward(decisions, script["sentences"], script["budget_K"])
                rewards.append(r)
                all_decisions.append(decisions)

        avg_reward = sum(rewards) / len(rewards)

        save_dir = save_dir + "/policy_evaluation.txt"
        Path(save_dir).parent.mkdir(parents=True, exist_ok=True)

        with open(save_dir, "w") as file:
            #print(f"Average synthetic reward on novel script: {avg_reward:.3f}")
            file.write(f"Synthetic reward on novel script: {rewards[0]:.3f}\n\n")

            for script, pause_duration in zip(script["sentences"], all_decisions[0]):
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

        groups = [{"params": list(model.classifier.parameters()), "lr": lr}]
        
        if model.use_lora:
            lora_params = [p for p in model.backbone.parameters() if p.requires_grad]
            assert lora_params, "No trainable LoRA parameters were found"
            groups.insert(0, {"params": lora_params, "lr": 2e-4})

        optimizer = AdamW(groups, weight_decay=0.01)
        updates = 0
        total_reward = 0.0

        for step in range(steps):
            script = scripts[step % len(scripts)]
            optimizer.zero_grad(set_to_none=True)
            
            loss, avg_reward = self.grpo_steps(model, tokenizer, script, group_size)
            total_reward += avg_reward

            if loss is not None:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
                optimizer.step()
                updates += 1

            #if (step + 1) % 5 == 0 or step == 0 or step + 1 == steps:
            loss_str = f"{loss:.3f}" if loss is not None else "N/A (skipped)"
            print(
                f"Step {step + 1:4d}/{steps} | Loss = {loss_str}| mean reward "
                f"{total_reward / (step + 1):.3f} | updates {updates}/{step + 1}"
            )

        save_policy(model, save_dir)
        self.evaluate_policy(model, tokenizer, script, 30, save_dir)