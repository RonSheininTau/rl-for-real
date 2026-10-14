"""Custom GRPO Trainer with reasoning-gym reward functions."""

import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from reasoning_gym.utils import extract_answer

from .dataset import ReasoningGymDataset

# Try to import wandb for logging (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class CustomGRPOTrainer(GRPOTrainer):
    """
    Extended GRPOTrainer with reasoning-gym reward functions.
    
    Includes:
        - Accuracy reward based on reasoning-gym's scoring
        - Format reward for proper <think>/<answer> tags
        - Optional reward variance logging
    """
    
    def __init__(
        self,
        model,
        args: GRPOConfig,
        tokenizer,
        train_dataset: ReasoningGymDataset,
        eval_dataset: ReasoningGymDataset,
        log_reward_variance: bool = True,
        verbose_variance_logging: bool = False,
        batch_counts_as_step: bool = False,
        use_process_reward: bool = False,
        process_reward_model: str = None,
        process_reward_weight: float = 1.0,
        process_reward_sample_rate: float = 0.15,
        process_reward_criteria: list[str] = None,
    ):
        reward_funcs = [self._accuracy_reward, self._format_reward]
        if use_process_reward:
            unknown = set(process_reward_criteria or []) - set(self._PROCESS_REWARD_CRITERIA)
            if unknown:
                raise ValueError(
                    f"Unknown process_reward_criteria: {unknown}. "
                    f"Valid: {list(self._PROCESS_REWARD_CRITERIA)}"
                )
            reward_funcs.append(self._process_reward)
            if args.reward_weights and len(args.reward_weights) < len(reward_funcs):
                args.reward_weights = list(args.reward_weights) + [process_reward_weight]

        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        self.log_reward_variance = log_reward_variance
        self.verbose_variance_logging = verbose_variance_logging
        self.batch_counts_as_step = batch_counts_as_step
        self._step_count = 0

        self.use_process_reward = use_process_reward
        self.process_reward_sample_rate = process_reward_sample_rate
        self.process_reward_criteria_names = process_reward_criteria or []
        if use_process_reward:
            self._judge_tokenizer = AutoTokenizer.from_pretrained(process_reward_model)
            self._judge_model = AutoModelForCausalLM.from_pretrained(
                process_reward_model, torch_dtype=torch.bfloat16
            ).to(self.accelerator.device)
            self._judge_model.eval()
    
    def log(self, logs: dict, start_time: float = None) -> None:
        """Override log to manually log to wandb with correct x-axis."""
        # First, let parent class merge _metrics into logs
        # (GRPOTrainer.log merges self._metrics[mode] into logs before calling super().log)
        mode = "train" if self.model.training else "eval"
        if hasattr(self, '_metrics') and mode in self._metrics:
            metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items() if val}
            if mode == "eval":
                metrics = {f"eval_{key}": val for key, val in metrics.items()}
            logs = {**logs, **metrics}
        
        # Now log to wandb with our custom step
        if self.state is not None and WANDB_AVAILABLE and wandb.run is not None:
            # Determine which step metric to use
            if self.batch_counts_as_step:
                step_value = self.state.global_step * self.args.gradient_accumulation_steps
            else:
                step_value = self.state.global_step
            
            try:
                wandb_logs = {}
                
                # Add all train metrics
                for key, value in logs.items():
                    # Convert string numbers to float (HuggingFace logs some metrics as strings)
                    if isinstance(value, str):
                        try:
                            value = float(value)
                        except (ValueError, TypeError):
                            continue  # Skip non-numeric strings
                    
                    if isinstance(value, (int, float)):
                        # Prefix with train/ if not already prefixed
                        if not key.startswith("train/") and not key.startswith("eval/"):
                            wandb_logs[f"train/{key}"] = value
                        else:
                            wandb_logs[key] = value
                
                # Use step parameter to force wandb to use our step value as the x-axis
                if wandb_logs:
                    wandb.log(wandb_logs, step=step_value)
            except Exception as e:
                # Log error but don't crash training
                print(f"Warning: Failed to log to wandb at step {step_value}: {e}")
        
        # Clear metrics before calling parent (to avoid double logging in parent's log method)
        if hasattr(self, '_metrics') and mode in self._metrics:
            self._metrics[mode].clear()
        
        # Call grandparent's log (skip GRPOTrainer.log since we already handled metrics)
        from transformers import Trainer
        Trainer.log(self, logs, start_time)

    def _accuracy_reward(self, completions: list[str], **kwargs) -> list[float]:
        """Compute accuracy reward using reasoning-gym's scoring function."""
        assert "item" in kwargs, "The 'item' argument must be provided."
        assert len(kwargs["item"]) == len(completions)
        
        answers = [extract_answer(c) for c in completions]
        rewards = [
            self.train_dataset.data.score_answer(answer, item) 
            for answer, item in zip(answers, kwargs["item"])
        ]
        
        if self.log_reward_variance:
            self._log_variance(rewards, answers, completions)
        
        return rewards
    
    def _log_variance(self, rewards: list[float], answers: list[str], completions: list[str]):
        """Log the percentage of prompts with reward variance across rollouts."""
        num_gens = self.args.num_generations
        num_prompts = len(rewards) // num_gens
        
        if num_prompts == 0:
            return
        
        prompts_with_reward_variance = 0
        prompts_with_answer_variance = 0
        
        for i in range(num_prompts):
            start_idx = i * num_gens
            end_idx = (i + 1) * num_gens
            group_rewards = rewards[start_idx:end_idx]
            group_answers = answers[start_idx:end_idx]
            
            has_reward_variance = len(set(group_rewards)) > 1
            has_answer_variance = len(set(group_answers)) > 1
            
            if has_reward_variance:
                prompts_with_reward_variance += 1
            if has_answer_variance:
                prompts_with_answer_variance += 1
            
            # Verbose: print full completions when there's variance
            if self.verbose_variance_logging and has_answer_variance:
                group_completions = completions[start_idx:end_idx]
                print(f"\n  {'─'*60}")
                print(f"  📊 Prompt {i}: rewards={group_rewards}")
                for gen_idx, (comp, rew, ans) in enumerate(zip(group_completions, group_rewards, group_answers)):
                    comp_preview = comp[:300].replace('\n', ' ↵ ')
                    if len(comp) > 300:
                        comp_preview += "..."
                    print(f"    Gen {gen_idx} [reward={rew}]: {comp_preview}")
                    print(f"           extracted: {ans}")
                print(f"  {'─'*60}")
        
        reward_variance_pct = (prompts_with_reward_variance / num_prompts * 100)
        answer_variance_pct = (prompts_with_answer_variance / num_prompts * 100)
        
        self._step_count += 1
        
        if self.verbose_variance_logging:
            print(f"🎲 Step {self._step_count}: {reward_variance_pct:.0f}% prompts have reward variance, "
                  f"{answer_variance_pct:.0f}% have answer variance ({num_prompts} prompts × {num_gens} gens)")
        
        # Log to wandb
        if WANDB_AVAILABLE and wandb.run is not None:
            try:
                wandb.log({
                    "train/reward_variance_pct": reward_variance_pct,
                    "train/answer_variance_pct": answer_variance_pct,
                }, commit=False)
            except:
                pass
        
        # Periodic GPU cleanup
        if self._step_count % 10 == 0:
            torch.cuda.empty_cache()

    def _format_reward(self, completions: list[str], **kwargs) -> list[float]:
        """Reward proper formatting with <think> and <answer> tags."""
        def count_tags(text: str) -> float:
            count = 0.0
            if re.search(r"\s*<think>\s*", text):
                count += 0.25
            if re.search(r"\s*</think>\s*", text):
                count += 0.25
            if re.search(r"\s*<answer>\s*", text):
                count += 0.25
            if re.search(r"\s*</answer>\s*", text):
                count += 0.25
            return count

        return [count_tags(c) for c in completions]

    def _process_reward(self, completions: list[str], **kwargs) -> list[float]:
        """LLM-judge reward rating <think> quality, independent of final-answer correctness.

        Samples whole prompt-groups (not individual completions) so judged vs. unjudged
        never creates an artificial advantage gap within the same group - see
        _log_variance for the same group-slicing pattern.
        """
        n = len(completions)
        num_gens = self.args.num_generations
        num_groups = n // num_gens
        rewards = [0.0] * n
        if num_groups == 0:
            return rewards

        sample_size = max(1, round(num_groups * self.process_reward_sample_rate))
        sampled_groups = random.sample(range(num_groups), min(sample_size, num_groups))

        items = kwargs.get("item", [None] * n)
        for g in sampled_groups:
            start, end = g * num_gens, (g + 1) * num_gens
            for i in range(start, end):
                think = self._extract_think(completions[i])
                if not think:
                    continue
                question = items[i]["question"] if items[i] else ""
                rewards[i] = self._judge_score(question, think)
        return rewards

    @staticmethod
    def _extract_think(text: str) -> str:
        match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    # Registry of candidate rubric criteria. Which ones are active is controlled by
    # process_reward_criteria in the config - this dict is the full menu to pick from.
    # Scored via P(Yes) logprob rather than a generated scalar: small judge models are
    # unreliable at holistic 0.0-1.0 grading but comparably reliable at yes/no
    # classification - same reasoning behind _format_reward's additive binary checks.
    _PROCESS_REWARD_CRITERIA = {
        "engagement": "Does this reasoning explicitly engage with the specific details "
                      "given in the question (not generic or templated text)?",
        "non_contradiction": "Is this reasoning free of self-contradiction (it never "
                             "states two incompatible things)?",
        "self_consistency": "Does the reasoning's own stated conclusion actually follow "
                            "from the steps shown, regardless of whether that conclusion "
                            "is factually correct?",
        "step_structure": "Does the reasoning proceed through clear, discrete steps "
                          "rather than jumping straight to a conclusion?",
        "no_repetition": "Is the reasoning free of pointless repetition of the same "
                         "phrase or sentence?",
    }

    def _judge_score(self, question: str, reasoning: str) -> float:
        criteria = [self._PROCESS_REWARD_CRITERIA[name] for name in self.process_reward_criteria_names]
        if not criteria:
            return 0.0
        scores = [self._yes_prob(question, reasoning, c) for c in criteria]
        return sum(scores) / len(scores)

    def _yes_prob(self, question: str, reasoning: str, criterion: str) -> float:
        prompt = (
            f"Question: {question}\n\nReasoning: {reasoning}\n\n{criterion} "
            "Answer with exactly one word: Yes or No."
        )
        chat = [{"role": "user", "content": prompt}]
        judge_input = self._judge_tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = self._judge_tokenizer(judge_input, return_tensors="pt").to(self._judge_model.device)
        with torch.no_grad():
            logits = self._judge_model(**inputs).logits[0, -1]
        yes_id = self._judge_tokenizer.encode(" Yes", add_special_tokens=False)[0]
        no_id = self._judge_tokenizer.encode(" No", add_special_tokens=False)[0]
        probs = torch.softmax(logits[[yes_id, no_id]], dim=0)
        return probs[0].item()
