import copy
import json
import os
import random
import time
import tracemalloc
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "main"
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Strictly separated DEV split (used ONLY to select/optimize edits and compute steering vector)
DEV_PAIRS = [
    ("Fråga: Vad är huvudstaden i Sverige?\nSvar:", " Huvudstaden i Sverige är Stockholm.", " The capital of Sweden is Stockholm."),
    ("Fråga: Vilken färg har himlen en klar dag?\nSvar:", " Himlen är blå en klar dag.", " The sky is blue on a clear day."),
    ("Fråga: Hur många dagar finns det på en vecka?\nSvar:", " Det finns sju dagar på en vecka.", " There are seven days in a week."),
    ("Fråga: Vad dricker man på morgonen som är varmt?\nSvar:", " Många dricker kaffe eller te på morgonen.", " Many people drink coffee or tea in the morning."),
    ("Fråga: Vilket djur säger mjau?\nSvar:", " Det är en katt som säger mjau.", " It is a cat that says meow."),
    ("Fråga: Vad heter Sveriges största sjö?\nSvar:", " Sveriges största sjö heter Vänern.", " Sweden's largest lake is called Vänern."),
]

# Held-out TEST split (never seen during attribution, search, or steering vector extraction)
TEST_PAIRS = [
    ("Fråga: Vilken årstid kommer efter sommaren?\nSvar:", " Efter sommaren kommer hösten.", " After summer comes autumn."),
    ("Fråga: Vad använder man för att skriva på papper?\nSvar:", " Man använder en penna för att skriva.", " You use a pen to write on paper."),
    ("Fråga: Hur många månader har ett år?\nSvar:", " Ett år har tolv månader.", " A year has twelve months."),
    ("Fråga: Vilket språk talas i Sverige?\nSvar:", " I Sverige talas svenska.", " In Sweden people speak Swedish."),
    ("Fråga: Vad gör man när man är trött på kvällen?\nSvar:", " Man går och lägger sig för att sova.", " You go to bed to sleep."),
    ("Fråga: Vilken planet bor vi på?\nSvar:", " Vi bor på planeten jorden.", " We live on the planet Earth."),
    ("Fråga: Vad är motsatsen till kallt?\nSvar:", " Motsatsen till kallt är varmt.", " The opposite of cold is warm."),
    ("Fråga: Var lånar man böcker gratis?\nSvar:", " Man lånar böcker på biblioteket.", " You borrow books at the library."),
    ("Fråga: Vad heter den svenska valutan?\nSvar:", " Den svenska valutan heter krona.", " The Swedish currency is called the krona."),
    ("Fråga: Vilken månad firar man midsommar i Sverige?\nSvar:", " Midsommar firas i juni.", " Midsummer is celebrated in June."),
]

# Regression check: unrelated arithmetic & general factual prompts
REGRESSION_PAIRS = [
    ("Question: What is 14 + 27?\nAnswer:", " 41", " 52"),
    ("Question: Which planet is closest to the Sun?\nAnswer:", " Mercury", " Neptune"),
    ("Question: What is the chemical symbol for water?\nAnswer:", " H2O", " CO2"),
    ("Question: How many continents are there on Earth?\nAnswer:", " Seven", " Twelve"),
    ("Question: Who wrote Hamlet?\nAnswer:", " William Shakespeare", " Charles Dickens"),
    ("Question: What is 9 multiplied by 6?\nAnswer:", " 54", " 45"),
    ("Question: What is the boiling point of water in Celsius?\nAnswer:", " 100", " 50"),
    ("Question: Which gas do plants absorb from the atmosphere?\nAnswer:", " Carbon dioxide", " Helium"),
]


@dataclass
class EvalMetrics:
    condition: str
    dev_sv_win_rate: float
    dev_mean_margin: float
    test_sv_win_rate: float
    test_mean_margin: float
    regression_accuracy: float
    regression_mean_margin: float
    edit_l2_norm: float
    pct_weights_changed: float
    edit_wall_time_s: float
    peak_python_mem_mb: float
    inference_ms_per_token: float
    sample_output: str


def build_concat_ids(tokenizer, prompt: str, continuation: str, device: str) -> Tuple[torch.Tensor, int]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    cont_ids = tokenizer.encode(continuation, add_special_tokens=False)
    full_ids = torch.tensor([prompt_ids + cont_ids], dtype=torch.long, device=device)
    return full_ids, len(prompt_ids)


def sequence_logprob(model, tokenizer, prompt: str, continuation: str, device: str) -> float:
    full_ids, prompt_len = build_concat_ids(tokenizer, prompt, continuation, device)
    with torch.no_grad():
        logits = model(full_ids).logits
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = full_ids[:, 1:]
        tok_lp = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return float(tok_lp[:, prompt_len - 1 :].mean().item())


def evaluate_pairs(
    model, tokenizer, pairs: List[Tuple[str, str, str]], device: str, prefix: str = ""
) -> Tuple[float, float]:
    wins = 0
    margins = []
    for prompt, pos_cont, neg_cont in pairs:
        p = prefix + prompt
        margin = sequence_logprob(model, tokenizer, p, pos_cont, device) - sequence_logprob(
            model, tokenizer, p, neg_cont, device
        )
        margins.append(margin)
        if margin > 0:
            wins += 1
    return wins / len(pairs), float(np.mean(margins))


def generate_sample(model, tokenizer, device: str, prefix: str = "") -> str:
    prompt = prefix + "Fråga: Varför är det viktigt att dricka vatten varje dag?\nSvar:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()


def benchmark_inference_latency(
    model, tokenizer, device: str, prefix: str = "", runs: int = 2, n_tokens: int = 24
) -> float:
    """Returns mean milliseconds per generated token on held-out Swedish prompt."""
    prompt = prefix + "Fråga: Varför är det viktigt att dricka vatten varje dag?\nSvar:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    times = []
    for _ in range(runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=n_tokens,
                min_new_tokens=n_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        gen_tokens = max(int(out.shape[1] - inputs.input_ids.shape[1]), 1)
        times.append(elapsed_ms / gen_tokens)
    return round(float(np.mean(times)), 2)


def measure_edit_cost(fn: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
    """Measures wall-clock runtime (s) and peak Python heap allocation (MB) of an edit step."""
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    finally:
        elapsed_s = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return result, {
        "edit_wall_time_s": round(elapsed_s, 2),
        "peak_python_mem_mb": round(peak_bytes / (1024 * 1024), 2),
    }


def weight_diff_stats(model, base_state: Dict[str, torch.Tensor]) -> Tuple[float, float]:
    """Returns Frobenius L2 norm of delta W and percentage of modified weights."""
    sq_sum = 0.0
    changed = 0
    total = 0
    for name, param in model.named_parameters():
        diff = param.detach().cpu() - base_state[name]
        sq_sum += float(torch.sum(diff * diff).item())
        changed += int(torch.count_nonzero(diff).item())
        total += int(diff.numel())
    pct_changed = (100.0 * changed / total) if total > 0 else 0.0
    return float(np.sqrt(sq_sum)), pct_changed


def get_target_mlp_param(model) -> torch.nn.Parameter:
    target = None
    for name, param in model.named_parameters():
        if "mlp.down_proj.weight" in name:
            target = param  # Final transformer block MLP down_proj
    if target is None:
        raise ValueError("Could not locate mlp.down_proj.weight in model.")
    return target


def direct_weight_attribution_edit(model, tokenizer, dev_pairs, device: str, step_scale: float = 0.02) -> None:
    """Single-pass gradient-weight attribution update (zero closed-loop feedback iterations)."""
    for param in model.parameters():
        param.requires_grad_(False)

    target_param = get_target_mlp_param(model)
    target_param.requires_grad_(True)
    model.zero_grad()

    losses = []
    for prompt, sv_cont, en_cont in dev_pairs:
        for cont, sign in [(sv_cont, -1.0), (en_cont, 1.0)]:
            full_ids, prompt_len = build_concat_ids(tokenizer, prompt, cont, device)
            logits = model(full_ids).logits
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            targets = full_ids[:, 1:]
            tok_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[:, prompt_len - 1 :]
            losses.append(sign * tok_lp.mean())

    loss_total = torch.stack(losses).mean()
    loss_total.backward()

    with torch.no_grad():
        grad = target_param.grad
        row_scores = torch.norm(grad * target_param.data, dim=1)
        top_rows = torch.topk(row_scores, k=min(32, row_scores.shape[0])).indices
        mask = torch.zeros_like(target_param.data)
        mask[top_rows] = 1.0
        update = -step_scale * torch.sign(grad) * mask * torch.std(target_param.data)
        target_param.data.add_(update)

    model.zero_grad()
    target_param.requires_grad_(False)


def feedback_evolutionary_edit(
    model,
    tokenizer,
    dev_pairs,
    device: str,
    target_l2_norm: float,
    steps: int = 10,
    candidates_per_step: int = 4,
) -> None:
    """Closed-loop (1 + lambda) zero-order search in weight space guided by repeated DEV feedback."""
    target_param = get_target_mlp_param(model)
    orig_weight = target_param.data.clone()
    best_weight = orig_weight.clone()
    _, best_margin = evaluate_pairs(model, tokenizer, dev_pairs, device)

    sigma = target_l2_norm / np.sqrt(orig_weight.numel())
    for _ in range(steps):
        improved = False
        for _ in range(candidates_per_step):
            cand = best_weight + torch.randn_like(orig_weight) * sigma
            delta = cand - orig_weight
            norm = torch.norm(delta)
            if norm > 0:
                cand = orig_weight + delta * (target_l2_norm / norm)
            target_param.data.copy_(cand)
            _, margin = evaluate_pairs(model, tokenizer, dev_pairs, device)
            if margin > best_margin:
                best_margin = margin
                best_weight = cand.clone()
                improved = True
        if not improved:
            sigma *= 0.85

    target_param.data.copy_(best_weight)


def random_norm_matched_edit(model, target_l2_norm: float) -> None:
    target_param = get_target_mlp_param(model)
    with torch.no_grad():
        noise = torch.randn_like(target_param.data)
        noise = noise * (target_l2_norm / torch.norm(noise))
        target_param.data.add_(noise)


def compute_steering_vector(model, tokenizer, dev_pairs, layer_idx: int, device: str) -> torch.Tensor:
    """Contrastive activation steering vector (mean Swedish - English residual state)."""
    diffs = []
    with torch.no_grad():
        for prompt, sv_cont, en_cont in dev_pairs:
            sv_ids, _ = build_concat_ids(tokenizer, prompt, sv_cont, device)
            en_ids, _ = build_concat_ids(tokenizer, prompt, en_cont, device)
            sv_h = model(sv_ids, output_hidden_states=True).hidden_states[layer_idx + 1][:, -1, :]
            en_h = model(en_ids, output_hidden_states=True).hidden_states[layer_idx + 1][:, -1, :]
            diffs.append((sv_h - en_h).squeeze(0))
    vec = torch.stack(diffs).mean(dim=0)
    return vec / (torch.norm(vec) + 1e-8)


def record_condition(
    name: str,
    model,
    tokenizer,
    base_state: Dict[str, torch.Tensor],
    device: str,
    edit_cost: Dict[str, float],
    prefix: str = "",
) -> EvalMetrics:
    dev_win, dev_margin = evaluate_pairs(model, tokenizer, DEV_PAIRS, device, prefix=prefix)
    test_win, test_margin = evaluate_pairs(model, tokenizer, TEST_PAIRS, device, prefix=prefix)
    reg_acc, reg_margin = evaluate_pairs(model, tokenizer, REGRESSION_PAIRS, device, prefix=prefix)
    l2_norm, pct_changed = weight_diff_stats(model, base_state)
    latency_ms = benchmark_inference_latency(model, tokenizer, device, prefix=prefix)
    sample = generate_sample(model, tokenizer, device, prefix=prefix)
    return EvalMetrics(
        condition=name,
        dev_sv_win_rate=round(dev_win, 4),
        dev_mean_margin=round(dev_margin, 4),
        test_sv_win_rate=round(test_win, 4),
        test_mean_margin=round(test_margin, 4),
        regression_accuracy=round(reg_acc, 4),
        regression_mean_margin=round(reg_margin, 4),
        edit_l2_norm=round(l2_norm, 4),
        pct_weights_changed=round(pct_changed, 5),
        edit_wall_time_s=edit_cost["edit_wall_time_s"],
        peak_python_mem_mb=edit_cost["peak_python_mem_mb"],
        inference_ms_per_token=latency_ms,
        sample_output=sample,
    )


def run_all() -> None:
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float32
    ).to(device)
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad_(False)

    commit_hash = getattr(base_model.config, "_commit_hash", "unknown")
    base_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
    zero_cost = {"edit_wall_time_s": 0.0, "peak_python_mem_mb": 0.0}
    results: List[EvalMetrics] = []

    # 1. Unchanged baseline
    results.append(record_condition("1. Unchanged Baseline", base_model, tokenizer, base_state, device, zero_cost))

    # 2. Prompting control (tests whether Swedish capability already exists latently)
    sv_prefix = "Instruktion: Svara alltid på naturlig och korrekt svenska.\n"
    results.append(
        record_condition("2. Prompting Control", base_model, tokenizer, base_state, device, zero_cost, prefix=sv_prefix)
    )

    # 3. Conventional activation steering control (residual-stream addition at 2/3 depth)
    steer_layer = int(len(base_model.model.layers) * 2 / 3)
    steer_alpha = 2.0
    steer_vec, steer_cost = measure_edit_cost(
        compute_steering_vector, base_model, tokenizer, DEV_PAIRS, steer_layer, device
    )

    def steering_hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (output[0] + steer_alpha * steer_vec,) + output[1:]
        return output + steer_alpha * steer_vec

    handle = base_model.model.layers[steer_layer].register_forward_hook(steering_hook)
    try:
        results.append(
            record_condition(
                f"3. Activation Steering (L{steer_layer}, a={steer_alpha})",
                base_model,
                tokenizer,
                base_state,
                device,
                steer_cost,
            )
        )
    finally:
        handle.remove()

    # 4. Direct single-pass weight attribution edit (zero feedback iterations)
    model_direct = copy.deepcopy(base_model)
    _, direct_cost = measure_edit_cost(direct_weight_attribution_edit, model_direct, tokenizer, DEV_PAIRS, device)
    direct_norm, _ = weight_diff_stats(model_direct, base_state)
    results.append(
        record_condition("4. Direct Weight Attribution", model_direct, tokenizer, base_state, device, direct_cost)
    )

    # 5. Feedback-driven evolutionary weight edit (exact same L2 norm budget)
    model_fb = copy.deepcopy(base_model)
    _, fb_cost = measure_edit_cost(
        feedback_evolutionary_edit, model_fb, tokenizer, DEV_PAIRS, device, target_l2_norm=direct_norm
    )
    results.append(
        record_condition("5. Feedback Evolutionary Edit", model_fb, tokenizer, base_state, device, fb_cost)
    )

    # 6. Random untargeted perturbation (exact same L2 norm budget)
    model_rand = copy.deepcopy(base_model)
    _, rand_cost = measure_edit_cost(random_norm_matched_edit, model_rand, target_l2_norm=direct_norm)
    results.append(
        record_condition("6. Random Norm-Matched Control", model_rand, tokenizer, base_state, device, rand_cost)
    )

    payload = {
        "config": {
            "model_id": MODEL_ID,
            "requested_revision": MODEL_REVISION,
            "resolved_commit_hash": commit_hash,
            "seed": SEED,
            "device": device,
            "steer_layer": steer_layer,
            "steer_alpha": steer_alpha,
        },
        "metrics": [asdict(r) for r in results],
    }
    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    lines = [
        f"**Resolved Model Revision:** `{MODEL_ID}` (`{commit_hash}`) | **Seed:** `{SEED}` | **Device:** `{device}`\n",
        "| Condition | Dev Win | Test Win | Test Margin | Reg Acc | Weights Changed | Edit $L_2$ | Edit Time | Latency |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        lines.append(
            f"| {r.condition} | {r.dev_sv_win_rate:.0%} | {r.test_sv_win_rate:.0%} | "
            f"{r.test_mean_margin:+.3f} | {r.regression_accuracy:.0%} | "
            f"`{r.pct_weights_changed:.4f}%` | `{r.edit_l2_norm:.4f}` | "
            f"`{r.edit_wall_time_s:.2f}s` | `{r.inference_ms_per_token:.1f} ms/tok` |"
        )
    lines.append("\n### Qualitative Held-Out Generations (`Varför är det viktigt att dricka vatten varje dag?`)\n")
    for r in results:
        lines.append(f"- **{r.condition}:** `{r.sample_output}`")

    summary_md = "\n".join(lines) + "\n"
    with open("results/summary_table.md", "w", encoding="utf-8") as f:
        f.write(summary_md)

    print(summary_md)


if __name__ == "__main__":
    run_all()