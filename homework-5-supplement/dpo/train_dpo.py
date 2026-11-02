import argparse
import gzip
import json
import random
import re
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


SUBSETS = [
    "harmless-base",
    "helpful-base",
    "helpful-online",
    "helpful-rejection-sampled",
]

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task.\n\n"
    "### Instruction:\n{}\n\n"
    "### Response:\n"
)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_single_turn(text):
    """提取单轮问题和回答，普通段落中的空行不会被切开。"""
    parts = re.split(
        r"\n\n(?=Human:|Assistant:)",
        text.strip(),
    )

    if len(parts) != 2:
        return None

    if not parts[0].startswith("Human:"):
        return None

    if not parts[1].startswith("Assistant:"):
        return None

    question = parts[0][len("Human:"):].strip()
    response = parts[1][len("Assistant:"):].strip()

    return question, response


def load_data(data_path, split, tokenizer, max_seq_len, limit=0):
    """读取、筛选、套模板、编码；暂不padding，也不做标签错位。"""
    items = []
    raw_count = 0
    invalid_count = 0
    long_count = 0

    for subset in SUBSETS:
        path = Path(data_path) / subset / f"{split}.jsonl.gz"
        print("读取：", path, flush=True)

        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                raw_count += 1
                row = json.loads(line)

                chosen = extract_single_turn(row["chosen"])
                rejected = extract_single_turn(row["rejected"])

                if chosen is None or rejected is None:
                    invalid_count += 1
                    continue

                question, chosen_response = chosen
                rejected_question, rejected_response = rejected

                if (
                    question != rejected_question
                    or not question
                    or not chosen_response
                    or not rejected_response
                ):
                    invalid_count += 1
                    continue

                prompt = PROMPT_TEMPLATE.format(question)  

                prompt_ids = tokenizer.encode(
                    prompt,
                    add_special_tokens=False,
                )
                chosen_ids = tokenizer.encode(
                    chosen_response,
                    add_special_tokens=False,
                ) + [tokenizer.eos_token_id]

                rejected_ids = tokenizer.encode(
                    rejected_response,
                    add_special_tokens=False,
                ) + [tokenizer.eos_token_id]

                chosen_full = prompt_ids + chosen_ids
                rejected_full = prompt_ids + rejected_ids

                if max(len(chosen_full), len(rejected_full)) > max_seq_len:
                    long_count += 1
                    continue

                items.append({                              ## 每个item的内容是一个字典 chosen input ids是prompt+回答。没有human和assistant了
                    "chosen_input_ids": chosen_full,
                    "chosen_response_mask": (
                        [0] * len(prompt_ids)
                        + [1] * len(chosen_ids)
                    ),
                    "rejected_input_ids": rejected_full,
                    "rejected_response_mask": (
                        [0] * len(prompt_ids)
                        + [1] * len(rejected_ids)
                    ),
                })

    print("\n数据划分：", split)
    print("原始样本对：", raw_count)
    print("非单轮、问题不一致或空白过滤：", invalid_count)
    print("超长过滤：", long_count)
    print("合格样本对：", len(items))

    if limit > 0:
        items = items[:limit]

    print("本次使用样本对：", len(items), flush=True)
    return items
def get_batch(items, start, batch_size):
    """从已经打乱的数据中，按顺序切出一批。"""
    return items[start:start + batch_size]
def make_batch(items,pad_token_id,device):  # items 是一个micro batch
    batch={}
    for name in ["chosen", "rejected"]:
        max_length = max(
            len(item[f"{name}_input_ids"])
            for item in items
        ) - 1
        input_ids = []
        labels = []
        response_mask = []
        for item in items:
            full_ids=item[f"{name}_input_ids"]
            full_mask=item[f"{name}_response_mask"]
            x = full_ids[:-1]
            y = full_ids[1:]
            mask = full_mask[1:]
            pad_length = max_length - len(x)
            input_ids.append(
                x + [pad_token_id] * pad_length
            )
            labels.append(
                y + [pad_token_id] * pad_length
            )
            response_mask.append(
                mask + [0] * pad_length
            )
        batch[name] = {
            "input_ids": torch.tensor(
                input_ids, dtype=torch.long, device=device
            ),
            "labels": torch.tensor(
                labels, dtype=torch.long, device=device
            ),
            "response_mask": torch.tensor(
                response_mask, dtype=torch.float32, device=device
            ),
        }

    return batch
def get_response_log_probs(model, batch):
    logits = model(
        input_ids=batch["input_ids"],
        use_cache=False,
    ).logits.float()

    target_logits = torch.gather(
        logits,
        dim=-1,
        index=batch["labels"].unsqueeze(-1),
    ).squeeze(-1)  # 取出labels指定的目标token的原始分数。

    token_log_probs = (
        target_logits - torch.logsumexp(logits, dim=-1)
    )  # 得到目标token的对数概率。

    return (
        token_log_probs * batch["response_mask"]
    ).sum(dim=-1)
def compute_dpo_loss(policy, reference, batch, beta):
    """返回每对样本的DPO loss，以及隐式奖励差。"""
    # reference固定，不构建反向传播计算图。
    # 先算reference，避免其临时张量与policy计算图同时驻留。
    with torch.no_grad():
        reference_chosen = get_response_log_probs(
            reference, batch["chosen"]
        )
        reference_rejected = get_response_log_probs(
            reference, batch["rejected"]
        )

    policy_chosen = get_response_log_probs(
        policy, batch["chosen"]
    )
    policy_rejected = get_response_log_probs(
        policy, batch["rejected"]
    )
    chosen_reward = beta * (
        policy_chosen - reference_chosen
    )
    rejected_reward = beta * (
        policy_rejected - reference_rejected
    )
    reward_margin = chosen_reward - rejected_reward
    losses = -F.logsigmoid(reward_margin)

    return losses, reward_margin
def save_checkpoint(policy, tokenizer, path):
    """只保存模型和tokenizer，不保存优化器。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    policy.save_pretrained(path)
    tokenizer.save_pretrained(path)

    print("已保存模型：", path.resolve(), flush=True)
def train(
    policy,
    reference,
    tokenizer,
    items,
    args,
    epochs=1,
    max_steps=0,
    save=False,
):
    policy.train()

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
    )

    saved_checkpoints = []
    step_times = []
    history = []
    step = 0
    total_response_tokens = 0

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    training_start = time.perf_counter()
    for epoch in range(epochs):
        random.shuffle(items)

        for start in range(0, len(items), args.batch_size):
            if max_steps > 0 and step >= max_steps:
                break

            update_items = get_batch(
                items, start, args.batch_size
            )
            optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            step_start = time.perf_counter()

            loss_sum = 0.0
            correct_count = 0
            response_tokens = 0
            for begin in range(len(update_items)):
                batch = make_batch(
                    update_items[begin:begin + 1],
                    tokenizer.pad_token_id,
                    "cuda",
            )
                losses, margins = compute_dpo_loss(
                    policy, reference, batch, args.beta
                )
                loss = losses.sum() / len(update_items)
                loss.backward()

                loss_sum += losses.detach().sum().item()
                correct_count += (
                    margins.detach() > 0
                ).sum().item()

                response_tokens += int(
                    batch["chosen"]["response_mask"].sum().item()
                    + batch["rejected"]["response_mask"].sum().item()
                )

                del loss, losses, margins, batch
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), args.max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            seconds = time.perf_counter() - step_start

            step += 1
            step_times.append(seconds)
            total_response_tokens += response_tokens

            record = {
                "step": step,
                "epoch": epoch + 1,
                "pairs": len(update_items),
                "loss": loss_sum / len(update_items),
                "preference_accuracy": (
                    correct_count / len(update_items)
                ),
                "response_tokens": response_tokens,
                "seconds": seconds,
            }
            history.append(record)

            print(
                f"step={step} "
                f"epoch={epoch + 1} "
                f"loss={record['loss']:.4f} "
                f"pref_acc={record['preference_accuracy']:.2%} "
                f"tokens={response_tokens} "
                f"time={seconds:.2f}s",
                flush=True,
            )

            if save and step % args.save_every == 0:
                path = (
                    Path(args.output_dir)
                    / f"checkpoint-{step:06d}"
                )
                save_checkpoint(policy, tokenizer, path)
                saved_checkpoints.append(path)

                # 只删除本次调用中登记的旧周期checkpoint。
                if len(saved_checkpoints) > 2:
                    old_path = saved_checkpoints.pop(0)
                    shutil.rmtree(old_path)

        if max_steps > 0 and step >= max_steps:
            break
    torch.cuda.synchronize()
    duration = time.perf_counter() - training_start

    # 前两次更新作为预热，不计入稳定更新耗时。
    stable_times = step_times[2:]
    stable_mean = (
        sum(stable_times) / len(stable_times)
        if stable_times else None
    )

    result = {
        "steps": step,
        "training_seconds_including_periodic_saves": duration,
        "mean_update_seconds": sum(step_times) / len(step_times),
        "mean_update_seconds_after_first_two": stable_mean,
        "response_tokens": total_response_tokens,
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated() / 1024**3
        ),
        "peak_reserved_gib": (
            torch.cuda.max_memory_reserved() / 1024**3
        ),
        "history": history,
    }

    print("\n训练统计：")
    for key, value in result.items():
        if key != "history":
            print(key, "：", value, flush=True)

    if save:
        save_checkpoint(
            policy,
            tokenizer,
            Path(args.output_dir) / "final",
        )

    return result
@torch.no_grad()
def evaluate(policy, reference, tokenizer, items, args):
    policy.eval()

    loss_sum = 0.0
    correct_count = 0
    margin_sum = 0.0

    torch.cuda.synchronize()
    start = time.perf_counter()

    for index, item in enumerate(items, start=1):
        batch = make_batch(
            [item], tokenizer.pad_token_id, "cuda"
        )

        losses, margins = compute_dpo_loss(
            policy, reference, batch, args.beta
        )

        loss_sum += losses.sum().item()
        correct_count += (margins > 0).sum().item()
        margin_sum += margins.sum().item()

        if index % 100 == 0:
            print(
                f"已测试：{index}/{len(items)}",
                flush=True,
            )

    torch.cuda.synchronize()
    duration = time.perf_counter() - start

    result = {
        "policy_model": args.model_path,
        "reference_model": args.reference_path,
        "pairs": len(items),
        "beta": args.beta,
        "mean_dpo_loss": loss_sum / len(items),
        "preference_accuracy": correct_count / len(items),
        "mean_reward_margin": margin_sum / len(items),
        "seconds": duration,
    }

    print("\n测试结果：")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["smoke", "train", "eval"],
        default="smoke",
    )
    parser.add_argument(
        "--model_path",
        default="result/sft_full_13129/final",
    )
    parser.add_argument(
        "--reference_path",
        default="result/sft_full_13129/final",
    )
    parser.add_argument(
        "--data_path",
        default="data/hh-rlhf",
    )
    parser.add_argument(
        "--output_dir",
        default="result/dpo_smoke",
    )
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="使用前N对合格数据；0表示全部。smoke模式不使用此参数。",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.reference_path,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    split = "test" if args.mode == "eval" else "train"
    limit = 0 if args.mode == "smoke" else args.limit

    items = load_data(
        args.data_path,
        split,
        tokenizer,
        args.max_seq_len,
        limit,
    )

    print("\n加载policy：", args.model_path, flush=True)
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda")
    policy.config.use_cache = False

    print("加载reference：", args.reference_path, flush=True)
    reference = AutoModelForCausalLM.from_pretrained(
        args.reference_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda")
    reference.config.use_cache = False
    reference.requires_grad_(False)
    reference.eval()

    # DPO计算固定文本概率，关闭dropout造成的随机扰动。
    for model in [policy, reference]:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0

    save_json(output_dir / "config.json", vars(args))

    if args.mode == "eval":
        result = evaluate(
            policy, reference, tokenizer, items, args
        )
        path = output_dir / f"eval_results_{time.time_ns()}.json"
        save_json(path, result)
        print("结果文件：", path.resolve())
        return

    policy.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        }
    )

    if args.mode == "smoke":
        # 普通部分：随机选1000对，完成20次更新。
        ordinary_items = random.sample(
            items, min(1000, len(items))
        )

        print("\n========== 普通样本试跑 ==========", flush=True)
        ordinary_result = train(
            policy,
            reference,
            tokenizer,
            ordinary_items,
            args,
            epochs=1,
            max_steps=20,
        )
        save_json(
            output_dir / "smoke_ordinary.json",
            ordinary_result,
        )

        # 压力部分：取最长的16对，遍历两次，共两次更新。
        # 此部分沿用试跑policy，但新建优化器；不作为正式训练起点。
        longest_items = sorted(
            items,
            key=lambda item: max(
                len(item["chosen_input_ids"]),
                len(item["rejected_input_ids"]),
            ),
            reverse=True,
        )[:16]

        torch.cuda.empty_cache()

        print("\n========== 长样本压力试跑 ==========", flush=True)
        long_result = train(
            policy,
            reference,
            tokenizer,
            longest_items,
            args,
            epochs=2,
            max_steps=2,
        )
        save_json(
            output_dir / "smoke_long.json",
            long_result,
        )

        print("\n试跑完成，没有保存试跑模型。")
        print("统计目录：", output_dir.resolve())

    else:
        result = train(
            policy,
            reference,
            tokenizer,
            items,
            args,
            epochs=1,
            save=True,
        )
        save_json(output_dir / "train_results.json", result)
        print("\n正式训练完成，未自动运行test。")


if __name__ == "__main__":
    main()

