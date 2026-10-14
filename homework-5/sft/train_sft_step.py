import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict
import multiprocessing as mp

import torch
from torch.optim import AdamW
from transformers import (
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
)


PROJECT_DIR = Path("/home/hong/cyan/homework-5")

args = SimpleNamespace(
    model_id=str(PROJECT_DIR / "model/Qwen2.5-Math-1.5B"),
    train_data_path=str(PROJECT_DIR / "data/gsm8k/train_r1_style.jsonl"),
    output_dir=str(PROJECT_DIR / "result/checkpoints"),
    test_data_path=str(PROJECT_DIR / "data/gsm8k/test_r1_style.jsonl"),

    device="cuda:0",
    lr=2e-5,
    max_steps=200,
    batch_size=16,
    micro_batch_size=1,
    seed=42,
)


def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, torch.Tensor]:
    if not prompt_strs or len(prompt_strs) != len(output_strs):
        raise ValueError("数据不能为空，且 prompt 和 response 数量必须一致")

    all_input_ids = []
    all_mask = []
    all_length = []

    for p_str, o_str in zip(prompt_strs, output_strs):
        p_ids = tokenizer.encode(p_str, add_special_tokens=False)
        o_ids = tokenizer.encode(o_str, add_special_tokens=False)

        input_ids = p_ids + o_ids
        all_input_ids.append(input_ids)
        all_length.append(len(input_ids))

        mask = [0] * len(p_ids) + [1] * len(o_ids)
        all_mask.append(mask)

    max_len = max(all_length)
    batch_size = len(prompt_strs)
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_id is None:
        raise ValueError("分词器没有 pad_token_id 或 eos_token_id")

    padded_input_ids = torch.full(
        (batch_size, max_len), pad_id, dtype=torch.long
    )
    padded_mask = torch.zeros(
        (batch_size, max_len), dtype=torch.long
    )

    for i, (ids, m) in enumerate(zip(all_input_ids, all_mask)):
        length = all_length[i]
        padded_input_ids[i, :length] = torch.tensor(ids)
        padded_mask[i, :length] = torch.tensor(m)

    return {
        "input_ids": padded_input_ids[:, :-1].clone(),
        "labels": padded_input_ids[:, 1:].clone(),
        "response_mask": padded_mask[:, 1:].clone(),
    }


def get_batch(train_data, batch_size, device):
    train_data_len = len(train_data["input_ids"])
    batch_indices = random.sample(range(train_data_len), batch_size)

    return {
        "input_ids": train_data["input_ids"][batch_indices].to(device),
        "labels": train_data["labels"][batch_indices].to(device),
        "response_mask": train_data["response_mask"][batch_indices].to(device),
    }


def save_checkpoint(policy, tokenizer, optimizer, step, args, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 模型和分词器：用于评估、加载模型
    policy.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # 训练状态：为后续完整断点续训保留
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "args": vars(args).copy(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        },
        save_dir / "training_state.pt",
    )

    print(f"\n模型与训练状态已保存：{save_dir}", flush=True)


def run_sft(args, save_dir):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.max_steps <= 0:
        raise ValueError("max_steps 必须大于 0")
    if args.micro_batch_size <= 0 or args.batch_size <= 0:
        raise ValueError("batch size 必须大于 0")
    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError("batch_size 必须能被 micro_batch_size 整除")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(args.device)

    policy.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    policy.config.use_cache = False

    optimizer = AdamW(policy.parameters(), lr=args.lr)

    with open(args.train_data_path, "r", encoding="utf-8") as f:
        raw_train_data = [
            json.loads(line) for line in f if line.strip()
        ]

    if len(raw_train_data) < args.micro_batch_size:
        raise ValueError("训练样本数少于 micro_batch_size")

    tokenized_train_data = tokenize_prompt_and_output(
        prompt_strs=[item["prompt"] for item in raw_train_data],
        output_strs=[item["response"] for item in raw_train_data],
        tokenizer=tokenizer,
    )

    policy.train()
    loss_function = torch.nn.CrossEntropyLoss(reduction="none")
    grad_accum_steps = args.batch_size // args.micro_batch_size
    optimizer.zero_grad(set_to_none=True)

    for step in range(args.max_steps):
        loss_sum = 0.0

        for _ in range(grad_accum_steps):
            batch = get_batch(
                tokenized_train_data,
                args.micro_batch_size,
                args.device,
            )
            labels = batch["labels"]
            response_mask = batch["response_mask"]

            logit = policy(batch["input_ids"]).logits

            token_loss = loss_function(
                logit.float().reshape(-1, logit.shape[-1]),
                labels.reshape(-1),
            ).reshape_as(labels)

            masked_token_loss = token_loss * response_mask

            # 保持你原来的归一化方式：
            # 回答 token 损失求和，再按样本数平均
            loss = masked_token_loss.sum() / labels.shape[0]
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            loss_sum += loss.item()

            del logit, token_loss, masked_token_loss, loss, scaled_loss

        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        print(
            f"Step {step + 1}/{args.max_steps}: "
            f"loss={loss_sum / grad_accum_steps:.4f}",
            flush=True,
        )

    save_checkpoint(
        policy, tokenizer, optimizer,
        args.max_steps, args, save_dir,
    )


def run_test(args, save_dir):
    save_dir = Path(save_dir)
    output_file = save_dir / "test_results.jsonl"

    # 使用同一个 Conda 环境中的 Python，调用已有评估脚本
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_DIR / "evaluation/evaluate_zero_shot.py"),
            "--model_path", str(save_dir),
            "--data_path", args.test_data_path,
            "--output_file", str(output_file),
            "--limit", "0",  # 测试全部题目
        ],
        check=True,
        cwd=str(PROJECT_DIR),
    )

    # 现有评估脚本已经保存逐题结果。
    # 在此额外保存总体统计。
    stats = {
        "total": 0,
        "correct": 0,
        "format_error": 0,
        "correct_format_wrong_answer": 0,
    }

    with output_file.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            scores = json.loads(line)["scores"]

            stats["total"] += 1
            if scores.get("reward", 0) == 1:
                stats["correct"] += 1

            if scores.get("format_reward", 0) == 0:
                stats["format_error"] += 1
            elif scores.get("answer_reward", 0) == 0:
                stats["correct_format_wrong_answer"] += 1

    stats["accuracy"] = (
        stats["correct"] / stats["total"] if stats["total"] else 0.0
    )

    summary_file = save_dir / "test_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n逐题结果：{output_file}")
    print(f"统计结果：{summary_file}")


if __name__ == "__main__":
    # 先检查文件，避免训练完才发现评估文件缺失
    for path in [
        args.train_data_path,
        args.test_data_path,
        PROJECT_DIR / "evaluation/evaluate_zero_shot.py",
    ]:
        if not Path(path).is_file():
            raise FileNotFoundError(f"找不到文件：{path}")

    # 每次运行生成一个新目录，避免覆盖以前的结果
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_dir = Path(args.output_dir) / f"{run_name}_step_{args.max_steps}"

    # 训练在独立进程中执行。
    # 进程结束后，训练占用的 GPU 显存被释放，再启动评估。
    context = mp.get_context("spawn")
    training_process = context.Process(
        target=run_sft,
        args=(args, str(save_dir)),
    )
    training_process.start()
    training_process.join()

    if training_process.exitcode != 0:
        raise RuntimeError("训练失败，未启动测试；请查看上方错误")

    run_test(args, save_dir)