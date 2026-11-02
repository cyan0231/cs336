import argparse
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
## 主线流程 先是load_data 将预处理后的json文件（一条json 包含prompt部分，以及response）变成input_ids（prompt+response+eos），mask，valid_token_num
## 主函数 训练流程：进入第一个循环，由最大step决定，在这个循环中，先通过get_batch得到一批随机的items（就是从load_data返回的总items中随机数抽一批）
## 接着进入第二个循环，进行梯度累计，每次从batch中取出来一个micro batch 通过make_batch 返回x，y以及mask。然后让x进行前向传播，计算损失，并累积。
## 直至累计完一个大的batch 才更新

def parse_args():
    parser = argparse.ArgumentParser(description="全参数 SFT 与测试集 loss")

    # 新增：三种运行模式。
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "train_eval"],
        default="train",
    )

    parser.add_argument(
        "--model_path",
        default="model/Qwen2.5-1.5B",
    )
    parser.add_argument(
        "--data_path",
        default="data/sft_processed/sample_train.jsonl",
    )
    parser.add_argument(
        "--eval_data_path",
        default="data/sft_processed/test.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default="result/sft_smoke_20",
    )

    # 新增：checkpoint 和测试参数。
    parser.add_argument("--resume_from", default=None)
    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
        help="每多少次更新保存一次；0 表示只在结束时保存",
    )
    parser.add_argument(
        "--eval_limit",
        type=int,
        default=10,
        help="测试前 N 条合格样本；0 表示全部",
    )
    parser.add_argument("--eval_batch_size", type=int, default=1)

    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
    )

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    for name in (
        "max_steps",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "eval_batch_size",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name} 至少为 1")

    if args.max_seq_len < 2:
        parser.error("--max_seq_len 至少为 2")
    if args.learning_rate <= 0:
        parser.error("--learning_rate 必须大于 0")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("--warmup_ratio 必须在 [0, 1) 内")
    if args.max_grad_norm <= 0:
        parser.error("--max_grad_norm 必须大于 0")
    if args.save_every < 0:
        parser.error("--save_every 不能小于 0")
    if args.eval_limit < 0:
        parser.error("--eval_limit 不能小于 0")

    if args.mode == "eval" and args.resume_from:
        parser.error(
            "只测试时，用 --model_path 指向 checkpoint，"
            "不使用 --resume_from"
        )

    return args


def save_json(path, data):
    """先写临时文件，再替换正式文件。"""
    path = Path(path)
    temporary_path = path.with_name(path.name + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    temporary_path.replace(path)


def file_sha256(path):
    """检查恢复训练时，数据文件内容是否改变。"""
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def load_data(path, tokenizer, max_seq_len, limit=0):
    """
    读取预处理后的 prompt、response，并提前编码。

    不再次套模板，不做全数据 padding。

    limit=0：读取全部合格样本。
    limit=N：读取文件顺序中的前 N 条合格样本。
    """
    items = []
    total = 0
    filtered = 0

    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer 没有 eos_token_id")

    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            row = json.loads(line)
            prompt = row["prompt"]
            response = row["response"]

            if not isinstance(prompt, str) or not isinstance(response, str):
                raise ValueError(
                    f"第 {line_number} 行的 prompt 和 response 必须为字符串"
                )

            if not prompt.strip() or not response.strip():
                raise ValueError(
                    f"第 {line_number} 行存在空白内容，请检查预处理结果"
                )

            total += 1

            prompt_ids = tokenizer.encode(
                prompt,
                add_special_tokens=False,
            )
            response_ids = tokenizer.encode(
                response,
                add_special_tokens=False,
            )

            if not prompt_ids or not response_ids:
                raise ValueError(
                    f"第 {line_number} 行编码后的 prompt 或 response 为空"
                )

            target_ids = response_ids + [tokenizer.eos_token_id]
            input_ids = prompt_ids + target_ids

            if len(input_ids) > max_seq_len:
                filtered += 1
                continue

            response_mask = (
                [0] * len(prompt_ids)
                + [1] * len(target_ids)
            )

            items.append({
                "input_ids": input_ids,
                "response_mask": response_mask,
                "target_tokens": len(target_ids),
            })

            if limit > 0 and len(items) >= limit:
                break

    if not items:
        raise ValueError(f"过滤后没有可用样本：{path}")

    print(f"\n数据文件：{path}")
    print(f"本次扫描样本数：{total}")
    print(f"扫描范围内超长过滤数：{filtered}")
    print(f"实际加载样本数：{len(items)}")
    print(
        "加载样本最大总长度：",
        max(len(item["input_ids"]) for item in items),
        flush=True,
    )

    return items


def get_batch(items, batch_size):
    """
    随机抽取 batch_size 条样本。

    同一次抽样内不重复，不同更新之间允许重复。
    """
    if batch_size > len(items):
        raise ValueError(
            f"batch_size={batch_size} 超过样本数 {len(items)}"
        )

    return random.sample(items, batch_size)


def make_batch(items, pad_token_id, device):
    """
    接收一个小批次，动态右侧 padding，并完成标签错位。

    原序列：        [P1, P2, R1, R2, EOS]
    input_ids：    [P1, P2, R1, R2]
    labels：       [P2, R1, R2, EOS]
    response_mask：[ 0,  1,  1,   1]

    response_mask 对齐 labels。
    """
    batch_size = len(items)
    max_length = max(len(item["input_ids"]) for item in items)

    full_input_ids = torch.full(
        (batch_size, max_length),
        fill_value=pad_token_id,
        dtype=torch.long,
    )

    full_response_mask = torch.zeros(
        (batch_size, max_length),
        dtype=torch.float32,
    )

    for index, item in enumerate(items):
        length = len(item["input_ids"])

        full_input_ids[index, :length] = torch.tensor(
            item["input_ids"],
            dtype=torch.long,
        )

        full_response_mask[index, :length] = torch.tensor(
            item["response_mask"],
            dtype=torch.float32,
        )

    input_ids = full_input_ids[:, :-1].contiguous()
    labels = full_input_ids[:, 1:].contiguous()
    response_mask = full_response_mask[:, 1:].contiguous()

    return {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "response_mask": response_mask.to(device),
    }


def get_learning_rate(step, max_steps, peak_lr, warmup_steps):
    """step 从 0 开始：先 warmup，再 cosine 衰减。"""
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps

    decay_steps = max(1, max_steps - warmup_steps)
    progress = (step - warmup_steps) / decay_steps

    return peak_lr * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


@torch.no_grad()
def evaluate(model, tokenizer, args, device, model_source):
    """
    新增：测试集 loss。

    不更新参数，不自由生成回答。
    保留先计算逐 token 损失，再乘 mask 的方式。
    """
    eval_items = load_data(
        args.eval_data_path,
        tokenizer,
        args.max_seq_len,
        limit=args.eval_limit,
    )

    was_training = model.training
    model.eval()

    total_loss_sum = 0.0
    total_target_tokens = 0

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    try:
        for start in range(0, len(eval_items), args.eval_batch_size):
            micro_items = eval_items[
                start:start + args.eval_batch_size
            ]

            batch = make_batch(
                micro_items,
                tokenizer.pad_token_id,
                device,
            )

            outputs = model(
                input_ids=batch["input_ids"],
                use_cache=False,
            )

            logits = outputs.logits
            labels = batch["labels"]
            response_mask = batch["response_mask"]

            # 与训练保持一致：先计算全部位置的损失。
            token_losses = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            ).reshape_as(labels)

            # 再用 mask 屏蔽 prompt 和 padding。
            loss_sum = (
                token_losses * response_mask
            ).sum()

            total_loss_sum += loss_sum.item()
            total_target_tokens += sum(
                item["target_tokens"] for item in micro_items
            )

            del (
                outputs,
                logits,
                labels,
                response_mask,
                token_losses,
                loss_sum,
                batch,
            )

        torch.cuda.synchronize()
        duration = time.perf_counter() - start_time

    finally:
        model.train(was_training)

    result = {
        "model_path": str(model_source),
        "eval_data_path": str(Path(args.eval_data_path).resolve()),
        "eval_limit": args.eval_limit,
        "eval_batch_size": args.eval_batch_size,
        "max_seq_len": args.max_seq_len,
        "samples": len(eval_items),
        "target_tokens": total_target_tokens,
        "mean_token_loss": total_loss_sum / total_target_tokens,
        "evaluation_seconds": duration,
    }

    # 不同测试分别保存，避免冒烟测试覆盖全量结果。
    result_path = (
        Path(args.output_dir)
        / f"eval_results_{time.time_ns()}.json"
    )
    save_json(result_path, result)

    print("\n测试完成")
    print(f"样本数：{result['samples']}")
    print(f"有效回答 token 数：{total_target_tokens}")
    print(f"平均 token loss：{result['mean_token_loss']:.6f}")
    print(f"测试耗时：{duration:.2f} 秒")
    print(f"结果文件：{result_path.resolve()}", flush=True)

    return result


def save_checkpoint(
    model,
    tokenizer,
    optimizer,
    step,
    training_config,
    output_dir,
    final=False,
):
    """
    新增：保存完整 checkpoint。

    周期 checkpoint 只保留最新两个。
    final 单独保存，不参与周期清理。
    """
    output_dir = Path(output_dir)
    name = "final" if final else f"checkpoint-{step:06d}"

    checkpoint_dir = output_dir / name
    temporary_dir = output_dir / f".{name}.tmp"

    if checkpoint_dir.exists() or temporary_dir.exists():
        raise FileExistsError(
            f"保存目录或临时目录已存在：{checkpoint_dir}。"
            "请先检查，不自动覆盖。"
        )

    temporary_dir.mkdir()

    # 保存后的模型可以用于推理，当前训练仍保持关闭 KV cache。
    old_use_cache = model.config.use_cache

    try:
        model.config.use_cache = True
        model.save_pretrained(
            temporary_dir,
            safe_serialization=True,
        )
    finally:
        model.config.use_cache = old_use_cache

    tokenizer.save_pretrained(temporary_dir)

    state = {
        "step": step,
        "training_config": training_config,
        "optimizer": optimizer.state_dict(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }

    torch.save(
        state,
        temporary_dir / "training_state.pt",
    )
    save_json(
        temporary_dir / "training_config.json",
        training_config,
    )

    # 写完后再改成正式目录名。
    temporary_dir.rename(checkpoint_dir)

    print(f"\n已保存 checkpoint：{checkpoint_dir}", flush=True)

    if not final:
        # 清单只登记本脚本创建的周期 checkpoint。
        manifest_path = output_dir / "checkpoints.json"

        if manifest_path.exists():
            with manifest_path.open(encoding="utf-8") as f:
                managed = json.load(f)
        else:
            managed = []

        if name not in managed:
            managed.append(name)

        save_json(manifest_path, managed)

        while len(managed) > 2:
            old_name = managed[0]

            if (
                not old_name.startswith("checkpoint-")
                or not old_name.removeprefix("checkpoint-").isdigit()
            ):
                raise ValueError("checkpoint 清单中的目录名不合法")

            old_dir = output_dir / old_name

            if old_dir.is_symlink():
                raise ValueError("不自动清理符号链接目录")

            if old_dir.exists():
                shutil.rmtree(old_dir)
                print(f"已清理旧 checkpoint：{old_dir}", flush=True)

            managed.pop(0)
            save_json(manifest_path, managed)

    return checkpoint_dir


def load_checkpoint(checkpoint_path, optimizer, expected_config):
    """
    新增：恢复优化器、已完成步数和随机状态。

    模型权重由 main 中的 from_pretrained 加载。
    """
    state_path = Path(checkpoint_path) / "training_state.pt"

    # 只加载自己保存的可信 checkpoint。
    state = torch.load(
        state_path,
        map_location="cpu",
        weights_only=False,
    )

    saved_config = state["training_config"]

    differences = [
        key
        for key, value in expected_config.items()
        if saved_config.get(key) != value
    ]

    if differences:
        raise ValueError(
            "恢复训练的关键配置不一致："
            + ", ".join(differences)
        )

    optimizer.load_state_dict(state["optimizer"])

    # 放在模型、优化器初始化之后，避免初始化消耗随机状态。
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])

    completed_steps = int(state["step"])

    print(
        f"已恢复训练状态，完成更新数：{completed_steps}",
        flush=True,
    )

    return completed_steps


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到 CUDA GPU")

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU 环境不支持 BF16")

    device = torch.device("cuda:0")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    is_training = args.mode in ("train", "train_eval")
    output_dir = Path(args.output_dir).resolve()

    # 新训练不覆盖旧目录；恢复训练使用原目录。
    if is_training and not args.resume_from and output_dir.exists():
        raise FileExistsError(
            f"输出目录已存在，请使用新目录：{output_dir}"
        )

    if args.resume_from:
        resume_path = Path(args.resume_from).resolve()

        if not (resume_path / "training_state.pt").is_file():
            raise FileNotFoundError(
                f"找不到完整训练状态：{resume_path}"
            )

        if resume_path.parent != output_dir:
            raise ValueError(
                "--output_dir 必须是恢复 checkpoint 的父目录"
            )

        # 同一训练目录继续时，从最新的已完成保存点恢复，
        # 避免覆盖后续已经存在的 checkpoint。
        final_path = output_dir / "final"
        manifest_path = output_dir / "checkpoints.json"

        if final_path.exists():
            latest_path = final_path.resolve()
        elif manifest_path.exists():
            with manifest_path.open(encoding="utf-8") as f:
                names = json.load(f)

            if not names:
                raise ValueError("checkpoint 清单为空")

            latest_path = (output_dir / names[-1]).resolve()
        else:
            latest_path = resume_path

        if resume_path != latest_path:
            raise ValueError(
                f"请从该目录最新的 checkpoint 恢复：{latest_path}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    # 恢复时加载 checkpoint 权重，否则加载 model_path。
    model_source = args.resume_from or args.model_path

    tokenizer = AutoTokenizer.from_pretrained(model_source)

    if tokenizer.eos_token_id is None:
        raise ValueError("模型 tokenizer 没有 EOS token")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_training:
        items = load_data(
            args.data_path,
            tokenizer,
            args.max_seq_len,
        )

        effective_batch_size = (
            args.micro_batch_size
            * args.gradient_accumulation_steps
        )

        if effective_batch_size > len(items):
            raise ValueError(
                f"有效 batch size={effective_batch_size}，"
                f"但只有 {len(items)} 条合格样本"
            )

        # 恢复时核对这些会影响训练过程的配置。
        training_config = {
            "base_model_path": str(Path(args.model_path).resolve()),
            "data_sha256": file_sha256(args.data_path),
            "max_steps": args.max_steps,
            "max_seq_len": args.max_seq_len,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": (
                args.gradient_accumulation_steps
            ),
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "max_grad_norm": args.max_grad_norm,
            "seed": args.seed,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "loss_method": "cross_entropy_then_mask",
        }

    print(f"\n正在加载模型：{model_source}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)

    model.config.use_cache = False

    if is_training:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
            }
        )
        model.train()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            foreach=False,
        )

        start_step = 0

        if args.resume_from:
            start_step = load_checkpoint(
                args.resume_from,
                optimizer,
                training_config,
            )

        if not 0 <= start_step <= args.max_steps:
            raise ValueError("checkpoint 中的更新步数不合法")

        warmup_steps = math.ceil(
            args.max_steps * args.warmup_ratio
        )

        print(f"设备：{torch.cuda.get_device_name(0)}")
        print(f"训练样本数：{len(items)}")
        print(f"已完成更新次数：{start_step}")
        print(f"计划总更新次数：{args.max_steps}")
        print(f"micro-batch size：{args.micro_batch_size}")
        print(f"梯度累积次数：{args.gradient_accumulation_steps}")
        print(f"有效 batch size：{effective_batch_size}")
        print(f"warmup 步数：{warmup_steps}")
        print(f"峰值学习率：{args.learning_rate}", flush=True)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        training_start = time.perf_counter()
        step_times = []

        # 训练主体保留原来的计算方式。
        for step in range(start_step, args.max_steps):
            torch.cuda.synchronize()
            step_start = time.perf_counter()

            learning_rate = get_learning_rate(
                step=step,
                max_steps=args.max_steps,
                peak_lr=args.learning_rate,
                warmup_steps=warmup_steps,
            )

            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            update_items = get_batch(
                items,
                effective_batch_size,
            )

            total_target_tokens = sum(
                item["target_tokens"]
                for item in update_items
            )

            optimizer.zero_grad(set_to_none=True)

            update_loss_sum = torch.zeros(
                (),
                device=device,
                dtype=torch.float32,
            )

            for start in range(
                0,
                effective_batch_size,
                args.micro_batch_size,
            ):
                micro_items = update_items[
                    start:start + args.micro_batch_size
                ]

                batch = make_batch(
                    micro_items,
                    tokenizer.pad_token_id,
                    device,
                )

                # 右侧 padding 不会被前面的有效位置看到。
                outputs = model(
                    input_ids=batch["input_ids"],
                    use_cache=False,
                )

                # make_batch 已经完成错位，不再切片。
                logits = outputs.logits
                labels = batch["labels"]
                response_mask = batch["response_mask"]

                # 保留：先计算每个位置的交叉熵。
                token_losses = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    reduction="none",
                ).reshape_as(labels)

                # 保留：再用 mask 屏蔽 prompt 和 padding。
                loss_sum = (
                    token_losses * response_mask
                ).sum()

                # 一次更新的全部有效回答 token 共同归一化。
                loss = loss_sum / total_target_tokens
                loss.backward()

                update_loss_sum += loss_sum.detach()

                del (
                    outputs,
                    logits,
                    labels,
                    response_mask,
                    token_losses,
                    loss_sum,
                    loss,
                    batch,
                )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.max_grad_norm,
            )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            duration = time.perf_counter() - step_start
            step_times.append(duration)

            mean_loss = (
                update_loss_sum.item() / total_target_tokens
            )
            peak_allocated = (
                torch.cuda.max_memory_allocated() / 1024**3
            )
            completed_step = step + 1

            print(
                f"step {completed_step:>4}/{args.max_steps} | "
                f"loss={mean_loss:.4f} | "
                f"lr={learning_rate:.3e} | "
                f"answer_tokens={total_target_tokens} | "
                f"time={duration:.2f}s | "
                f"peak_allocated={peak_allocated:.2f}GiB",
                flush=True,
            )

            # 新增：完成更新后，按间隔保存。
            if (
                args.save_every > 0
                and completed_step % args.save_every == 0
            ):
                save_checkpoint(
                    model,
                    tokenizer,
                    optimizer,
                    completed_step,
                    training_config,
                    output_dir,
                )

        torch.cuda.synchronize()
        training_duration = time.perf_counter() - training_start

        print("\n训练完成")
        print(
            f"本次训练循环耗时（含周期保存）："
            f"{training_duration:.2f} 秒"
        )

        stable_times = step_times[2:]
        if stable_times:
            print(
                "排除本次前两步后的平均更新耗时（不含保存）："
                f"{sum(stable_times) / len(stable_times):.2f} 秒"
            )

        print(
            "峰值已分配显存："
            f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB"
        )
        print(
            "峰值保留显存："
            f"{torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB"
        )

        final_path = output_dir / "final"

        # 从已经完成训练的 final 恢复时，不重复保存。
        resumed_final = (
            args.resume_from is not None
            and Path(args.resume_from).resolve() == final_path
            and start_step == args.max_steps
        )

        if not resumed_final:
            save_checkpoint(
                model,
                tokenizer,
                optimizer,
                args.max_steps,
                training_config,
                output_dir,
                final=True,
            )

        model_source = final_path

        # 测试前释放优化器状态与训练数据。
        del optimizer, items
        torch.cuda.empty_cache()

    # 新增：只测试，或训练完成后测试。
    if args.mode in ("eval", "train_eval"):
        evaluate(
            model,
            tokenizer,
            args,
            device,
            model_source,
        )


if __name__ == "__main__":
    main()