import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd


SYSTEM_PROMPT = (
    "Below is a list of conversations between a human and an AI assistant (you).\n"
    'Users place their queries under "### Instruction:", and your responses are under "### Response:".\n'
    "You are a helpful, respectful, and honest assistant.\n"
    "You should always answer as helpfully as possible while ensuring safety.\n"
    "Your answers should be well-structured and provide detailed information. "
    "They should also have an engaging tone.\n"
    "Your responses must not contain any fake, harmful, unethical, racist, sexist, "
    "toxic, dangerous, or illegal content, even if it may be helpful.\n"
    "Your response must be socially responsible, and thus you can reject to answer "
    "some controversial topics."
)

PROMPT_TEMPLATE = (
    "{system_prompt}\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)

MMLU_TASK_TEMPLATE = (
    "Answer the following multiple choice question about {subject}. Respond with a single "
    'sentence of the form "The correct answer is _", filling the blank with the letter '
    "corresponding to the correct answer (i.e., A, B, C or D).\n\n"
    "Question: {question}\n"
    "A. {optA}\n"
    "B. {optB}\n"
    "C. {optC}\n"
    "D. {optD}"
)

MAX_NEW_TOKENS = 64
STOP_SEQUENCES = ["### Instruction:"]


def build_prompt(instruction):
    return PROMPT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        instruction=instruction,
    )


def load_mmlu_data(data_dir):
    """读取六列 CSV，构造提示词；标准答案只用于评分。"""
    data_dir = Path(data_dir)

    if not data_dir.is_dir():
        raise NotADirectoryError(f"找不到数据目录：{data_dir}")

    csv_files = sorted(data_dir.glob("*_test.csv"))
    if not csv_files:
        raise FileNotFoundError(f"{data_dir} 中没有 *_test.csv 文件")

    items = []

    for csv_file in csv_files:
        subject = (
            csv_file.name
            .removesuffix("_test.csv")
            .replace("_", " ")
        )

        df = pd.read_csv(csv_file, header=None)

        if df.shape[1] != 6:
            raise ValueError(
                f"{csv_file} 应有六列，实际为 {df.shape[1]} 列"
            )

        df.columns = ["question", "A", "B", "C", "D", "answer"]

        for _, row in df.iterrows():
            instruction = MMLU_TASK_TEMPLATE.format(
                subject=subject,
                question=row["question"],
                optA=row["A"],
                optB=row["B"],
                optC=row["C"],
                optD=row["D"],
            )

            gold = str(row["answer"]).strip().upper()
            if gold not in {"A", "B", "C", "D"}:
                raise ValueError(
                    f"{csv_file} 中存在非法标准答案：{gold!r}"
                )

            items.append({
                "subject": subject,
                "full_prompt": build_prompt(instruction),
                "gold": gold,
            })

    if not items:
        raise ValueError("没有读到可评测的题目")

    return items


def parse_mmlu_response(text):
    """保持原来的选项提取规则。"""
    if not text:
        return None

    match = re.search(
        r"[Tt]he correct answer is\s*([A-D])",
        text,
    )
    if match:
        return match.group(1).upper()

    match = re.search(r"\b([A-D])\b", text)
    if match:
        return match.group(1).upper()

    return None


def evaluate(llm, items, sampling_params):
    prompts = [item["full_prompt"] for item in items]

    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    duration = time.perf_counter() - start_time

    if len(outputs) != len(items):
        raise RuntimeError("模型返回数量与题目数量不一致")

    details = []

    for item, request in zip(items, outputs):
        if len(request.outputs) != 1:
            raise RuntimeError("每道题应恰好生成一个回答")

        answer = request.outputs[0]
        prediction = parse_mmlu_response(answer.text)

        details.append({
            "subject": item["subject"],
            "gold": item["gold"],
            "pred": prediction,
            "output": answer.text,
            "is_correct": prediction == item["gold"],
            "finish_reason": answer.finish_reason,
            "stop_reason": answer.stop_reason,
            "generated_tokens": len(answer.token_ids),
        })

    total = len(details)
    correct = sum(row["is_correct"] for row in details)

    metrics = {
        "accuracy": correct / total,
        "throughput": total / duration if duration > 0 else 0.0,
        "total": total,
        "success": total,
        "correct_count": correct,
        "parsing_failure_count": sum(
            row["pred"] is None for row in details
        ),
        "empty_count": sum(
            not row["output"].strip() for row in details
        ),
        "length_limited_count": sum(
            row["finish_reason"] == "length" for row in details
        ),
        "generation_time_seconds": duration,
    }

    return metrics, details


def print_report(metrics, details):
    subject_stats = {}

    for row in details:
        subject = row["subject"]
        if subject not in subject_stats:
            subject_stats[subject] = {"correct": 0, "total": 0}

        subject_stats[subject]["total"] += 1
        subject_stats[subject]["correct"] += int(row["is_correct"])

    print()
    print(f"{'Subject':40s} | {'Accuracy':>10s} | Count")
    print("-" * 70)

    for subject in sorted(subject_stats):
        stats = subject_stats[subject]
        accuracy = stats["correct"] / stats["total"]
        print(
            f"{subject:40s} | {accuracy:10.2%} | {stats['total']}"
        )

    print("-" * 70)
    print(f"总题数：{metrics['total']}")
    print(f"答对题数：{metrics['correct_count']}")
    print(f"总体准确率：{metrics['accuracy']:.2%}")
    print(f"解析失败数：{metrics['parsing_failure_count']}")
    print(f"空回答数：{metrics['empty_count']}")
    print(f"达到长度上限的回答数：{metrics['length_limited_count']}")
    print(f"生成吞吐量：{metrics['throughput']:.2f} 题/秒")


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用统一提示词格式评测 MMLU"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument(
        "--limit", type=int, default=10,
        help="默认前 10 题，0 表示全部",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.7
    )
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--max_num_seqs", type=int, default=4)
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16"],
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit 不能小于 0")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor_parallel_size 至少为 1")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu_memory_utilization 必须在 0 和 1 之间")
    if args.max_model_len <= MAX_NEW_TOKENS:
        parser.error("--max_model_len 必须大于 64")
    if args.max_num_seqs < 1:
        parser.error("--max_num_seqs 至少为 1")

    return args


def main():
    args = parse_args()

    output_path = Path(args.output_file)
    if output_path.exists():
        raise FileExistsError(f"文件已存在，请使用新文件名：{output_path}")

    items = load_mmlu_data(args.data_path)
    print(f"测试集共 {len(items)} 道题", flush=True)

    if args.limit:
        items = items[:args.limit]

    print(f"本次评测 {len(items)} 道题", flush=True)
    print("\n第一题实际提示词：\n" + items[0]["full_prompt"], flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_path,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=True,
        seed=args.seed,
    )

    tokenizer = llm.get_tokenizer()

    for index, item in enumerate(items, start=1):
        input_length = len(tokenizer.encode(item["full_prompt"]))
        if input_length + MAX_NEW_TOKENS > args.max_model_len:
            raise ValueError(
                f"第 {index} 题输入为 {input_length} token，"
                f"加上 {MAX_NEW_TOKENS} 个输出 token 后超过 "
                f"max_model_len={args.max_model_len}。"
                "请提高 --max_model_len 后重新运行。"
            )

    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=MAX_NEW_TOKENS,
        stop=STOP_SEQUENCES,
        ignore_eos=False,
        include_stop_str_in_output=False,
    )

    metrics, details = evaluate(llm, items, sampling_params)

    result = {
        "config": {
            **vars(args),
            "protocol": "instruction_response_v1",
            "system_prompt": SYSTEM_PROMPT,
            "prompt_template": PROMPT_TEMPLATE,
            "task_template": MMLU_TASK_TEMPLATE,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": MAX_NEW_TOKENS,
            "stop_sequences": STOP_SEQUENCES,
            "ignore_eos": False,
        },
        "metrics": metrics,
        "details": details,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print_report(metrics, details)
    print(f"结果已保存：{output_path.resolve()}")


if __name__ == "__main__":
    main()