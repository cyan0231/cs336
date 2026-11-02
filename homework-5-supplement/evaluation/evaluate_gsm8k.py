import argparse
import json
import re
import time
from pathlib import Path


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

MAX_NEW_TOKENS = 512
STOP_SEQUENCES = ["### Instruction:"]


def build_prompt(question):
    return PROMPT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        instruction=question,
    )


def load_gsm8k_data(data_path):
    """读取原始 GSM8K JSONL，提取题目和标准数值答案。"""
    data_path = Path(data_path)

    if not data_path.is_file():
        raise FileNotFoundError(f"数据文件不存在：{data_path}")

    items = []

    with data_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            example = json.loads(line)
            question = example["question"]
            answer = example["answer"]

            if not isinstance(question, str) or not isinstance(answer, str):
                raise ValueError(
                    f"第 {line_number} 行的 question 和 answer 必须是字符串"
                )

            if "####" not in answer:
                raise ValueError(
                    f"第 {line_number} 行标准答案缺少 ####，"
                    "请使用原始 GSM8K test.jsonl"
                )

            gold_text = answer.split("####")[-1].strip().replace(",", "")

            try:
                gold = float(gold_text)
            except ValueError as exc:
                raise ValueError(
                    f"第 {line_number} 行标准答案无法转换为数字：{gold_text!r}"
                ) from exc

            items.append({
                "question": question,
                "gold": gold,
            })

    if not items:
        raise ValueError("数据文件中没有题目")

    return items


def parse_gsm8k_response(text):
    """保持原来的规则：去除数字间逗号，提取最后一个数值。"""
    if not text:
        return None

    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)

    if not numbers:
        return None

    try:
        return float(numbers[-1])
    except ValueError:
        return None


def evaluate(llm, sampling_params, items, prompts):
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    duration = time.perf_counter() - start_time

    if len(outputs) != len(items):
        raise RuntimeError("输出数量与题目数量不一致")

    details = []

    for item, request in zip(items, outputs):
        if len(request.outputs) != 1:
            raise RuntimeError("每道题应该恰好生成一个回答")

        answer = request.outputs[0]
        pred = parse_gsm8k_response(answer.text)

        details.append({
            "question": item["question"],
            "gold": item["gold"],
            "pred": pred,
            "output": answer.text,
            "is_correct": pred is not None and pred == item["gold"],
            "finish_reason": answer.finish_reason,
            "stop_reason": answer.stop_reason,
            "generated_tokens": len(answer.token_ids),
        })

    total = len(details)
    correct = sum(row["is_correct"] for row in details)

    metrics = {
        "total_evaluated": total,
        "correct_count": correct,
        "accuracy": correct / total,
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
        "throughput_samples_per_sec": (
            total / duration if duration > 0 else 0.0
        ),
    }

    return metrics, details


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用统一提示词格式评测 GSM8K"
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
        parser.error("--max_model_len 必须大于 512")
    if args.max_num_seqs < 1:
        parser.error("--max_num_seqs 至少为 1")

    return args


def main():
    args = parse_args()

    output_path = Path(args.output_file)
    if output_path.exists():
        raise FileExistsError(f"文件已存在，请使用新文件名：{output_path}")

    all_items = load_gsm8k_data(args.data_path)
    items = all_items if args.limit == 0 else all_items[:args.limit]
    prompts = [build_prompt(item["question"]) for item in items]

    print(f"模型路径：{args.model_path}")
    print(f"数据集题数：{len(all_items)}")
    print(f"本次评测题数：{len(items)}")
    print("\n第一题实际提示词：\n" + prompts[0], flush=True)

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

    for index, prompt in enumerate(prompts, start=1):
        input_length = len(tokenizer.encode(prompt))
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

    metrics, details = evaluate(
        llm, sampling_params, items, prompts
    )

    result = {
        "config": {
            **vars(args),
            "protocol": "instruction_response_v1",
            "system_prompt": SYSTEM_PROMPT,
            "prompt_template": PROMPT_TEMPLATE,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": MAX_NEW_TOKENS,
            "stop_sequences": STOP_SEQUENCES,
            "ignore_eos": False,
            "answer_extraction": "last_number",
        },
        "metrics": metrics,
        "details": details,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\nGSM8K 评测完成")
    print(f"总题数：{metrics['total_evaluated']}")
    print(f"答对题数：{metrics['correct_count']}")
    print(f"准确率：{metrics['accuracy']:.2%}")
    print(f"解析失败数：{metrics['parsing_failure_count']}")
    print(f"空回答数：{metrics['empty_count']}")
    print(f"达到长度上限的回答数：{metrics['length_limited_count']}")
    print(
        f"推理吞吐量："
        f"{metrics['throughput_samples_per_sec']:.2f} 题/秒"
    )
    print(f"结果已保存：{output_path.resolve()}")


if __name__ == "__main__":
    main()