import argparse
import json
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

MAX_NEW_TOKENS = 1024
STOP_SEQUENCES = ["### Instruction:"]


def build_prompt(instruction):
    """输入中不包含参考回答。"""
    return PROMPT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        instruction=instruction,
    )


def main():
    parser = argparse.ArgumentParser(
        description="使用统一提示词格式生成 AlpacaEval 回答"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--generator", default="Qwen2.5-1.5B-Base")
    parser.add_argument(
        "--limit", type=int, default=10,
        help="默认前 10 题，0 表示全部",
    )
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.7
    )
    parser.add_argument("--max_num_seqs", type=int, default=4)
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit 不能小于 0")
    if args.max_model_len <= MAX_NEW_TOKENS:
        parser.error("--max_model_len 必须大于 1024")
    if args.max_num_seqs < 1:
        parser.error("--max_num_seqs 至少为 1")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu_memory_utilization 必须在 0 和 1 之间")

    output_path = Path(args.output_file)
    config_path = output_path.with_suffix(".config.json")

    for path in (output_path, config_path):
        if path.exists():
            raise FileExistsError(f"文件已存在，请使用新文件名：{path}")

    with open(args.data_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if not rows:
        raise ValueError("数据集为空")

    for index, row in enumerate(rows, start=1):
        if not isinstance(row.get("instruction"), str):
            raise ValueError(
                f"第 {index} 条 instruction 缺失或不是字符串"
            )

    if args.limit:
        rows = rows[:args.limit]

    prompts = [build_prompt(row["instruction"]) for row in rows]

    print(f"本次生成题数：{len(rows)}")
    print("\n第一题实际提示词：\n" + prompts[0], flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=True,
        seed=0,
    )

    tokenizer = llm.get_tokenizer()

    for index, prompt in enumerate(prompts, start=1):
        length = len(tokenizer.encode(prompt))
        if length + MAX_NEW_TOKENS > args.max_model_len:
            raise ValueError(
                f"第 {index} 条输入为 {length} token，"
                f"加上 {MAX_NEW_TOKENS} 个输出 token 后超过 "
                f"max_model_len={args.max_model_len}。"
                "请提高 --max_model_len 后重新运行。"
            )

    params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=MAX_NEW_TOKENS,
        stop=STOP_SEQUENCES,
        ignore_eos=False,
        include_stop_str_in_output=False,
    )

    outputs = llm.generate(prompts, params, use_tqdm=True)

    if len(outputs) != len(rows):
        raise RuntimeError("返回数量与输入题数不一致")

    predictions = []
    diagnostics = []

    for row, request in zip(rows, outputs):
        if len(request.outputs) != 1:
            raise RuntimeError("每条指令应恰好生成一个回答")

        answer = request.outputs[0]

        # 保持原有字段，兼容 judge_alpaca_tool.py。
        predictions.append({
            "instruction": row["instruction"],
            "output": answer.text.strip(),
            "generator": args.generator,
            "dataset": row.get("dataset", "alpaca_eval"),
        })

        diagnostics.append({
            "finish_reason": answer.finish_reason,
            "stop_reason": answer.stop_reason,
            "generated_tokens": len(answer.token_ids),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # AlpacaEval 回答文件是 JSON 数组。
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    with config_path.open("w", encoding="utf-8") as f:
        json.dump({
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
                "dtype": "bfloat16",
                "seed": 0,
            },
            "diagnostics": diagnostics,
        }, f, ensure_ascii=False, indent=2)

    print("\n生成完成")
    print("题数：", len(predictions))
    print("空回答数：", sum(not row["output"] for row in predictions))
    print("达到长度上限的回答数：", sum(
        row["finish_reason"] == "length" for row in diagnostics
    ))
    print("回答文件：", output_path.resolve())
    print("配置与诊断：", config_path.resolve())

    for row in predictions[:2]:
        print("\n指令：", row["instruction"])
        print("回答前 500 字符：", repr(row["output"][:500]))


if __name__ == "__main__":
    main()