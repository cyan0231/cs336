import argparse
import json
import re
from pathlib import Path

from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn


PROJECT_DIR = Path(__file__).resolve().parent.parent


def load_math_data(data_path):
    with open(data_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_vllm(
    vllm_model,
    reward_fn,
    prompts,
    ground_truths,
    eval_sampling_params,
    output_file,
):
    if not prompts:
        raise ValueError("没有可评测的题目")
    if len(prompts) != len(ground_truths):
        raise ValueError("题目数量和标准答案数量不一致")

    print(f"开始生成，共 {len(prompts)} 道题……")
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    if len(outputs) != len(ground_truths):
        raise ValueError("模型输出数量和标准答案数量不一致")

    stats = {
        "total": 0,
        "correct": 0,
        "format_error": 0,
        "correct_format_wrong_answer": 0,
    }

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 逐题评分并保存；同名结果文件会被覆盖
    with output_file.open("w", encoding="utf-8") as f:
        for output_item, gold_answer in zip(outputs, ground_truths):
            generated_text = output_item.outputs[0].text
            scores = reward_fn(generated_text, gold_answer)

            stats["total"] += 1

            if scores.get("reward", 0) == 1:
                stats["correct"] += 1

            if scores.get("format_reward", 0) == 0:
                stats["format_error"] += 1
            elif scores.get("answer_reward", 0) == 0:
                stats["correct_format_wrong_answer"] += 1

            result_entry = {
                "prompt": output_item.prompt,
                "ground_truth": gold_answer,
                "generated_text": generated_text,
                "scores": scores,
            }
            f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

    accuracy = stats["correct"] / stats["total"]

    print("\n" + "=" * 40)
    print(f"总题数：{stats['total']}")
    print(f"完全正确：{stats['correct']}")
    print(f"正确率（格式和答案均通过）：{accuracy:.2%}")
    print(f"格式错误：{stats['format_error']}")
    print(f"格式正确但答案错误：{stats['correct_format_wrong_answer']}")
    print(f"结果已保存：{output_file}")
    print("=" * 40)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        default=str(PROJECT_DIR / "model/Qwen2.5-Math-1.5B"),
    )
    parser.add_argument(
        "--data_path",
        default=str(PROJECT_DIR / "data/gsm8k/test_r1_style.jsonl"),
    )
    parser.add_argument(
        "--output_file",
        default=str(PROJECT_DIR / "result/baseline_10.jsonl"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="评测题数；0 表示全部，默认 10",
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit 不能小于 0")

    raw_data = load_math_data(args.data_path)
    if args.limit > 0:
        raw_data = raw_data[:args.limit]

    prompts = []
    ground_truths = []

    ans_re = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)

    for item in raw_data:
        full_prompt = item["prompt"]
        raw_response = item["response"]

        match = ans_re.search(raw_response)
        if match:
            gold_answer = match.group(1).strip()
        else:
            gold_answer = raw_response.split("####")[-1].strip()

        prompts.append(full_prompt)
        ground_truths.append(gold_answer)

    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        enforce_eager=True,
        seed=0,
    )

    sampling_params = SamplingParams(
        n=1,
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # 确保每题都有足够空间生成 1024 个 token，
    # 避免总长度上限悄悄缩短回答。
    tokenizer = llm.get_tokenizer()
    for i, prompt in enumerate(prompts):
        input_length = len(tokenizer.encode(prompt))
        if input_length + sampling_params.max_tokens > 2048:
            raise ValueError(
                f"第 {i + 1} 题输入有 {input_length} 个 token，"
                "加上 1024 个输出 token 超过 max_model_len=2048；"
                "请调大模型配置及此处检查中的总长度上限。"
            )

    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()