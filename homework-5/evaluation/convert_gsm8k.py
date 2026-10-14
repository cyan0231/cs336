import json
import re
from pathlib import Path


# 根据脚本位置找到 homework-5，不依赖终端当前目录
PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_DIR / "data/gsm8k/test.jsonl"
OUTPUT_FILE = PROJECT_DIR / "data/gsm8k/test_r1_style.jsonl"

SYSTEM_PROMPT_TEMPLATE = (
    "A conversation between User and Assistant. "
    "The User asks a question, and the Assistant solves it. "
    "The Assistant first thinks about the reasoning process in the mind "
    "and then provides the User with the answer. "
    "The reasoning process is enclosed within <think> </think> "
    "and answer is enclosed within <answer> </answer> tags, "
    "respectively, i.e., <think> reasoning process here </think> "
    "<answer> answer here </answer>. "
    "The final result only requires a direct output of the answer "
    "without any explanations.\n"
    "User: {question}\n"
    "Assistant: <think>"
)


def clean_gsm8k_reasoning(text):
    cleaned = re.sub(r"<<.*?>>", "", text)
    return cleaned.strip()


def process_line(line):
    item = json.loads(line)
    question = item["question"]
    answer_field = item["answer"]

    if "####" not in answer_field:
        raise ValueError("数据缺少 #### 答案分隔符")

    reasoning_part, gold_answer = answer_field.rsplit("####", 1)
    reasoning_part = clean_gsm8k_reasoning(reasoning_part)
    gold_answer = gold_answer.strip()

    prompt = SYSTEM_PROMPT_TEMPLATE.format(question=question)
    response = (
        f"{reasoning_part} </think> <answer>{gold_answer}</answer>"
    )

    return {
        "question": question,
        "prompt": prompt,
        "response": response,
        "gold": gold_answer,
        # 以下是标准数据的预设标记，不是模型评测结果
        "is_correct": True,
        "reward_details": {
            "format_reward": 1.0,
            "answer_reward": 1.0,
            "reward": 1.0,
        },
    }


def main():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"找不到测试集：{INPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with INPUT_FILE.open(encoding="utf-8") as f_in:
        with OUTPUT_FILE.open("w", encoding="utf-8") as f_out:
            for line in f_in:
                if not line.strip():
                    continue

                item = process_line(line)
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1

    print(f"转换完成：{count} 条")
    print(f"输出文件：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()