import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


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


def extract_single_turn(text):
    """按角色边界切分，回答内部的普通空行不会被切开。"""
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
    answer = parts[1][len("Assistant:"):].strip()

    return question, answer


def summarize_lengths(values):
    values = np.asarray(values, dtype=np.int64)

    return {
        "min": int(values.min()),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": int(values.max()),
    }


def inspect_split(data_path, split, tokenizer):
    counters = Counter()
    kept_by_subset = Counter()

    lengths = {
        "prompt": [],
        "chosen_response_with_eos": [],
        "rejected_response_with_eos": [],
        "chosen_total": [],
        "rejected_total": [],
        "pair_max_total": [],
    }

    for subset in SUBSETS:
        path = data_path / subset / f"{split}.jsonl.gz"

        if not path.is_file():
            raise FileNotFoundError(path)

        print(f"\n正在读取：{path}", flush=True)

        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    counters["blank_lines"] += 1
                    continue

                counters["raw_records"] += 1

                # JSON错误、缺字段和类型错误直接报错。
                try:
                    item = json.loads(line)
                    chosen_raw = item["chosen"]
                    rejected_raw = item["rejected"]

                    if not isinstance(chosen_raw, str):
                        raise TypeError("chosen必须是字符串")
                    if not isinstance(rejected_raw, str):
                        raise TypeError("rejected必须是字符串")
                except Exception as exc:
                    raise ValueError(
                        f"{path} 第{line_number}行数据格式错误"
                    ) from exc

                chosen = extract_single_turn(chosen_raw)
                rejected = extract_single_turn(rejected_raw)

                if chosen is None or rejected is None:
                    counters["not_single_turn_or_bad_roles"] += 1
                    continue

                c_question, c_answer = chosen
                r_question, r_answer = rejected

                if not all([
                    c_question,
                    r_question,
                    c_answer,
                    r_answer,
                ]):
                    counters["empty_question_or_answer"] += 1
                    continue

                if c_question != r_question:
                    counters["question_mismatch"] += 1
                    continue

                prompt = PROMPT_TEMPLATE.format(c_question)

                p_tokens = tokenizer.encode(
                    prompt,
                    add_special_tokens=False,
                )
                c_tokens = tokenizer.encode(
                    c_answer,
                    add_special_tokens=False,
                )
                r_tokens = tokenizer.encode(
                    r_answer,
                    add_special_tokens=False,
                )

                # 与后续训练保持一致：末尾显式追加一个EOS。
                c_tokens.append(tokenizer.eos_token_id)
                r_tokens.append(tokenizer.eos_token_id)

                p_len = len(p_tokens)
                c_len = len(c_tokens)
                r_len = len(r_tokens)

                c_total = p_len + c_len
                r_total = p_len + r_len

                lengths["prompt"].append(p_len)
                lengths["chosen_response_with_eos"].append(c_len)
                lengths["rejected_response_with_eos"].append(r_len)
                lengths["chosen_total"].append(c_total)
                lengths["rejected_total"].append(r_total)
                lengths["pair_max_total"].append(
                    max(c_total, r_total)
                )

                counters["kept_pairs"] += 1
                kept_by_subset[subset] += 1

                if counters["kept_pairs"] % 10000 == 0:
                    print(
                        f"{split} 已统计合格样本对："
                        f"{counters['kept_pairs']}",
                        flush=True,
                    )

    total = counters["kept_pairs"]

    if total == 0:
        raise ValueError(f"{split}没有合格的单轮偏好数据")

    summaries = {
        name: summarize_lengths(values)
        for name, values in lengths.items()
    }

    pair_lengths = np.asarray(lengths["pair_max_total"])
    limits = {}

    for limit in [512, 1024, 2048, 4096]:
        filtered = int((pair_lengths > limit).sum())

        limits[str(limit)] = {
            "filtered_pairs": filtered,
            "filtered_fraction": filtered / total,
            "retained_pairs": total - filtered,
        }

    result = {
        "counts": dict(counters),
        "kept_by_subset": dict(kept_by_subset),
        "lengths": summaries,
        "length_limits": limits,
    }

    print(f"\n========== {split} 汇总 ==========")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="data/hh-rlhf",
    )
    parser.add_argument(
        "--model_path",
        default="result/sft_full_13129/final",
    )
    parser.add_argument(
        "--output_file",
        default="result/dpo_length_stats.json",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
    )

    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer没有eos_token_id")

    results = {
        "config": {
            "data_path": str(Path(args.data_path).resolve()),
            "model_path": str(Path(args.model_path).resolve()),
            "prompt_template": PROMPT_TEMPLATE,
            "add_special_tokens": False,
            "append_eos": True,
            "truncation": False,
        }
    }

    for split in ["train", "test"]:
        results[split] = inspect_split(
            Path(args.data_path),
            split,
            tokenizer,
        )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_name(
        output_path.name + ".tmp"
    )
    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )
    temporary_path.replace(output_path)

    print("\n统计结果已保存：", output_path.resolve())


if __name__ == "__main__":
    main()