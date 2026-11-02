import argparse
import json
import math
import os
import tempfile
from pathlib import Path

from transformers import AutoTokenizer


FILE_NAMES = (
    "sample_train.jsonl",
    "train.jsonl",
    "test.jsonl",
)


def summarize_lengths(lengths):
    """分位数使用 nearest-rank，即排序后向上取整的位置。"""
    ordered = sorted(lengths)
    count = len(ordered)

    def percentile(percent):
        index = max(0, math.ceil(count * percent / 100) - 1)
        return ordered[index]

    return {
        "count": count,
        "sum": sum(ordered),
        "mean": sum(ordered) / count,
        "min": ordered[0],
        "p50": percentile(50),
        "p75": percentile(75),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "max": ordered[-1],
    }


def inspect_file(path, tokenizer, thresholds):
    prompt_lengths = []
    response_lengths = []
    total_lengths = []

    longest_sample = None
    blank_lines = 0

    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                blank_lines += 1
                continue

            item = json.loads(line)

            if not isinstance(item, dict):
                raise ValueError(
                    f"{path} 第 {line_number} 行不是 JSON 对象"
                )

            for field in ("prompt", "response"):
                value = item.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"{path} 第 {line_number} 行的 {field} "
                        "缺失、类型错误或为空白"
                    )

            # 这里读取的是已套好模板的数据，不再重复套模板。
            # 分开编码，与之前作业的 prompt/response 编码方式一致。
            prompt_ids = tokenizer.encode(
                item["prompt"],
                add_special_tokens=False,
                truncation=False,
            )
            response_ids = tokenizer.encode(
                item["response"],
                add_special_tokens=False,
                truncation=False,
            )

            prompt_length = len(prompt_ids)
            response_length = len(response_ids)

            # 假定训练时在回答末尾显式追加一个 EOS。
            total_length = prompt_length + response_length + 1

            prompt_lengths.append(prompt_length)
            response_lengths.append(response_length)
            total_lengths.append(total_length)

            if (
                longest_sample is None
                or total_length > longest_sample["total_tokens"]
            ):
                longest_sample = {
                    "line_number": line_number,
                    "prompt_tokens": prompt_length,
                    "response_tokens": response_length,
                    "total_tokens": total_length,
                }

            count = len(total_lengths)
            if count % 10000 == 0:
                print(
                    f"  {path.name}：已统计 {count} 条",
                    flush=True,
                )

    count = len(total_lengths)
    if count == 0:
        raise ValueError(f"{path} 没有有效样本")

    threshold_results = []

    for limit in thresholds:
        # 这里只统计“会超过上限”，不实际过滤或截断。
        over_limit = sum(
            length > limit for length in total_lengths
        )

        # 若保留完整 prompt，并要求至少留一个回答 token 和 EOS，
        # 这部分样本的空间不足。
        insufficient_room = sum(
            length + 2 > limit for length in prompt_lengths
        )

        threshold_results.append({
            "max_total_tokens": limit,
            "fits_count": count - over_limit,
            "over_limit_count": over_limit,
            "over_limit_ratio": over_limit / count,
            "insufficient_room_for_answer_and_eos_count": (
                insufficient_room
            ),
        })

    return {
        "path": str(path),
        "sample_count": count,
        "blank_lines_skipped": blank_lines,
        "prompt_tokens": summarize_lengths(prompt_lengths),
        "response_tokens_without_eos": summarize_lengths(
            response_lengths
        ),
        "total_tokens_with_eos": summarize_lengths(total_lengths),
        "thresholds": threshold_results,
        "longest_sample": longest_sample,
    }


def print_report(result):
    print(f"\n文件：{Path(result['path']).name}")
    print(f"样本数：{result['sample_count']:,}")

    for title, key in (
        ("提示词", "prompt_tokens"),
        ("回答（不含 EOS）", "response_tokens_without_eos"),
        ("完整序列（含 EOS）", "total_tokens_with_eos"),
    ):
        stats = result[key]
        print(
            f"{title}：均值={stats['mean']:.1f}，"
            f"P50={stats['p50']}，P90={stats['p90']}，"
            f"P95={stats['p95']}，P99={stats['p99']}，"
            f"最大={stats['max']}"
        )

    print(
        "完整序列 token 总数："
        f"{result['total_tokens_with_eos']['sum']:,}"
    )

    print("\n长度上限 | 超长样本数 | 超长比例 | 完整提示词后空间不足")
    for row in result["thresholds"]:
        print(
            f"{row['max_total_tokens']:8d} | "
            f"{row['over_limit_count']:10d} | "
            f"{row['over_limit_ratio']:8.2%} | "
            f"{row['insufficient_room_for_answer_and_eos_count']}"
        )

    print("最长样本：", result["longest_sample"])


def save_json(path, data):
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temporary_path = Path(f.name)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")

        os.replace(temporary_path, path)
        temporary_path = None

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="仅用 CPU 统计处理后 SFT 数据的 token 长度"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096],
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if any(limit < 1 for limit in args.thresholds):
        parser.error("所有长度上限都必须大于 0")

    model_path = Path(args.model_path).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_path = Path(args.output_file).expanduser().resolve()
    thresholds = sorted(set(args.thresholds))

    if not model_path.is_dir():
        raise NotADirectoryError(f"模型目录不存在：{model_path}")

    input_paths = [data_dir / name for name in FILE_NAMES]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"数据文件不存在：{path}")

    if output_path in {path.resolve() for path in input_paths}:
        raise ValueError("统计结果不能覆盖输入数据文件")

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"结果文件已存在：{output_path}；"
            "如需覆盖，请添加 --overwrite"
        )

    print("正在从本地模型目录加载 tokenizer……", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
    )

    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer 未定义 eos_token_id")

    report = {
        "model_path": str(model_path),
        "data_dir": str(data_dir),
        "tokenizer_class": type(tokenizer).__name__,
        "eos_token_id": tokenizer.eos_token_id,
        "encoding_method": (
            "encode prompt and response separately without special "
            "tokens; count one appended EOS"
        ),
        "length_definition": (
            "prompt tokens + response tokens + 1 EOS, before label shift"
        ),
        "percentile_method": "nearest_rank",
        "thresholds": thresholds,
        "complete": False,
        "files": [],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for path in input_paths:
        print(f"\n正在统计：{path.name}", flush=True)
        result = inspect_file(path, tokenizer, thresholds)
        report["files"].append(result)
        save_json(output_path, report)
        print_report(result)

    report["complete"] = True
    save_json(output_path, report)

    print("\n三个文件统计完成")
    print("结果已保存：", output_path)


if __name__ == "__main__":
    main()