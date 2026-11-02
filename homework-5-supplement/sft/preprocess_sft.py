import argparse
import json
import os
import tempfile
from pathlib import Path


FILE_NAMES = (
    "sample_train.jsonl",
    "train.jsonl",
    "test.jsonl",
)

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
    SYSTEM_PROMPT
    + "\n\n"
    + "### Instruction:\n{instruction}\n\n"
    + "### Response:\n"
)


def build_prompt(instruction):
    """使用与四项测评一致的系统提示词和外层格式。"""
    return PROMPT_TEMPLATE.format(instruction=instruction)


def process_file(input_path, output_path):
    """逐行转换；空白字段跳过，其他异常立即报错。"""
    sample_count = 0
    nonempty_lines = 0
    blank_lines = 0
    skipped_samples = []
    first_example = None
    temporary_path = None

    try:
        with input_path.open(encoding="utf-8") as source:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as destination:
                temporary_path = Path(destination.name)

                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        blank_lines += 1
                        continue

                    nonempty_lines += 1
                    location = f"{input_path} 第 {line_number} 行"

                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"{location}：不是合法 JSON，{exc.msg}"
                        ) from exc

                    if not isinstance(item, dict):
                        raise ValueError(
                            f"{location}：每条数据必须是 JSON 对象"
                        )

                    # 先检查字段是否存在、类型是否正确。
                    # 这些问题不能静默跳过。
                    for field in ("prompt", "response"):
                        if field not in item:
                            raise ValueError(
                                f"{location}：缺少 {field} 字段"
                            )

                        if not isinstance(item[field], str):
                            raise ValueError(
                                f"{location}：{field} 必须是字符串"
                            )

                    # 空问题或空回答：跳过整条，并记录原因。
                    empty_fields = [
                        field
                        for field in ("prompt", "response")
                        if not item[field].strip()
                    ]

                    if empty_fields:
                        skipped_samples.append({
                            "line_number": line_number,
                            "reason": "empty_or_whitespace",
                            "fields": empty_fields,
                        })
                        continue

                    # strip() 只用于空白检查，不修改原始文本。
                    processed = {
                        "prompt": build_prompt(item["prompt"]),
                        "response": item["response"],
                    }

                    destination.write(
                        json.dumps(processed, ensure_ascii=False) + "\n"
                    )

                    sample_count += 1

                    if first_example is None:
                        first_example = processed

                if sample_count == 0:
                    raise ValueError(
                        f"{input_path}：过滤后没有有效样本"
                    )

            # 文件全部处理成功，才替换正式输出。
            os.replace(temporary_path, output_path)
            temporary_path = None

    finally:
        # 出错时清理临时文件，保留原有正式输出。
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    statistics = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "nonempty_line_count": nonempty_lines,
        "sample_count": sample_count,
        "blank_lines_skipped": blank_lines,
        "skipped_sample_count": len(skipped_samples),
        "skipped_samples": skipped_samples,
    }

    return statistics, first_example


def save_json(path, data):
    """先写临时文件，再替换正式文件。"""
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
        description="过滤空白样本，并生成套好模板的 SFT 数据"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="包含三个原始 JSONL 文件的目录",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="处理后数据的保存目录",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许替换已有的处理后文件",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入目录不存在：{input_dir}")

    if input_dir == output_dir:
        raise ValueError(
            "输入目录和输出目录不能相同，避免覆盖原始数据"
        )

    input_paths = [input_dir / name for name in FILE_NAMES]
    output_paths = [output_dir / name for name in FILE_NAMES]
    config_path = output_dir / "preprocess_config.json"

    # 开始前确认三个源文件都存在。
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"原始数据文件不存在：{path}")

    source_paths = {path.resolve() for path in input_paths}
    targets = output_paths + [config_path]

    for path in targets:
        if path.resolve() in source_paths:
            raise ValueError(f"输出路径指向原始文件：{path}")

        if path.exists() and not path.is_file():
            raise ValueError(
                f"输出路径已存在且不是文件：{path}"
            )

    existing = [path for path in targets if path.exists()]

    if existing and not args.overwrite:
        files = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "以下输出文件已存在：\n"
            f"{files}\n"
            "请使用新目录，或显式添加 --overwrite。"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "complete": False,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "system_prompt": SYSTEM_PROMPT,
        "prompt_template": PROMPT_TEMPLATE,
        "output_fields": ["prompt", "response"],
        "tokenized": False,
        "eos_added": False,
        "truncated": False,
        "empty_field_policy": "skip_and_record",
        "invalid_json_or_schema_policy": "raise",
        "files": [],
    }

    # 先标记为未完成，避免失败后误读旧的完成记录。
    save_json(config_path, config)

    example = None

    for input_path, output_path in zip(input_paths, output_paths):
        print(f"\n正在处理：{input_path.name}", flush=True)

        statistics, first_example = process_file(
            input_path,
            output_path,
        )

        config["files"].append(statistics)

        # 每完成一个文件，就保存该文件的数量和过滤记录。
        save_json(config_path, config)

        if example is None:
            example = first_example

        print(
            f"原始样本：{statistics['nonempty_line_count']} 条",
            flush=True,
        )
        print(
            f"保留样本：{statistics['sample_count']} 条",
            flush=True,
        )
        print(
            f"过滤样本：{statistics['skipped_sample_count']} 条",
            flush=True,
        )
        print(
            f"忽略空白行：{statistics['blank_lines_skipped']} 行",
            flush=True,
        )
        print(f"输出：{output_path}", flush=True)

    config["complete"] = True
    save_json(config_path, config)

    print("\n第一条处理后的示例：")
    print(json.dumps(example, ensure_ascii=False, indent=2))

    print("\n全部预处理完成")
    print(f"处理记录：{config_path}")


if __name__ == "__main__":
    main()