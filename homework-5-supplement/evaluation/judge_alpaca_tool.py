import argparse
import hashlib
import json
import logging
import multiprocessing
import os
import time
import uuid
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    wait,
)
from pathlib import Path


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    """仅由主进程调用：写临时文件，再替换正式文件。"""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    temporary.replace(path)


def judge_one_attempt(
    index,
    prediction,
    reference,
    config_dir,
    output_dir,
    attempt,
):
    """子进程负责一道题的一次评分，不修改共享进度文件。"""
    # 第一次立即执行；重试前等待 2 秒、4 秒……
    if attempt > 1:
        time.sleep(2 * (attempt - 1))

    attempt_name = f"{index + 1:04d}_{uuid.uuid4().hex}"
    attempt_dir = Path(output_dir) / "attempts" / attempt_name

    record = {
        "question_number": index + 1,
        "attempt": attempt,
        "directory": str(attempt_dir),
    }

    try:
        attempt_dir.mkdir(parents=True)

        # API 环境变量由主进程提前设置，spawn 子进程继承。
        from alpaca_eval import evaluate

        logging.getLogger().setLevel(logging.WARNING)

        evaluate(
            model_outputs=[prediction],
            reference_outputs=[reference],
            annotators_config=str(config_dir),
            base_dir=str(Path(config_dir).parent),
            output_path=str(attempt_dir),

            # 不复用默认排行榜。
            precomputed_leaderboard=None,
            is_cache_leaderboard=False,
            is_return_instead_of_print=True,

            # 保持普通胜率及原来的随机种子。
            fn_metric="get_winrate",
            sort_by="win_rate",
            seed=0,

            # 每次尝试使用独立缓存和输出目录。
            caching_path=str(attempt_dir / "cache.json"),
            is_store_missing_annotations=True,
        )

        annotations = read_json(attempt_dir / "annotations.json")

        if not isinstance(annotations, list) or len(annotations) != 1:
            raise ValueError("没有得到一条评分记录")

        annotation = annotations[0]
        preference = annotation.get("preference")

        # 官方工具将随机排列后的结果映射回原始顺序：
        # 1   = 参考回答获胜
        # 2   = 本地模型回答获胜
        # 1.5 = 平局
        if isinstance(preference, bool) or preference not in (1, 1.5, 2):
            raise ValueError("裁判结果无法解析为有效偏好")

        record["status"] = "success"
        return annotation, record

    except Exception as exc:
        record["status"] = "failed"
        record["error_type"] = type(exc).__name__
        record["http_status"] = getattr(exc, "status_code", None)
        return None, record


def write_summary(output_dir, state, total):
    """按照原始题目顺序汇总，继续使用官方普通胜率函数。"""
    from alpaca_eval.metrics import get_winrate

    annotations = []
    missing_questions = []

    for index in range(total):
        item_key = str(index)

        if item_key in state["completed"]:
            annotations.append(state["completed"][item_key])
        else:
            missing_questions.append(index + 1)

    metrics = get_winrate(annotations) if annotations else {}

    metrics = {
        name: value.item() if hasattr(value, "item") else value
        for name, value in metrics.items()
    }

    summary = {
        "total": total,
        "valid": len(annotations),
        "missing": len(missing_questions),
        "missing_question_numbers": missing_questions,
        "complete": len(missing_questions) == 0,
        "metrics": metrics,
    }

    save_json(output_dir / "annotations.json", annotations)
    save_json(output_dir / "summary.json", summary)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="使用官方 AlpacaEval 工具和 DeepSeek 并发评分"
    )
    parser.add_argument("--model_outputs", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--annotators_config",
        default="evaluation/deepseek_judge",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="默认评前 10 条，0 表示全部",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="同时评分的进程数，默认 4",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=3,
        help="每次运行中，每道失败题最多尝试几次",
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit 不能小于 0")
    if args.max_workers < 1:
        parser.error("--max_workers 至少为 1")
    if args.max_attempts < 1:
        parser.error("--max_attempts 至少为 1")

    # 1. 读取模型回答。
    predictions = read_json(args.model_outputs)

    if not isinstance(predictions, list) or not predictions:
        raise ValueError("模型回答文件必须是非空 JSON 数组")

    if args.limit:
        predictions = predictions[:args.limit]

    total = len(predictions)

    # 2. 按文件顺序读取参考回答。
    with open(args.data_path, encoding="utf-8") as f:
        references = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if len(references) < total:
        raise ValueError("参考数据条数少于模型回答条数")

    selected_references = references[:total]

    # 3. 按位置检查，两边必须对应同一道题。
    for index, (prediction, reference) in enumerate(
        zip(predictions, selected_references),
        start=1,
    ):
        if not isinstance(prediction, dict) or not isinstance(reference, dict):
            raise ValueError(f"第 {index} 条数据必须是 JSON 对象")

        for field in ("instruction", "output", "generator"):
            if not isinstance(prediction.get(field), str):
                raise ValueError(
                    f"第 {index} 条模型回答的 {field} 字段不是字符串"
                )

            if not isinstance(reference.get(field), str):
                raise ValueError(
                    f"第 {index} 条参考数据的 {field} 字段不是字符串"
                )

        if prediction["instruction"] != reference["instruction"]:
            raise ValueError(
                f"第 {index} 条指令不一致，请检查数据和生成结果"
            )

    # 4. 读取裁判配置，保持原来的实验指纹计算方式。
    config_dir = Path(args.annotators_config).resolve()

    config_text = (config_dir / "configs.yaml").read_text(
        encoding="utf-8"
    )
    prompt_text = (config_dir / "prompt.txt").read_text(
        encoding="utf-8"
    )

    experiment = {
        "predictions": predictions,
        "references": selected_references,
        "judge_config": config_text,
        "judge_prompt": prompt_text,
        "seed": 0,
        "metric": "get_winrate",
    }

    # 并发数、重试次数不改变评分规则，不加入实验指纹。
    fingerprint = hashlib.sha256(
        json.dumps(
            experiment,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    # 5. 创建结果目录或恢复已有进度。
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    state_path = output_dir / "progress.json"

    if state_path.exists():
        state = read_json(state_path)

        if state["fingerprint"] != fingerprint:
            raise ValueError(
                "输入或裁判配置已改变，请使用新的 --output_dir"
            )
    else:
        state = {
            "fingerprint": fingerprint,
            "total": total,
            "completed": {},
            "attempts": [],
        }
        save_json(output_dir / "experiment.json", experiment)
        save_json(state_path, state)

    # 6. 设置 API 环境变量，必须早于导入官方工具及启动子进程。
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()

    if not key or any(char.isspace() for char in key):
        raise ValueError("DEEPSEEK_API_KEY 为空或包含空白")

    os.environ["OPENAI_API_KEY"] = key
    os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"
    os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com"

    # 提前验证评分依赖，避免每道题都因缺失依赖失败。
    from alpaca_eval import evaluate
    from alpaca_eval.metrics import get_winrate

    logging.getLogger().setLevel(logging.WARNING)

    pending_indices = [
        index
        for index in range(total)
        if str(index) not in state["completed"]
    ]

    print("本次评分题数：", total)
    print("已有成功评分：", len(state["completed"]))
    print("待评分题数：", len(pending_indices))
    print("并发进程数：", args.max_workers)
    print("结果目录：", output_dir, flush=True)

    # 7. 并发执行；只有主进程修改和保存 state。
    if pending_indices:
        context = multiprocessing.get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=args.max_workers,
            mp_context=context,
        ) as executor:
            futures = {}
            remaining = iter(pending_indices)

            def submit_attempt(index, attempt):
                future = executor.submit(
                    judge_one_attempt,
                    index,
                    predictions[index],
                    selected_references[index],
                    str(config_dir),
                    str(output_dir),
                    attempt,
                )
                futures[future] = (index, attempt)

            def submit_next_question():
                index = next(remaining, None)
                if index is not None:
                    submit_attempt(index, 1)

            # 同时只保持最多 max_workers 个任务。
            for _ in range(min(args.max_workers, len(pending_indices))):
                submit_next_question()

            while futures:
                finished, _ = wait(
                    futures,
                    return_when=FIRST_COMPLETED,
                )

                for future in finished:
                    index, attempt = futures.pop(future)

                    try:
                        annotation, record = future.result()
                    except Exception as exc:
                        # 这里通常表示进程异常，而非普通 API 错误。
                        state["attempts"].append({
                            "question_number": index + 1,
                            "attempt": attempt,
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "http_status": None,
                        })
                        save_json(state_path, state)

                        for pending_future in futures:
                            pending_future.cancel()

                        raise RuntimeError(
                            "评分子进程异常，已保存此前进度。"
                            "修复问题后重新执行相同命令即可续评。"
                        ) from exc

                    state["attempts"].append(record)

                    if annotation is not None:
                        state["completed"][str(index)] = annotation

                    # 每次成功或失败尝试结束，都保存进度。
                    save_json(state_path, state)

                    if annotation is not None:
                        print(
                            f"第 {index + 1} 题评分成功；"
                            f"已成功评分：{len(state['completed'])}/{total}",
                            flush=True,
                        )
                        submit_next_question()
                    else:
                        print(
                            f"第 {index + 1} 题，第 {attempt} 次尝试失败："
                            f"{record['error_type']}；"
                            f"HTTP 状态：{record['http_status']}",
                            flush=True,
                        )

                        if attempt < args.max_attempts:
                            submit_attempt(index, attempt + 1)
                        else:
                            print(
                                f"第 {index + 1} 题本轮重试已用完，"
                                "下次运行可继续补评。",
                                flush=True,
                            )
                            submit_next_question()

    # 8. 按原题顺序汇总。
    summary = write_summary(output_dir, state, total)

    print("\n有效评分数：", summary["valid"])
    print("缺失评分数：", summary["missing"])

    if summary["valid"]:
        win_rate = summary["metrics"]["win_rate"]

        if summary["missing"]:
            print(f"已评分部分的临时胜率：{win_rate:.2f}%")
        else:
            print(f"普通胜率：{win_rate:.2f}%")

    if summary["complete"]:
        print("本次评分全部完成。")
    else:
        print("评分尚未全部完成。")
        print("重新运行相同命令，可以补评缺失题目。")

    print("结果目录：", output_dir)


if __name__ == "__main__":
    main()