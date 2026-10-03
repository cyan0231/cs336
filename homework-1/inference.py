"""交互式文本续写：python inference.py，每条提示词独立生成。"""

import argparse
import math
from pathlib import Path
import sys

import torch

from nn import TransformerLM
from preprocess import SPECIAL_TOKENS, load_trained_tokenizer
import train as training_config


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="加载当前 CS336 模型，交互式续写文本。")
    parser.add_argument("--checkpoint", type=Path, default=BASE_DIR / "output" / "checkpoint.pt")
    parser.add_argument("--vocab", type=Path, default=BASE_DIR / "output" / "vocab.json")
    parser.add_argument("--merges", type=Path, default=BASE_DIR / "output" / "merges.txt")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)

    # 默认值复用当前训练配置；加载旧权重时，可以显式指定当时的参数。
    parser.add_argument("--vocab-size", type=int, default=training_config.vocab_size)
    parser.add_argument("--context-length", type=int, default=training_config.context_length)
    parser.add_argument("--d-model", type=int, default=training_config.d_model)
    parser.add_argument("--num-layers", type=int, default=training_config.num_layers)
    parser.add_argument("--num-heads", type=int, default=training_config.attention_head)
    parser.add_argument("--d-ff", type=int, default=training_config.d_ff)
    parser.add_argument("--theta", type=float, default=training_config.theta)
    args = parser.parse_args()

    for name in ("max_new_tokens", "vocab_size", "context_length", "d_model",
                 "num_layers", "num_heads", "d_ff"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须是正整数")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature 必须是有限的正数")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p 必须满足 0 < top_p <= 1")
    if not math.isfinite(args.theta) or args.theta <= 0:
        parser.error("--theta 必须是有限的正数")
    if args.d_model % args.num_heads != 0:
        parser.error("--d-model 必须能被 --num-heads 整除")
    if (args.d_model // args.num_heads) % 2 != 0:
        parser.error("RoPE 要求每个注意力头的维度为偶数")
    for name in ("checkpoint", "vocab", "merges"):
        if not getattr(args, name).is_file():
            parser.error(f"找不到 {name} 文件：{getattr(args, name)}")
    return args


def load_model(args: argparse.Namespace, device: torch.device) -> TransformerLM:
    # 先在 CPU 加载，再整体移动到推理设备。
    # 兼容不支持 weights_only 的旧版 PyTorch；只加载自己训练保存的可信 checkpoint。
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("checkpoint 中缺少 model_state，请使用当前 checkpoint.py 保存的文件。")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        head_num=args.num_heads,
        d_ff=args.d_ff,
        context_length=args.context_length,
        theta=args.theta,
        output_size=args.vocab_size,
        num_layers=args.num_layers,
        device="cpu",
    )
    try:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except RuntimeError as error:
        raise ValueError(
            "权重与模型结构不匹配，请核对训练时的模型参数。\n" + str(error)
        ) from error
    model.to(device)
    model.eval()
    return model


def main() -> int:
    args = parse_args()
    try:
        device_name = args.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise ValueError("当前环境无法使用 CUDA，可检查 GPU 环境或指定 --device cpu。")
        device = torch.device(device_name)

        tokenizer = load_trained_tokenizer(args.vocab, args.merges, SPECIAL_TOKENS)
        if set(tokenizer.vocab) != set(range(args.vocab_size)):
            raise ValueError(
                "分词器的 ID 范围与 --vocab-size 不一致，请使用训练模型时的同一套词表。"
            )
        eos_token_id = tokenizer.bytes_to_id[b"<|endoftext|>"]
        model = load_model(args, device)

        print(f"模型已加载，设备：{device}，上下文长度：{args.context_length} tokens")
        print(f"最多新增 {args.max_new_tokens} tokens；"
              f"temperature={args.temperature}，top_p={args.top_p}")
        print("每次输入独立续写，不保留历史。输入 q / exit / quit 退出。")

        while True:
            prompt = input("\n提示词 > ")
            if prompt.strip().lower() in {"q", "exit", "quit"}:
                break
            if not prompt.strip():
                continue

            # 每轮只编码这一次的输入，batch 固定为 1。
            prompt_ids = tokenizer.encode(prompt)
            if not prompt_ids:
                continue
            if len(prompt_ids) > args.context_length:
                print(f"提示词有 {len(prompt_ids)} tokens，"
                      f"生成时仅使用最近 {args.context_length} 个 token 作为上下文。")
            inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            with torch.inference_mode():
                generated = model.generate(
                    inputs,
                    max_answer=args.max_new_tokens,
                    eos_token_id=eos_token_id,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

            # generate 返回“提示词 + 新 token”，这里只显示新增的正文。
            new_ids = generated[0, len(prompt_ids):].tolist()
            if new_ids and new_ids[-1] == eos_token_id:
                new_ids.pop()
            continuation = tokenizer.decode(new_ids)
            print("续写 > " + (continuation if continuation else "（没有新增正文，生成已结束。）"))

        return 0
    except (KeyboardInterrupt, EOFError):
        print("\n已退出。")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"推理失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
