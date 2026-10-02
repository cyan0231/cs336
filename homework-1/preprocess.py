"""将脚本旁的 TinyStories 文本编码为 uint16 原始二进制文件。"""

import json
import os
import tempfile
from pathlib import Path
import numpy as np

from bpe import bytes_to_unicode
from tokenizer import BPETokenizer


# 路径以当前脚本所在目录为基准，整个文件夹搬到服务器后仍然有效。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(BASE_DIR, "output", "vocab.json")
MERGES_PATH = os.path.join(BASE_DIR, "output", "merges.txt")
TRAIN_TXT = os.path.join(BASE_DIR, "TinyStoriesV2-GPT4-train.txt")
VALID_TXT = os.path.join(BASE_DIR, "TinyStoriesV2-GPT4-valid.txt")
TRAIN_BIN = os.path.join(BASE_DIR, "TinyStoriesV2-GPT4-train.bin")
VALID_BIN = os.path.join(BASE_DIR, "TinyStoriesV2-GPT4-valid.bin")

SPECIAL_TOKENS = ["<|endoftext|>"]
DOCUMENT_SEPARATOR = "<|endoftext|>"
READ_CHARS = 1024 * 1024  # 文本模式按字符读取，不是按字节读取。
WRITE_BATCH_SIZE = 100_000
# 保留完整文档再编码；遇到异常长文档时停止，避免悄悄读入整个大文件。
MAX_DOCUMENT_CHARS = 16 * 1024 * 1024
OUTPUT_DTYPE = np.dtype("uint16")


def load_trained_tokenizer(vocab_path, merges_path, special_tokens):
    """还原 bpe.py 保存的 ID→bytes 词表和有序合并规则。"""
    byte_decoder = {char: byte for byte, char in bytes_to_unicode().items()}

    def restore_bytes(text):
        return bytes(byte_decoder[char] for char in text)

    with open(vocab_path, "r", encoding="utf-8") as f:
        raw_vocab = json.load(f)
    vocab = {int(token_id): restore_bytes(text) for token_id, text in raw_vocab.items()}
    if not vocab:
        raise ValueError("词表不能为空")
    if min(vocab) < 0 or max(vocab) > np.iinfo(OUTPUT_DTYPE).max:
        raise ValueError("词表 ID 超出 uint16 的范围 0～65535")
    if len(set(vocab.values())) != len(vocab):
        raise ValueError("词表中存在重复的 bytes，无法建立唯一的 bytes→ID 映射")

    token_bytes = set(vocab.values())
    if any(bytes([i]) not in token_bytes for i in range(256)):
        raise ValueError("词表缺少基础字节 token")
    for token in special_tokens:
        if not token or token.encode("utf-8") not in token_bytes:
            raise ValueError(f"特殊 token 为空或未包含在词表中：{token!r}")

    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split(" ")
            if len(parts) != 2 or not all(parts):
                raise ValueError(f"合并规则第 {line_number} 行应包含两个以空格分隔的 token")
            left, right = map(restore_bytes, parts)
            if any(part not in token_bytes for part in (left, right, left + right)):
                raise ValueError(f"合并规则第 {line_number} 行与词表不匹配")
            merges.append((left, right))

    print(f"加载分词器：{len(vocab):,} 个 token，{len(merges):,} 条合并规则")
    return BPETokenizer(vocab, merges, special_tokens)


def iter_documents(input_txt, read_chars=READ_CHARS, max_document_chars=MAX_DOCUMENT_CHARS):
    """按完整特殊标记切分，保留所有原文字符和标记，不按任意字符数直接编码。"""
    if read_chars <= 0 or max_document_chars <= 0:
        raise ValueError("读取块大小和文档长度上限必须大于 0")
    pending = ""
    # newline="" 保留原始换行符，不把 CRLF 自动改成 LF。
    with open(input_txt, "r", encoding="utf-8", newline="") as f:
        while True:
            chunk = f.read(read_chars)
            if not chunk:
                break
            pending += chunk
            start = 0
            while True:
                boundary = pending.find(DOCUMENT_SEPARATOR, start)
                if boundary == -1:
                    break
                end = boundary + len(DOCUMENT_SEPARATOR)
                if end - start > max_document_chars:
                    raise ValueError("文档过长，请检查语料分隔标记或调整 MAX_DOCUMENT_CHARS")
                yield pending[start:end]
                start = end
            pending = pending[start:]
            if len(pending) > max_document_chars:
                raise ValueError("长时间未找到文档分隔标记，请检查语料或调整 MAX_DOCUMENT_CHARS")
        if pending:
            yield pending


def process_corpus(input_txt, output_bin, tokenizer, read_chars=READ_CHARS,
                   write_batch_size=WRITE_BATCH_SIZE):
    """流式编码，缓冲一批 ID 后写盘。失败时清理临时文件，不覆盖已有输出。"""
    if write_batch_size <= 0:
        raise ValueError("写入批大小必须大于 0")
    if DOCUMENT_SEPARATOR not in tokenizer.special_token:
        raise ValueError("文档分隔标记必须被分词器识别为特殊 token")
    source, destination = Path(input_txt), Path(output_bin)
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f"输出已存在，请更换路径或自行处理旧文件：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"开始处理：{source} → {destination}")

    total_tokens = 0
    token_buffer = []
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent,
                                         prefix=".preprocess-", delete=False) as output:
            temporary_path = Path(output.name)
            documents = iter_documents(source, read_chars=read_chars)
            for token_id in tokenizer.encode_iterable(documents):
                if not 0 <= token_id <= np.iinfo(OUTPUT_DTYPE).max:
                    raise ValueError(f"token ID 超出 uint16 范围：{token_id}")
                token_buffer.append(token_id)
                if len(token_buffer) >= write_batch_size:
                    output.write(np.asarray(token_buffer, dtype=OUTPUT_DTYPE).tobytes())
                    total_tokens += len(token_buffer)
                    token_buffer.clear()
                    print(f"已写入 {total_tokens:,} tokens", flush=True)
            if token_buffer:
                output.write(np.asarray(token_buffer, dtype=OUTPUT_DTYPE).tobytes())
                total_tokens += len(token_buffer)
        # 同目录临时文件写完后才发布；link 不会覆盖已经存在的目标文件。
        os.link(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print(f"完成：{total_tokens:,} tokens，{destination.stat().st_size:,} 字节，dtype=uint16")
    return total_tokens


def main():
    paths = {
        "VOCAB_PATH": VOCAB_PATH, "MERGES_PATH": MERGES_PATH,
        "TRAIN_TXT": TRAIN_TXT, "VALID_TXT": VALID_TXT,
        "TRAIN_BIN": TRAIN_BIN, "VALID_BIN": VALID_BIN,
    }
    missing = [name for name, value in paths.items() if not value.strip()]
    if missing:
        raise ValueError("请先填写文件顶部的路径：" + ", ".join(missing))
    for name in ("VOCAB_PATH", "MERGES_PATH", "TRAIN_TXT", "VALID_TXT"):
        if not Path(paths[name]).is_file():
            raise FileNotFoundError(f"{name}: {paths[name]}")
    if Path(TRAIN_BIN).resolve() == Path(VALID_BIN).resolve():
        raise ValueError("训练集与验证集必须使用不同的输出路径")
    for output in (TRAIN_BIN, VALID_BIN):
        if Path(output).exists():
            raise FileExistsError(f"输出已存在：{output}")

    tokenizer = load_trained_tokenizer(VOCAB_PATH, MERGES_PATH, SPECIAL_TOKENS)
    process_corpus(TRAIN_TXT, TRAIN_BIN, tokenizer)
    process_corpus(VALID_TXT, VALID_BIN, tokenizer)


if __name__ == "__main__":
    main()
