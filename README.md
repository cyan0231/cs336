# CS336 学习记录

用于管理 CS336 的课程笔记、作业代码和实验项目。

## 日常 Git 操作

```bash
git status
git add <文件或目录>
git commit -m "描述本次修改"
git push
```

数据集、模型权重、实验输出和密钥不提交到仓库。

## 交互式推理

在包含 `inference.py` 的目录执行：

```bash
python -u inference.py
```

默认读取脚本旁的 `output/checkpoint.pt`、`output/vocab.json` 和
`output/merges.txt`，模型结构默认复用当前 `train.py` 的配置。
只加载模型权重，不恢复优化器，也不会开始训练。

每次输入一条提示词，生成完成后显示新增的续写内容。每轮独立，batch 为 1，
不保留历史；空输入跳过，输入 `q`、`exit`、`quit` 或按 Ctrl+C 退出。
生成到 EOS 或达到新增 token 上限时停止；上下文超过限制时使用最近的 token。

调整生成参数或使用 CPU：

```bash
python -u inference.py --max-new-tokens 200 --temperature 0.8 --top-p 0.9
python -u inference.py --device cpu
```

可以用 `--checkpoint`、`--vocab`、`--merges` 指定其他文件路径。
显式传入的相对路径以当前终端目录为基准，默认路径以脚本目录为基准。

旧权重必须匹配训练当时的配置；如有不同，可以指定 `--vocab-size`、
`--context-length`、`--d-model`、`--num-layers`、`--num-heads`、
`--d-ff` 和 `--theta`。注意力头数和 RoPE 配置不一定能通过权重形状检查发现错误，
需要与训练时保持一致。词表和合并规则也必须使用训练时保存的同一套文件。

部署到服务器时同步 `inference.py` 和含有 `generate()` 的最新 `nn.py`，
并保留配套的分词器、checkpoint 和训练配置。
