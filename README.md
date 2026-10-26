# CS336 学习与实现记录

记录 CS336 作业的代码实现与实验结果，目前包含从头实现语言模型的 homework-1，以及基于 Qwen2.5-Math-1.5B 进行 SFT 和 GRPO 训练的 homework-5。

| 作业 | 内容 | 当前进度 |
| --- | --- | --- |
| [Homework 1](#homework-1基础语言模型) | BPE、Transformer、TinyStories 预训练与文本续写 | 已完成训练和交互式续写 |
| [Homework 5](#homework-5sft-与-grpo) | GSM8K 数据处理、SFT、GRPO 与 vLLM 评测 | 已完成单卡训练及完整测试集评测 |

数据集、基础模型和服务器生成的实验产物需要单独准备，克隆代码不会自动获得这些文件。

## Homework 1：基础语言模型

记录 CS336 homework-1 的学习与实现过程：从 BPE 分词、数据预处理，到 Transformer 预训练和交互式文本续写。模型使用 TinyStories 英文故事数据训练，当前主要用于故事续写，尚未进行指令微调。

实现说明：`preprocess.py` 和 `inference.py` 由 AI 生成。

### 实现内容

- 字节级 BPE：训练词表与合并规则，支持特殊标记、编码和解码。
- Transformer：自定义 Linear、Embedding、RMSNorm、RoPE、因果多头自注意力与 SwiGLU。
- 训练：交叉熵损失、PyTorch AdamW、学习率 warmup 与余弦调度、梯度裁剪、断点续训。
- 数据处理：按完整文档编码，保存为 uint16 二进制 token 序列，通过 NumPy memmap 读取。
- 推理：交互式输入、Temperature 与 Top-p 采样、EOS 停止生成。

### 目录结构

下面包含运行后生成的文件；数据和模型产物不一定随 Git 仓库提供。

```text
cs336/
├── README.md
└── homework-1/
    ├── bpe.py                       # 训练 BPE，保存词表与合并规则
    ├── tokenizer.py                 # BPE 编码与解码
    ├── preprocess.py                # 文本转 token 二进制文件
    ├── nn.py                        # Transformer 组件与生成方法
    ├── losses.py                    # 交叉熵损失
    ├── get_batch.py                 # 采样训练/验证批次
    ├── cos_schedule.py              # 学习率调度
    ├── clip_gradient_norm.py        # 梯度裁剪
    ├── checkpoint.py                # 保存与加载训练状态
    ├── train.py                     # 训练入口及超参数
    ├── inference.py                 # 交互式推理入口
    ├── TinyStories-train.txt        # 下载的训练文本
    ├── TinyStories-valid.txt        # 下载的验证文本
    ├── TinyStoriesV2-GPT4-train.bin  # 预处理生成
    ├── TinyStoriesV2-GPT4-valid.bin  # 预处理生成
    └── output/
        ├── vocab.json              # token ID 与字节表示的映射
        ├── merges.txt              # 有序 BPE 合并规则
        └── checkpoint.pt           # 模型、优化器状态与训练步数
```

### 环境准备

依赖为 Python、PyTorch、NumPy 和 regex。请先在当前 Python 环境中安装适合机器的 PyTorch；GPU 环境的安装命令以 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)为准。已提供 PyTorch 的云端镜像可以直接安装其余依赖：

```bash
python -m pip install numpy regex
```

项目目前没有锁定依赖版本，也没有提供跨版本兼容性测试结果。训练和推理优先使用 CUDA，不可用时回退到 CPU；完整训练建议使用 GPU。此前训练使用 RTX 3090 24GB。

克隆仓库并进入作业目录：

```bash
git clone https://github.com/cyan0231/cs336.git
cd cs336/homework-1
```

以下命令均在 homework-1 目录执行。

### 数据准备

从 [TinyStories 官方数据集文件页](https://huggingface.co/datasets/roneneldan/TinyStories/tree/main)下载以下两个文件，放在 homework-1 中，与 bpe.py 同级：

- [TinyStories-train.txt](https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt)
- [TinyStories-valid.txt](https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt)

当前代码读取的是上述文件，不是 TinyStoriesV2-GPT4 的文本版本。生成的 .bin 文件仍沿用 TinyStoriesV2-GPT4 前缀，这是代码中的历史命名，不代表使用了 V2-GPT4 数据。若要修改名称，需要同步修改 preprocess.py 与 train.py 中的路径。

### 从头训练

#### 1. 训练分词器

```bash
python -u bpe.py
```

默认目标词表大小为 10,000，特殊标记为 <|endoftext|>。输出到 output/vocab.json 和 output/merges.txt；再次运行会覆盖这两个文件。

BPE 训练目前会一次性读取完整训练文本，需要足够的 CPU 内存。若已有模型权重，请保留其对应的词表和合并规则，不要用重新训练的分词器直接替换。

#### 2. 预处理数据

```bash
python -u preprocess.py
```

加载分词器，将训练集和验证集转换为 uint16 token 序列：

- TinyStoriesV2-GPT4-train.bin
- TinyStoriesV2-GPT4-valid.bin

处理时会显示累计写入的 token 数量。脚本保留完整文档后再编码，避免任意字符切块改变分词结果。若目标 .bin 已存在，脚本会停止，避免覆盖；需要重新处理时，应先备份或自行处理旧文件。

#### 3. 训练模型

```bash
python -u train.py
```

训练参数在 train.py 顶部修改，当前默认配置如下：

| 参数 | 默认值 |
| --- | --- |
| 词表大小 | 10,000 |
| 上下文长度 | 256 tokens |
| 隐藏维度 d_model | 512 |
| Transformer 层数 | 4 |
| 注意力头数 | 16 |
| 前馈维度 d_ff | 1344 |
| Batch size | 64 |
| 总训练步数 | 7,000 |
| Warmup 步数 | 700 |
| 最大学习率 / 最小学习率 | 6e-4 / 6e-5 |
| RoPE theta | 10,000 |
| 梯度裁剪 max_norm | 1.0 |
| Weight decay | 0.1 |
| 优化器 | torch.optim.AdamW |

训练中每隔 100 步及最后一步打印训练损失和验证损失；验证损失来自一次随机采样的 batch，并非整个验证集的平均值。代码按迭代索引每隔 1,000 步及最后一步保存到 output/checkpoint.pt，后续保存会覆盖同一路径。

如果 checkpoint.pt 已存在，训练会自动恢复模型、优化器和训练步数，从下一步继续，直到总步数上限。要从头开始训练，应先备份并移走旧 checkpoint。修改模型结构后不能直接沿用旧权重。

### 交互式推理

推理需要同一次训练对应的三个文件：output/checkpoint.pt、output/vocab.json 和 output/merges.txt。仅推理时不需要原始数据或 .bin 文件。

```bash
python inference.py
```

输入英文提示词即可续写；输入 q、exit 或 quit 退出。每轮输入独立生成，不保留上一轮历史，界面仅显示新增文本。

自定义采样参数：

```bash
python inference.py --max-new-tokens 200 --temperature 0.8 --top-p 0.9
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| --max-new-tokens | 100 | 最多新增的 token 数，必须为正整数 |
| --temperature | 0.8 | 有限正数；较低通常使采样更集中，不支持设为 0 |
| --top-p | 0.9 | 核采样阈值，范围为 (0, 1]；1 表示不进行 Top-p 截断 |
| --device | auto | auto、cpu 或 cuda |
| --checkpoint | output/checkpoint.pt | 模型检查点路径 |
| --vocab | output/vocab.json | 词表路径 |
| --merges | output/merges.txt | 合并规则路径 |

默认文件路径相对于 inference.py 所在目录。显式传入的相对路径则相对于运行命令时的工作目录。

模型结构默认复用 train.py 的配置，也可通过 --vocab-size、--context-length、--d-model、--num-layers、--num-heads、--d-ff 和 --theta 指定训练时的值。加载已有权重时，这些配置和分词器必须与训练时一致。

输入和生成内容超过上下文长度时，每一步仅使用最近的 256 个 token（默认配置）预测下一个 token。当前没有 KV cache，每步会重新计算当前上下文。

#### 实际续写示例

下面是一次输入 hello 后的输出节选；采样具有随机性，再次运行不保证得到相同文本。

```text
提示词 > hello
续写 > !" Lily said. "I'm sorry I scared you."
The man looked at her and said, "It's okay. You don't have to be scared. You can trust me."
Lily felt a little better. She said, "Thank you. I'm not scared anymore. I'm brave."
```

### 数据与模型文件管理

当前 .gitignore 忽略了 TinyStories 文本、.bin、整个 output/ 目录及模型权重扩展名。因此，克隆代码后需要自行生成这些产物，或取得匹配的分词器与权重；本 README 暂未提供预训练权重下载地址。

vocab.json 和 merges.txt 体积较小，适合随代码发布，但需要先调整忽略规则。checkpoint.pt 含优化器状态，当前本地文件约 260 MiB，若发布到 GitHub 应另行配置 Git LFS 或提供外部下载链接。数据集通过官方链接获取，二进制 token 文件由预处理脚本生成。

### 当前范围

Homework 1 用于学习语言模型的基础实现，该模型主要学习英文故事文本，尚未进行聊天指令微调或偏好训练。交互式入口提供单条提示词续写；通用问答、中文能力和多轮对话不属于该模型当前已验证的能力。训练耗时、峰值显存与完整验证集指标尚未整理。

## Homework 5：SFT 与 GRPO

使用 Qwen2.5-Math-1.5B，在 GSM8K 上先进行监督微调，再通过模型生成回答、规则评分和组内优势进行 GRPO 训练。实验在 RTX 3090 24GB 单卡上完成。

### 实现内容与目录

- 数据转换：将 GSM8K 的 `question`、`answer` 转为 R1-Zero 风格的 `prompt`、`response` 和最终标准答案。
- SFT：错位构造输入与标签，只计算回答部分的交叉熵；回答 token 损失求和后按样本数平均，使用梯度累积和梯度裁剪。
- GRPO：vLLM 生成每题的多条回答；按组计算优势，固定旧概率，通过 GRPO-Clip 和固定常数归一化更新 Transformers 模型。
- 评测：使用 vLLM 生成，以 `r1_zero_reward_fn` 检查格式和最终答案，保存逐题结果。
- 保存与测试：训练在子进程中执行，完成并保存模型后退出，再由独立评测脚本加载模型，避免训练与最终测试同时占用 GPU。

```text
homework-5/
├── evaluation/
│   ├── convert_gsm8k.py       # 原始 JSONL 转 R1-Zero 风格数据
│   ├── drgrpo_grader.py       # 格式检查、答案提取与数学评分
│   └── evaluate_zero_shot.py  # 基础模型和微调模型共用的 vLLM 评测入口
├── sft/
│   └── train_sft_step.py      # SFT 训练、最终保存与自动测试
├── grpo/
│   ├── grpo_utils.py         # 分词、组内优势与逐 token Clip 损失
│   └── train_grpo.py         # 生成、评分、权重同步和训练主循环
├── data/gsm8k/               # 自行准备，不随代码提供
│   ├── train.jsonl
│   ├── test.jsonl
│   ├── train_r1_style.jsonl
│   └── test_r1_style.jsonl
├── model/Qwen2.5-Math-1.5B/  # 基础模型与分词器，自行准备
└── result/                  # 运行生成的模型和评测结果
    ├── checkpoints/         # SFT 输出
    └── grpo_checkpoints/    # GRPO 输出
```

### 环境与路径

本次服务器实验使用 Conda 环境 `336-a5`，Python 3.11、PyTorch 2.5.1+cu124、Transformers 4.50.3 和 vLLM 0.7.2。还需要 Accelerate，以及评分代码导入的 `sympy`、`latex2sympy2_extended`、`math_verify` 和 `pylatexenc` 等依赖；仓库尚未提供完整的环境锁定文件。

GRPO 使用了 vLLM V0 的内部权重同步接口和初始化 patch，不能直接假定任意新版 vLLM 都兼容。训练使用 BF16、梯度检查点，GRPO 的 Transformers 模型使用 SDPA；生成使用 vLLM。

运行前确认路径配置：

- `sft/train_sft_step.py` 的 `PROJECT_DIR` 当前写死为 `/home/hong/cyan/homework-5`；换机器时需要修改。
- GRPO、数据转换和评测脚本根据自身位置确定项目目录。
- `grpo/train_grpo.py` 的 `args.model_id` 默认指向此前的 SFT 检查点，需要替换为实际存在的目录；也可以改为基础模型目录进行对照实验。
- SFT 和 GRPO 的训练参数都在各自脚本顶部的 `SimpleNamespace` 中修改。

### 数据准备与评分方式

原始训练集有 7473 道题，测试集有 1319 道题。将每行包含 `question` 和 `answer` 的原始数据分别保存为 `data/gsm8k/train.jsonl` 和 `test.jsonl`；标准答案取自原始 `answer` 中 `####` 后面的内容。

SFT 需要处理后的训练数据。先把 `evaluation/convert_gsm8k.py` 顶部的路径改为：

```python
INPUT_FILE = PROJECT_DIR / "data/gsm8k/train.jsonl"
OUTPUT_FILE = PROJECT_DIR / "data/gsm8k/train_r1_style.jsonl"
```

然后执行：

```bash
python evaluation/convert_gsm8k.py
```

再把两个路径改为 `test.jsonl` 和 `test_r1_style.jsonl`，重复运行，生成评测数据。脚本当前默认转换测试集，没有切换数据集的命令行参数；同名输出文件会被覆盖。

`prompt` 已以 `Assistant: <think>` 结尾，模型续写推理过程和最终答案，例如：

```text
先计算 3 + 2 = 5。 </think> <answer>5</answer>
```

评分使用 `r1_zero_reward_fn`：先检查 `</think> <answer>` 和 `</answer>`，再提取最终答案并进行数学比较。格式和答案都通过时总奖励为 1，否则为 0。它不逐步检查推理过程，也不采用“提取全文最后一个数字”的方式。

数据转换输出中的 `is_correct`、`reward_details` 是标准数据的预设字段，不是模型评测结果。

### SFT 训练

```bash
python -u sft/train_sft_step.py
```

| 参数 | 当前默认值 |
| --- | --- |
| 初始模型 | Qwen2.5-Math-1.5B |
| 学习率 | 2e-5 |
| 参数更新次数 | 200 |
| 有效 batch size | 16 |
| Micro-batch size | 1 |
| 梯度累积次数 | 16 |
| 随机种子 | 42 |

训练完成后，模型、分词器与 `training_state.pt` 保存到 `result/checkpoints/<时间戳>_step_200/`。随后自动评测全部测试题，并保存 `test_results.jsonl` 和 `test_summary.json`。

虽然保存了优化器状态和随机数状态，当前 SFT 入口没有实现加载这些状态进行断点续训。

### GRPO 训练

GRPO 直接读取原始 `train.jsonl`，只取问题和最终标准答案；训练回答由当前模型现场生成，不使用数据集提供的推理过程。

```text
原始题库 -> 每轮随机抽 32 道题 -> 每题生成 8 条回答
    -> 256 条回答评分 -> 组内优势 -> 分词与错位标签
    -> 保存无梯度的 old_log_probs -> 当前模型前向与反向传播
    -> 梯度累积后更新参数 -> 同步给 vLLM -> 下一轮生成
```

```bash
python -u grpo/train_grpo.py
```

| 参数 | 当前本地默认值 |
| --- | --- |
| 初始模型 | 200 步 SFT 检查点 |
| Rollout 轮数 | 200 |
| 每轮题目数 / 每题回答数 | 32 / 8 |
| 每轮回答总数 | 256，由前两项相乘得到 |
| 每次参数更新的回答数 | 256 |
| 梯度累积次数 / Micro-batch size | 128 / 2 |
| 每批生成数据训练遍数 | 1 |
| 学习率 / AdamW weight decay | 1e-5 / 0 |
| 优势计算 | 奖励减去组均值，不除以标准差 |
| Clip 范围 | 0.8～1.2 |
| 回答损失归一化常数 | 1024 |
| 生成 temperature / top-p | 1.0 / 1.0 |
| 最多生成 / 最少生成 token 数 | 1024 / 4 |
| 总长度上限 | 2048 tokens |
| vLLM 显存比例 / 最大生成并发 | 0.2 / 4 |
| 训练与生成设备 | 均为 cuda:0 |
| 自动测试题数 | 10，`test_limit=0` 表示全部 |

单卡运行曾在 micro-batch 为 2 时出现 OOM，之后将 `gradient_accumulation_steps` 调为 256，使 micro-batch 降为 1；有效训练 batch 仍为 256。表中的 128 是当前本地代码默认值，复现实验时应核对实际运行目录中的 `args.json`。

每轮只训练一遍、更新一次参数，因此 200 轮 rollout 对应 200 次参数更新。组内全对或全错时，该题的优势全部为 0；只有组内奖励有差异的题提供非零优势信号。当前实现没有 Critic 或额外的 KL 正则项。

日志中的 `Reward` 是本轮更新前、随机抽取题目所生成回答的平均原始奖励，不是固定验证集准确率；`Loss` 可以接近零或为负，不能按 SFT 交叉熵的方式判断性能。

训练成功后输出到 `result/grpo_checkpoints/<时间戳>_step_200/`，包含模型、分词器、`args.json`、自动测试的 `test_results.jsonl` 和 `test_summary.json`。当前实现只在训练结束后保存模型，没有中途检查点、最佳模型选择、断点续训，也没有将训练日志和 rollout 样本写入文件。

### 完整测试集评测

SFT 与 GRPO 共用同一个评测入口；只需要更换 `--model_path`。`--limit 0` 评测全部题目，默认值 10 只取前 10 题。

以下命令评测本次完成的 GRPO 检查点，无需重新训练：

```bash
python -u evaluation/evaluate_zero_shot.py \
  --model_path result/grpo_checkpoints/20261019_232432_717475_step_200 \
  --data_path data/gsm8k/test_r1_style.jsonl \
  --output_file result/grpo_checkpoints/20261019_232432_717475_step_200/test_results_full.jsonl \
  --limit 0
```

评测设置为每题生成 1 条回答、temperature=1.0、top-p=1.0、最多新增 1024 tokens、总长度上限 2048、seed=0；遇到 `</answer>` 停止并保留结束标签。

独立评测脚本打印总体统计，并保存每题的 prompt、标准答案、生成文本和评分。同名结果文件会被覆盖；它不会更新训练结束时由父进程写出的 `test_summary.json`。因此上面命令的全量结果在 `test_results_full.jsonl` 中，原来的汇总文件仍可能只记录自动测试的 10 题。

### 已完成的实验结果

以下为服务器上已完成的单次实验记录，使用 GSM8K 全部 1319 道测试题和相同的评测脚本、生成设置及评分规则。

| 模型 | 完全正确 | 正确率（格式和答案均通过） | 格式错误 | 格式正确但答案错误 |
| --- | ---: | ---: | ---: | ---: |
| SFT，200 次更新 | 622 / 1319 | 47.16% | 12 | 685 |
| SFT + GRPO，200 轮 rollout / 200 次更新 | 1061 / 1319 | **80.44%** | 5 | 253 |

GRPO 相比 SFT 多答对 439 道题，提高约 33.28 个百分点。SFT 和 GRPO 使用同一训练集，测试使用独立的测试划分；上述数字是单次评测结果，不是多次实验的平均值，也不代表已经验证了其他数据集上的效果。


### 远程运行与参考实现

长时间训练建议在服务器的 `tmux` 会话内启动，例如 `tmux new -s cs336-grpo`，然后运行训练命令。按 `Ctrl+B` 再按 `D` 离开会话，之后用 `tmux attach -t cs336-grpo` 恢复。需要保存终端日志时，可将训练命令改为：

```bash
python -u grpo/train_grpo.py > grpo_train.log 2>&1
```

另一个 SSH 终端可执行 `watch -n 1 nvidia-smi` 查看显存与 GPU 使用率，或用 `tail -f grpo_train.log` 观察日志。训练、生成和首次优化器更新的显存占用不同，生成阶段能运行不代表训练阶段一定不会 OOM。

作业五的学习与实现参考了 [chaser026/cs336-5](https://github.com/chaser026/cs336-5)。数学评分代码保留了原有的 Apache-2.0 许可头及 [understand-r1-zero](https://github.com/sail-sg/understand-r1-zero) 来源说明；当前实现使用固定常数归一化、不除组内标准差等配置，不等同于完整复现原始 GRPO 论文的所有设置。
