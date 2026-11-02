# CS336 学习与实现记录

记录 CS336 作业的代码实现与实验结果，目前包含从头实现语言模型的 homework-1、基于 Qwen2.5-Math-1.5B 的 homework-5 SFT / GRPO 实验，以及基于普通 Qwen2.5-1.5B Base 的补充作业：SFT、DPO 与四项能力测评。

| 作业 | 内容 | 当前进度 |
| --- | --- | --- |
| [Homework 1](#homework-1基础语言模型) | BPE、Transformer、TinyStories 预训练与文本续写 | 已完成训练和交互式续写 |
| [Homework 5](#homework-5sft-与-grpo) | GSM8K 数据处理、SFT、GRPO 与 vLLM 评测 | 已完成单卡训练及完整测试集评测 |
| [Homework 5 补充](#homework-5-supplement) | 普通 Base 模型的 SFT、DPO，MMLU / GSM8K / AlpacaEval / Safety 测评 | 已完成 Base、SFT 和两版 DPO 实验；仓库收录 BF16 版 DPO 代码 |

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

作业五的学习与实现参考了 [chaser026/cs336-5](https://github.com/chaser026/cs336-5)。数学评分代码保留了原有的 Apache-2.0 许可头及 [understand-r1-zero](https://github.com/sail-sg/understand-r1-zero) 来源说明；当前实现使用固定常数归一化、不除组内标准差等配置，不等同于完整复现原始 GRPO 论文的所有设置。

<a id="homework-5-supplement"></a>

## Homework 5 补充：SFT、DPO 与四项测评

本部分使用普通的 [Qwen2.5-1.5B Base](https://huggingface.co/Qwen/Qwen2.5-1.5B)，先做全参数 SFT，再以 SFT 模型为起点进行离线 DPO。通过 MMLU、GSM8K、AlpacaEval 和 SimpleSafetyTests，观察知识问答、数学、有用性及安全性的变化。它与前一节使用 Math 模型和 GRPO 的实验分开记录。

**代码与实验版本：当前仓库中的 `dpo/train_dpo.py` 是实验室 RTX 3090 上的 BF16 参数版本。** 后来在 AutoDL 上完成了 FP32 参数更新、BF16 混合精度前向的对照实验，但对应修改后的脚本尚未收录。下文提供的 DPO 命令针对仓库现有 BF16 代码；不能只换一条命令就用它复现 FP32 实验。

### 目录与运行约定

```text
homework-5-supplement/
├── sft/
│   ├── preprocess_sft.py       # 套统一模板，过滤空白数据
│   ├── inspect_sft_lengths.py  # 分词长度统计，不截断
│   └── train_sft.py            # SFT训练、token loss测试、checkpoint恢复
├── dpo/
│   ├── inspect_dpo_lengths.py  # HH单轮筛选与偏好对长度统计
│   └── train_dpo.py            # 当前收录的BF16版DPO：smoke/train/eval
├── evaluation/
│   ├── evaluate_mmlu.py
│   ├── evaluate_gsm8k.py
│   ├── evaluate_alpaca.py      # 生成回答，不调用裁判
│   ├── evaluate_safety.py      # 生成回答，不调用裁判
│   ├── judge_alpaca_tool.py    # 官方AlpacaEval工具＋自定义DeepSeek裁判
│   ├── judge_safety.py         # DeepSeek安全性判断
│   └── deepseek_judge/
│       ├── configs.yaml
│       └── prompt.txt
├── model/Qwen2.5-1.5B/         # 自行下载的Base模型与tokenizer
├── data/                      # 自行准备，格式见下文
└── result/                    # 运行生成的模型、配置与逐题结果
```

`evaluation/test.py` 当前是空文件，不是测评入口。以下命令均在 `homework-5-supplement` 中执行；相对路径以当前工作目录为准：

```bash
cd cs336/homework-5-supplement
```

模型、原始数据、训练checkpoint和逐题输出未随本次代码提交提供；下面的数字来自服务器实验记录。当前仓库没有完整依赖锁定文件，也没有自动下载全部数据的统一脚本。

### 环境记录

| 用途 | 已使用的环境 |
| --- | --- |
| RTX 3090 24GB上的SFT、BF16版DPO及生成测评 | Python 3.11、PyTorch 2.5.1、Transformers 4.50.3、vLLM 0.7.2、pandas 2.2.3；Conda环境名 `336-a5` |
| AutoDL RTX 6000D上的FP32更新DPO | Python 3.12、PyTorch 2.8.0+cu128、Transformers 4.50.3、safetensors 0.5.3；GPU可用显存约83.05 GiB |
| AutoDL独立生成测评环境 `.venv-eval` | PyTorch 2.8.0+cu128、vLLM 0.10.2、Transformers 4.55.2、pandas 2.2.3、NumPy 2.2.6 |
| 独立API裁判环境 `.venv-judge` | alpaca-eval 0.6.6、openai 3.39.0、pandas 2.2.3；AutoDL另安装setuptools 79.0.1解决 `pkg_resources` 缺失 |

训练与vLLM依赖应分开管理，尤其不要在已配置好的训练环境中直接升级vLLM。AutoDL测评环境的建立方式如下，前提是基础环境已经安装可用的 PyTorch 2.8.0+cu128：

```bash
python -m venv --system-site-packages .venv-eval
.venv-eval/bin/python -m pip install \
  "vllm==0.10.2" "transformers==4.55.2" "pandas==2.2.3"
```

这个环境复用基础环境的PyTorch，将新增依赖安装在自己的目录；它不是可用于任意GPU的通用安装配方。实验室旧版vLLM与AutoDL新版vLLM的生成结果也不保证逐token一致。

### 数据来源与文件格式

| 用途 | 数据来源 | 本地文件及要求 |
| --- | --- | --- |
| SFT | [garg-aayush/sft-cs336-assign5-datasets](https://huggingface.co/datasets/garg-aayush/sft-cs336-assign5-datasets)，revision `ac710ef808346bf458d7b093bfe4586cb15699d9` | `data/sft-cs336-assign5-datasets/sft-instruct/{sample_train,train,test}.jsonl`；每行包含字符串 `prompt`、`response` |
| DPO | [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)，revision `09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa` | `data/hh-rlhf/<子集>/{train,test}.jsonl.gz`；每行包含完整对话字符串 `chosen`、`rejected` |
| MMLU | [hendrycks/test](https://github.com/hendrycks/test) | `data/mmlu/test/*_test.csv`；57个学科，无表头，每行依次为问题、A/B/C/D四个选项、标准答案，共14,042题 |
| GSM8K | [openai/grade-school-math](https://github.com/openai/grade-school-math) | `data/gsm8k/test.jsonl`；原始 `question`、`answer`，答案含 `####` 分隔符，共1,319题 |
| AlpacaEval | [tatsu-lab/alpaca_eval](https://huggingface.co/datasets/tatsu-lab/alpaca_eval) | `data/alpaca_eval/alpaca_eval.jsonl`；805条，每行保留 `instruction`、参考 `output`、参考 `generator`；本次参考回答来自 `text_davinci_003` |
| SimpleSafetyTests | [Bertievidgen/SimpleSafetyTests](https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests) | 原始 `sst_test_cases.csv` 保存为 `data/simple_safety_tests/simple_safety_tests.csv`；生成脚本要求列 `id`、`prompt`、`harm_area`，共100题，每类20题 |

SFT使用的数据是安全增广的UltraChat单轮指令数据。与原作业数学SFT不同，这里原始 `prompt` 是普通指令，预处理后才套入统一提示词。

HH四个子集为 `harmless-base`、`helpful-base`、`helpful-online`、`helpful-rejection-sampled`，分别覆盖无害性，以及不同采集阶段/方式的有用性偏好。它们不是同一批问题的四组回答；`chosen` 表示相对更受偏好，不保证绝对正确，`rejected` 也不是“拒绝回答”的意思。原数据可能包含多轮历史，本实验只保留单轮。

模型与训练数据可通过 `huggingface_hub.snapshot_download` 下载。HF官方站不可达时，实验中曾使用 `HF_ENDPOINT=https://hf-mirror.com`；该环境变量需要在导入下载库前设置。原始数据不必解压，DPO脚本直接使用 `gzip.open` 读取。

### 提示词与评分口径

SFT与四项生成测评使用相同的完整 `SYSTEM_PROMPT` 及外层格式，具体文本保存在预处理和测评脚本中：

```text
{SYSTEM_PROMPT}

### Instruction:
{问题或任务指令}

### Response:
```

MMLU在任务指令中附加选项与答案格式要求。标准答案、Alpaca参考回答不会放进被测模型的输入。

两版DPO训练使用参考仓库的简短模板，与上述完整系统提示词不同：

```text
Below is an instruction that describes a task.

### Instruction:
{问题}

### Response:
```

四个模型阶段的能力测评都保持完整统一模板；早期旧模板的结果不混入下面的主结果表。

| 测评 | 主要指标与评分方法 | 最多新增token / 总长度上限 |
| --- | --- | --- |
| MMLU | 优先提取 `The correct answer is A` 一类表达，否则取首个独立大写A–D；与标准选项比较 | 64 / 2048 |
| GSM8K | 从标准答案 `####` 后取数值；模型输出按当前正则处理数字间逗号后取最后一个数，转换为float精确比较 | 512 / 2048 |
| AlpacaEval | 自定义DeepSeek裁判比较模型回答与固定参考回答，使用官方工具的普通胜率 `get_winrate` | 1024 / 4096 |
| SimpleSafetyTests | 裁判仅输出True（不安全）或False（安全），统计判安全比例与类别比例 | 512 / 2048 |

生成共同设置为BF16、temperature=0、top-p=1、每题1条回答、seed=0、`stop=["### Instruction:"]`、正常EOS停止、`max_num_seqs=4`、显存比例0.7和eager模式。达到长度上限的回答保留并记录；MMLU/GSM8K解析失败按错误计入总题数，不为提高成绩临时修改解析规则。

AlpacaEval这里的胜率是对同一套参考回答的胜率，**不是DPO直接对战SFT的胜率，也不是官方排行榜或长度控制胜率**。Safety衡量裁判认为回答安全的比例，不等于有用性、正确拒答率或所有安全场景的通过率。

### SFT：预处理、训练与测试

先预处理，再统计分词长度：

```bash
python -u sft/preprocess_sft.py \
  --input_dir data/sft-cs336-assign5-datasets/sft-instruct \
  --output_dir data/sft_processed

python -u sft/inspect_sft_lengths.py \
  --model_path model/Qwen2.5-1.5B \
  --data_dir data/sft_processed \
  --output_file result/sft_length_stats.json
```

预处理逐行套模板，保留原始问题与回答文本；过滤空白字段，格式错误直接报错。处理后仍只有 `prompt`、`response` 两个字段，不分词、不截断、不追加EOS。默认不覆盖已有预处理或统计结果，重新生成需显式加 `--overwrite`。训练读取的是已套模板的prompt，不能再次套模板。

| 数据 | 原始条数 | 空白过滤后 | 总长度不超过2048、含EOS |
| --- | ---: | ---: | ---: |
| sample_train | 1,000 | 1,000 | 试跑时按同一长度规则过滤，未单独汇总数量 |
| train | 210,348 | 210,326 | 210,049 |
| test | 23,110 | 23,108 | 23,083 |

长度统计中，训练集完整序列平均637.5 token、P90为1,070、P99为1,454；只有277条超过2,048。因此采用超长整条过滤，不截断、不packing。

训练时先将prompt和response分别编码（`add_special_tokens=False`），显式追加真实EOS。`make_batch`在micro-batch内动态右填充并提前错位，返回输入、labels和有效label mask：

```text
完整序列： [P1, P2, R1, R2, EOS]
input_ids：[P1, P2, R1, R2]
labels：   [P2, R1, R2, EOS]
mask：     [ 0,  1,  1,   1]
```

先算全部位置的交叉熵，再乘mask；prompt和PAD不参与loss，回答与EOS参与。一次参数更新中，各micro-batch的loss sum除以该更新所有有效回答token总数，不简单平均各micro-batch的mean loss，也不再额外除以累积次数。

先用调试集冒烟测试：

```bash
python -u sft/train_sft.py \
  --mode train_eval \
  --data_path data/sft_processed/sample_train.jsonl \
  --output_dir result/sft_smoke_20 \
  --max_steps 20 \
  --save_every 20 \
  --eval_limit 10
```

正式训练的参数与运行方式：

```bash
python -u sft/train_sft.py \
  --mode train_eval \
  --model_path model/Qwen2.5-1.5B \
  --data_path data/sft_processed/train.jsonl \
  --eval_data_path data/sft_processed/test.jsonl \
  --output_dir result/sft_full_13129 \
  --max_steps 13129 \
  --max_seq_len 2048 \
  --micro_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --max_grad_norm 1.0 \
  --save_every 500 \
  --eval_limit 0 \
  --seed 42
```

使用BF16参数、SDPA、gradient checkpointing，关闭KV cache；AdamW的weight decay为0、betas为(0.9, 0.999)、eps为1e-8，3% warmup后cosine衰减。每500次更新保存完整checkpoint，保留最新两个周期checkpoint，结束另存 `final`。新训练要求输出目录不存在；支持通过 `--resume_from` 加载完整训练状态，但恢复功能尚未做实际中断验证。

**13,129次更新不等于完整遍历训练集一轮。** 当前SFT的 `get_batch` 每次独立使用 `random.sample` 抽16条，跨更新可能重复；共抽取210,064个样本位置，独立样本覆盖率的期望约63%。这与下面按epoch顺序遍历的DPO不同。

正式训练耗时67,694.20秒，约18小时48分钟；平均更新5.12秒，峰值已分配显存15.77 GiB、峰值保留显存22.87 GiB。结束后在23,083条合格test样本上计算teacher-forced回答token平均loss：共7,620,719个有效token，loss为 **1.112176**，耗时1,275.53秒。这个loss不是自由生成的正确率。

需要单独运行完整token loss测试时：

```bash
python -u sft/train_sft.py \
  --mode eval \
  --model_path result/sft_full_13129/final \
  --eval_data_path data/sft_processed/test.jsonl \
  --output_dir result/sft_test_eval \
  --eval_limit 0
```

### DPO：数据筛选与仓库现有训练代码

先统计四个子集的单轮数据和长度：

```bash
python -u dpo/inspect_dpo_lengths.py \
  --data_path data/hh-rlhf \
  --model_path result/sft_full_13129/final \
  --output_file result/dpo_length_stats.json
```

训练脚本自己读取原始压缩文件，不依赖外部 `convert_hh.py`。按 `Human:`、`Assistant:` 角色边界筛选单轮，检查两边问题一致，删除外层角色前缀；回答内部的普通空行不会被当作新一轮。四个子集的原始train全部用于训练，test只用于训练后的独立测试，不从train另切验证集。

| 数据 | 原始偏好对 | 单轮等条件筛选后 | 任一序列超过512时过滤整对后 |
| --- | ---: | ---: | ---: |
| train | 160,800 | 49,278 | 48,925 |
| test | 8,552 | 2,598 | 2,578 |

长度分别计算“简短模板＋问题＋chosen/rejected回答＋EOS”，以两条序列中的较大长度判断是否超限。训练对最大序列长度平均132.84、P99为441；512上限过滤353对，占单轮合格数据的0.72%。

每条样本提前编码为chosen/rejected各自的 `input_ids` 和 `response_mask`；micro-batch内右填充、构造错位labels。两个模型对数据里相同的两个回答计算概率，不现场采样新回答；reference是固定SFT副本，不是API裁判。

```text
chosen_logp   = 回答与EOS的token log probability之和
rejected_logp = 回答与EOS的token log probability之和

margin = beta × [(policy_chosen_logp − policy_rejected_logp)
                 − (reference_chosen_logp − reference_rejected_logp)]
loss = −logsigmoid(margin)
```

DPO对偏好对loss取平均，不按回答token数平均。仓库版本的micro-batch固定为1对，累积16对更新一次；每个epoch先打乱，再顺序遍历。正式训练1个epoch，共3,058次更新，最后13对也参与一次更新。

先试跑普通样本20次更新，再用最长16对做2次压力更新，两部分都执行真实参数更新，试跑权重不保存：

```bash
python -u dpo/train_dpo.py \
  --mode smoke \
  --model_path result/sft_full_13129/final \
  --reference_path result/sft_full_13129/final \
  --output_dir result/dpo_smoke \
  --max_seq_len 512 \
  --batch_size 16
```

正式训练：

```bash
python -u dpo/train_dpo.py \
  --mode train \
  --model_path result/sft_full_13129/final \
  --reference_path result/sft_full_13129/final \
  --data_path data/hh-rlhf \
  --output_dir result/dpo_full_epoch1 \
  --max_seq_len 512 \
  --batch_size 16 \
  --learning_rate 1e-6 \
  --beta 0.1 \
  --max_grad_norm 1.0 \
  --save_every 500 \
  --seed 42 \
  --limit 0
```

policy/reference均以BF16加载在同一张卡，仅policy开启梯度；policy开启gradient checkpointing。AdamW使用固定学习率1e-6、无warmup或衰减，weight decay为0、betas为(0.9, 0.999)、eps为1e-8。每500步保存模型和tokenizer，保留本次运行登记的最新两个周期checkpoint，结束另存 `final`；**DPO未保存优化器和取样状态，不支持完整断点续训**。请为每次实验使用新的输出目录。

HH偏好测试独立执行：

```bash
python -u dpo/train_dpo.py \
  --mode eval \
  --model_path result/dpo_full_epoch1/final \
  --reference_path result/sft_full_13129/final \
  --output_dir result/dpo_full_epoch1/test_eval \
  --max_seq_len 512 \
  --beta 0.1 \
  --limit 0
```

这里的偏好准确率统计 `margin > 0` 的比例，表示policy相对reference提高了chosen/rejected的概率比，不是回答正确率。policy与reference初始一致时margin为0，按严格大于判断会记为0%，loss为约0.69315；平局不表示回答全部错误。

### AutoDL上的FP32参数更新对照（脚本未收录）

为了检查小更新的数值精度，另在RTX 6000D上重新从同一个SFT起点训练1轮，使用相同数据、512长度上限、beta=0.1、固定学习率1e-6、有效batch=16和seed=42。policy与reference均以FP32存储，前向使用BF16 autocast；policy梯度和AdamW状态为FP32，优化器改为 `fused=True`，micro-batch为8、通常累积2次更新。

micro-batch=8的试跑普通部分稳定耗时1.13秒/更新，长样本部分2.28秒/更新，长样本峰值已分配显存44.98 GiB。这些是试跑统计，不是完整训练耗时。该实验的服务器输出目录为 `result/dpo_fp32_epoch1_mb8/final`；仓库现有DPO脚本没有 `--micro_batch_size` 参数，也没有这套FP32加载与autocast实现。

| 独立HH测试，2,578对 | BF16更新DPO | FP32更新DPO |
| --- | ---: | ---: |
| 平均DPO loss | 0.67945 | 0.60332 |
| 偏好准确率 | 56.59% | 67.26% |
| 平均隐式奖励差 | 0.03487 | 0.57561 |

BF16版正式训练耗时24,240.50秒，约6小时44分钟，平均更新7.92秒，峰值已分配显存18.49 GiB。对比SFT与该DPO最终checkpoint时，约98.69%的参数元素完全相同，只有1.31%发生变化，平均绝对变化约3.02e-7；各层变化比例普遍约1.2%～1.4%。

BF16参数直接更新时，小于可表示数间隔的更新可能被舍入，当前普通AdamW也没有另建FP32主权重。参数比较与FP32实验的HH改善支持这一解释，但最终未变不代表每一步都没有梯度或没有变化；两次还改变了GPU、micro-batch、优化器实现和reference计算精度，不能当作只控制精度这一变量的严格消融。

### 四项测评命令

以下以仓库BF16版DPO的输出为例。评测Base或SFT时，只替换模型路径和结果标识；FP32实验也使用同样的完整提示词和采样参数，但在AutoDL上通过 `.venv-eval/bin/python` 调用新版vLLM。

```bash
MODEL_PATH=result/dpo_full_epoch1/final
RUN_NAME=dpo_bf16
EVAL_PYTHON=python

# AutoDL上改为：EVAL_PYTHON=.venv-eval/bin/python
# 已有FP32实验模型时可改为：
# MODEL_PATH=result/dpo_fp32_epoch1_mb8/final
# RUN_NAME=dpo_fp32

"$EVAL_PYTHON" -u evaluation/evaluate_mmlu.py \
  --model_path "$MODEL_PATH" \
  --data_path data/mmlu/test \
  --output_file "result/mmlu_${RUN_NAME}_unified_full.json" \
  --limit 0

"$EVAL_PYTHON" -u evaluation/evaluate_gsm8k.py \
  --model_path "$MODEL_PATH" \
  --data_path data/gsm8k/test.jsonl \
  --output_file "result/gsm8k_${RUN_NAME}_unified_full.json" \
  --limit 0

"$EVAL_PYTHON" -u evaluation/evaluate_alpaca.py \
  --model_path "$MODEL_PATH" \
  --data_path data/alpaca_eval/alpaca_eval.jsonl \
  --output_file "result/alpaca_${RUN_NAME}_unified_full.json" \
  --generator "Qwen2.5-1.5B-${RUN_NAME}" \
  --limit 0

"$EVAL_PYTHON" -u evaluation/evaluate_safety.py \
  --model_path "$MODEL_PATH" \
  --data_path data/simple_safety_tests/simple_safety_tests.csv \
  --output_file "result/safety_${RUN_NAME}_unified_full.jsonl" \
  --limit 0
```

GPU命令顺序运行，不与训练同时启动。默认 `--limit 10` 用于冒烟测试，`--limit 0` 才是全量。Alpaca/Safety同时生成 `.config.json`，其中保存配置和生成诊断；Safety裁判会读取这个配套文件，迁移回答时需要一起复制。Alpaca/Safety生成器拒绝覆盖已有输出，重新运行需换文件名。MMLU/GSM8K保存逐题结果与解析、停止原因等统计。

### API裁判配置与运行

本次使用DeepSeek官方API `https://api.deepseek.com`、模型名 `deepseek-flash`，温度0、关闭思考。Alpaca裁判最多输出256 token，Safety最多10 token；评分均使用4路并发，每道失败题每次运行最多尝试3次。Alpaca外层使用独立进程，配置文件中的 `num_procs=1` 保持不变。

建立独立环境，API评分无需GPU：

```bash
python -m venv .venv-judge
.venv-judge/bin/python -m pip install \
  "alpaca-eval==0.6.6" "openai==3.39.0" \
  "pandas==2.2.3" "setuptools==79.0.1"

# 以下read语法用于服务器Bash；密钥不写入代码或命令历史。
read -rsp "请输入 DeepSeek API Key: " DEEPSEEK_API_KEY
echo
export DEEPSEEK_API_KEY
```

准备好同一批生成结果后，沿用上一段的 `RUN_NAME`：

```bash
.venv-judge/bin/python -u evaluation/judge_alpaca_tool.py \
  --model_outputs "result/alpaca_${RUN_NAME}_unified_full.json" \
  --data_path data/alpaca_eval/alpaca_eval.jsonl \
  --output_dir "result/alpaca_${RUN_NAME}_unified_judge_full" \
  --annotators_config evaluation/deepseek_judge \
  --max_workers 4 --max_attempts 3 --limit 0

.venv-judge/bin/python -u evaluation/judge_safety.py \
  --model_outputs "result/safety_${RUN_NAME}_unified_full.jsonl" \
  --output_dir "result/safety_${RUN_NAME}_unified_judge_full" \
  --max_workers 4 --max_attempts 3 --limit 0
```

Alpaca参考数据与模型回答按顺序核对instruction，不能随意重排。两个裁判支持读取已有评分进度；相同输入与规则可继续评分，输入或配置改变应使用新的输出目录。生成与评分分开执行，训练和测评脚本不会自动调用API。实际裁判返回的模型信息、失败重试等以评分目录记录为准；第三方API后端也可能随时间变化。

### 四项完整测评结果

下表均使用统一的完整测评模板。Base/SFT/BF16版DPO在实验室服务器生成，FP32版DPO在AutoDL生成；两边的vLLM、Transformers版本不同，不能保证生成过程逐token等价。所有Alpaca/Safety评分均完成全量样本，无缺失评分。

| 模型与训练方式                               | MMLU 准确率 | GSM8K 准确率 | Alpaca 普通胜率 | Safety 安全比例 |
| -------------------------------------------- | ----------- | ------------ | --------------- | --------------- |
| Base                                         | 51.97%      | 54.97%       | 31.06%          | 66.00%          |
| Base + SFT                                   | 56.45%      | 61.49%       | 38.01%          | 79.00%          |
| Base + SFT + DPO（BF16，长度512）            | **56.56%**  | 62.32%       | **38.88%**      | 80.00%          |
| Base + SFT + DPO（FP32，统一模板，长度1024） | 56.47%      | **64.37%**   | 32.52%*         | **88.00%**      |
| Base + DPO（FP32，统一模板，长度1024）       | 54.60%      | 57.92%       | 25.34%          | 85.00%          |

| Safety类别，每类20条 | Base | SFT | BF16更新DPO | FP32更新DPO |
| --- | ---: | ---: | ---: | ---: |
| Child Abuse | 45% | 50% | 55% | 45% |
| Illegal and Highly Regulated Items | 80% | 80% | 85% | 90% |
| Physical Harm and Violence | 70% | 100% | 100% | 90% |
| Scams and Fraud | 75% | 80% | 80% | 90% |
| Suicide, Self-Harm, and Eating Disorders | 60% | 85% | 80% | 80% |

### 观察与限制

SFT相对Base在四项汇总指标上均提高，是本组实验的主要收益。BF16版DPO在SFT基础上只有小幅变化；FP32版DPO将独立HH偏好准确率从56.59%提高到67.26%，但通用测评没有全面改善：MMLU和GSM8K仅小幅提高，Safety与SFT同为79%，AlpacaEval反而降到34.16%，比SFT低3.85个百分点。

这表明更充分地优化HH偏好目标，不必然提高其他指令分布上的回答质量。HH数据分布、只保留单轮的筛选方式、回答风格与长度、训练和测评提示词不同，都可能影响迁移；当前结果没有直接验证各因素的独立贡献，也不能仅凭Alpaca回退就断言过拟合。

全部结果来自单次训练/测评，没有多随机种子重复或配对显著性检验。Safety每5个百分点只对应同一类别中的1条样本；API裁判也可能有判断波动。MMLU/GSM8K解析规则是工程评分口径，不是对所有自由回答语义的严格判断。FP32实验还存在推理版本差异，因此小幅分数变化不能全部归因于参数精度。

**模型规模的可能影响（尚未验证）**：Qwen2.5-1.5B的容量可能限制偏好优化的收益，使模型更难同时兼顾知识、指令遵循与安全偏好。但本实验没有跨模型规模的对照，不能据此认定“小模型在SFT后已经没有提升空间”，也不能把模型规模当作性能回退的已确认原因。

后续采用完整统一训练模板、长度上限1024的FP32实验中，SFT→DPO的GSM8K从61.49%提高到64.37%，Safety从79%提高到88%，但Alpaca普通胜率从38.01%下降到32.52%（后者有效评分804/805条）。Base直接DPO的Alpaca胜率也从31.06%下降到25.34%。这些结果说明SFT后仍有部分指标的提升空间，同时生成质量可能退化；“SFT已经用完提升空间”无法解释两条路线共同出现的Alpaca下降。模型容量、偏好数据分布和训练目标的影响仍需分别验证。

本部分学习参考了 [chaser026/cs336-5](https://github.com/chaser026/cs336-5)。这里保留的是函数式的数据处理、训练和独立测评流程；数据筛选、SFT损失归一化、提示词、裁判模型及运行环境等与参考实现存在差异，不应视为其配置的完全复现。
