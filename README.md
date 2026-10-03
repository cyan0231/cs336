# homework-1

记录 CS336 homework-1 的学习与实现过程：从 BPE 分词、数据预处理，到 Transformer 预训练和交互式文本续写。模型使用 TinyStories 英文故事数据训练，当前主要用于故事续写，尚未进行指令微调。

注：preprocess.py 
	inference.py
上述代码由ai生成

## 实现内容

- 字节级 BPE：训练词表与合并规则，支持特殊标记、编码和解码。
- Transformer：自定义 Linear、Embedding、RMSNorm、RoPE、因果多头自注意力与 SwiGLU。
- 训练：交叉熵损失、PyTorch AdamW、学习率 warmup 与余弦调度、梯度裁剪、断点续训。
- 数据处理：按完整文档编码，保存为 uint16 二进制 token 序列，通过 NumPy memmap 读取。
- 推理：交互式输入、Temperature 与 Top-p 采样、EOS 停止生成。

## 目录结构

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

## 环境准备

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

## 数据准备

从 [TinyStories 官方数据集文件页](https://huggingface.co/datasets/roneneldan/TinyStories/tree/main)下载以下两个文件，放在 homework-1 中，与 bpe.py 同级：

- [TinyStories-train.txt](https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt)
- [TinyStories-valid.txt](https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt)

当前代码读取的是上述文件，不是 TinyStoriesV2-GPT4 的文本版本。生成的 .bin 文件仍沿用 TinyStoriesV2-GPT4 前缀，这是代码中的历史命名，不代表使用了 V2-GPT4 数据。若要修改名称，需要同步修改 preprocess.py 与 train.py 中的路径。

## 从头训练

### 1. 训练分词器

```bash
python -u bpe.py
```

默认目标词表大小为 10,000，特殊标记为 <|endoftext|>。输出到 output/vocab.json 和 output/merges.txt；再次运行会覆盖这两个文件。

BPE 训练目前会一次性读取完整训练文本，需要足够的 CPU 内存。若已有模型权重，请保留其对应的词表和合并规则，不要用重新训练的分词器直接替换。

### 2. 预处理数据

```bash
python -u preprocess.py
```

加载分词器，将训练集和验证集转换为 uint16 token 序列：

- TinyStoriesV2-GPT4-train.bin
- TinyStoriesV2-GPT4-valid.bin

处理时会显示累计写入的 token 数量。脚本保留完整文档后再编码，避免任意字符切块改变分词结果。若目标 .bin 已存在，脚本会停止，避免覆盖；需要重新处理时，应先备份或自行处理旧文件。

### 3. 训练模型

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

## 交互式推理

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

### 实际续写示例

下面是一次输入 hello 后的输出节选；采样具有随机性，再次运行不保证得到相同文本。

```text
提示词 > hello
续写 > !" Lily said. "I'm sorry I scared you."
The man looked at her and said, "It's okay. You don't have to be scared. You can trust me."
Lily felt a little better. She said, "Thank you. I'm not scared anymore. I'm brave."
```

## 数据与模型文件管理

当前 .gitignore 忽略了 TinyStories 文本、.bin、整个 output/ 目录及模型权重扩展名。因此，克隆代码后需要自行生成这些产物，或取得匹配的分词器与权重；本 README 暂未提供预训练权重下载地址。

vocab.json 和 merges.txt 体积较小，适合随代码发布，但需要先调整忽略规则。checkpoint.pt 含优化器状态，当前本地文件约 260 MiB，若发布到 GitHub 应另行配置 Git LFS 或提供外部下载链接。数据集通过官方链接获取，二进制 token 文件由预处理脚本生成。

## 当前范围

本项目用于学习语言模型的基础实现，当前模型主要学习英文故事文本，尚未进行聊天指令微调或偏好训练。交互式入口提供单条提示词续写；通用问答、中文能力和多轮对话不属于当前已验证的能力。训练耗时、峰值显存与完整验证集指标尚未整理，后续可补充实验记录。
