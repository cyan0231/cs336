import json
import multiprocessing as mp
import os
import random
import subprocess
import sys

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_utils import (
    compute_group_normalized_rewards,
    compute_grpo_clip_loss,
    tokenize_prompt_and_output
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# 复用已有提示词和奖励函数
from evaluation.convert_gsm8k import SYSTEM_PROMPT_TEMPLATE
from evaluation.drgrpo_grader import r1_zero_reward_fn

args = SimpleNamespace(
    # 从之前训练好的 SFT 模型开始
    model_id=str(
        PROJECT_DIR
        / "result/checkpoints/20261014_162108_763867_step_200"
    ),

    # GRPO 训练读取原始 GSM8K 数据
    train_data_path=str(
        PROJECT_DIR / "data/gsm8k/train.jsonl"
    ),

    # 独立测试脚本读取处理后的数据
    test_data_path=str(
        PROJECT_DIR / "data/gsm8k/test_r1_style.jsonl"
    ),

    output_dir=str(
        PROJECT_DIR / "result/grpo_checkpoints"
    ),

    # 总共进行多少轮“生成回答 -> 训练”
    n_grpo_steps=200,

    # 每轮抽几道题，每道题生成几个回答
    n_prompts_per_rollout=32,
    group_size=8,

    # 每次参数更新使用多少条回答
    train_batch_size=256,

    # 每次参数更新累积多少个 micro-batch
    # micro-batch 大小 = 256 / 128 = 2
    gradient_accumulation_steps=128,

    # 每轮生成的数据训练几遍
    epochs_per_rollout_batch=1,

    lr=1e-5,

    # 生成参数
    sampling_temperature=1.0,
    sampling_max_tokens=1024,
    sampling_min_tokens=4,
    max_len=2048,

    # GRPO 参数
    advantage_eps=1e-6,
    use_std_normalization=False,
    cliprange=0.2,

    # 固定常数归一化
    normalize_constant=1024.0,

    # 单卡：训练和生成都使用 cuda:0
    device="cuda:0",
    vllm_device="cuda:0",
    vllm_gpu_util=0.2,

    # 单卡减少生成并发
    max_num_seqs=4,

    seed=42,

    # 先测试 10 题；改为 0 则测试全部
    test_limit=10,
)


def load_gsm8k_dataset(path, prompt_template):
    """
    原始 question、answer
        -> 带模板的 prompt、用于评分的 gold
    """
    processed_items = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            question = item["question"]
            full_solution = item["answer"]
            gold_answer = full_solution.split("####")[-1].strip()

            processed_items.append({
                "prompt": prompt_template.replace(
                    "{question}", question
                ),
                "gold": gold_answer,
            })

    return processed_items

def init_vllm(args):
    """
    沿用仓库的初始化方式。

    内部 patch 和后面的权重同步接口，
    对应你之前使用的 vLLM 0.7.2。
    """
    from vllm import LLM

    with (
        patch(
            "torch.distributed.get_world_size",
            return_value=1,
        ),
        patch(
            "vllm.worker.worker.Worker."
            "_assert_memory_footprint_increased_during_profiling",
            return_value=None,
        ),
    ):
        return LLM(
            model=args.model_id,
            device=args.vllm_device,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=args.vllm_gpu_util,
            max_model_len=args.max_len,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=True,

            # 每轮会同步新权重，避免复用旧权重的前缀缓存
            enable_prefix_caching=False,
            seed=args.seed,
        )


def load_policy_into_vllm_instance(policy, llm):
    """把训练模型的当前权重同步给 vLLM。"""

    state_dict = policy.state_dict()

    vllm_model = (
        llm.llm_engine.model_executor
        .driver_worker.model_runner.model
    )

    vllm_model.load_weights(state_dict.items())


def get_response_log_probs(policy, input_ids, labels):
    logit=policy(input_ids).logits.float()
    target_logit=torch.gather(logit,dim=-1,index=labels.unsqueeze(-1)).squeeze(-1) #batch step
    return target_logit-torch.logsumexp(logit,dim=-1)

def run_grpo_training(args, save_dir):
    from vllm import SamplingParams

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    assert args.gradient_accumulation_steps > 0
    assert args.train_batch_size > 0
    assert (
        args.train_batch_size
        % args.gradient_accumulation_steps == 0
    )
    
    assert args.n_prompts_per_rollout > 0
    assert args.group_size > 1

    # 每轮回答总数 = 题目数 × 每题回答数
    rollout_batch_size = (
        args.n_prompts_per_rollout * args.group_size
    )

    assert rollout_batch_size % args.train_batch_size == 0
    assert args.normalize_constant > 0

    

    micro_train_batch_size = (
        args.train_batch_size
        // args.gradient_accumulation_steps
    )

    save_dir = Path(save_dir)

    question_pool=load_gsm8k_dataset(
        args.train_data_path,
        SYSTEM_PROMPT_TEMPLATE,
    ) #加载原始数据
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    #过滤过长的题目
    filtered_questions = []
    for item in question_pool:
        prompt_ids = tokenizer.encode(
            item["prompt"],
            add_special_tokens=False,
        )
        total_length = len(prompt_ids) + args.sampling_max_tokens
        if total_length <= args.max_len:
            filtered_questions.append(item)

    question_pool = filtered_questions
    #加载模型
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(args.device)

    policy.config.use_cache = False

    policy.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        }
    )

    optimizer = AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=0.0,
    )
    #加载vllm 方便生成多个回答
    vllm_inst = init_vllm(args)

    sampling_params = SamplingParams(
        n=args.group_size,
        temperature=args.sampling_temperature,
        top_p=1.0,
        max_tokens=args.sampling_max_tokens,
        min_tokens=args.sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    global_step=0 #记录参数更新了几次
    for step in range(args.n_grpo_steps):
        ##去问题库里随机抽题，然后每道题生成多个回答
        policy.eval()
        load_policy_into_vllm_instance(policy, vllm_inst)
        sampled_questions=random.sample(question_pool,args.n_prompts_per_rollout)
        prompts=[item['prompt'] for item in sampled_questions]
        outputs=vllm_inst.generate(prompts,sampling_params)

        ##评分
        flat_prompts = []
        rollout_responses = []
        repeated_ground_truths = []
        #同一道题的group_size个回答连续排列
        for item,output in zip(sampled_questions,outputs):
            for candidate in output.outputs:
                flat_prompts.append(item['prompt'])
                rollout_responses.append(candidate.text)
                repeated_ground_truths.append(item['gold'])
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_response=rollout_responses,
            true_answer=repeated_ground_truths,
            group_size=args.group_size,
            eps=args.advantage_eps,
            normalized_by_std=args.use_std_normalization,
        )

        #把生成的回答变成训练张量
        train_data=tokenize_prompt_and_output(prompt_strs=flat_prompts,output_strs=rollout_responses,tokenizer=tokenizer)
        train_data["advantages"] = advantages.unsqueeze(-1) #n_prompts_per_rollout*group_size,1
        num_responses = len(rollout_responses)


        ##更新计算前旧概率
        old_log_probs_list=[]
        with torch.no_grad():
            for start in range(
                0, num_responses, micro_train_batch_size #这里也可以一条条来，和micro_batch没有必然关系
            ):
                end = start + micro_train_batch_size

                input_ids = train_data["input_ids"][start:end].to(
                    args.device
                )
                labels = train_data["labels"][start:end].to(
                    args.device
                )

                old_log_probs = get_response_log_probs(
                    policy,
                    input_ids,
                    labels,
                )

                # 旧概率存 CPU，训练时按小批量送回 GPU
                old_log_probs_list.append(old_log_probs.cpu())

                del input_ids, labels, old_log_probs

        train_data["old_log_probs"] = torch.cat(
            old_log_probs_list,
            dim=0,
        )
        del old_log_probs_list


        ##使用本轮回答更新模型

        policy.train()
        for epoch in range(args.epochs_per_rollout_batch):
            indices = torch.randperm(num_responses) #对于256道题 若是更新两次，就有区别了。目前是256题更新一次，所以打乱了还是一样
            for batch_start in range(
                0, num_responses, args.train_batch_size
            ):
                batch_indices = indices[
                    batch_start:
                    batch_start + args.train_batch_size
                ]
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
        ## 梯度累计
                for micro_start in range(
                    0,
                    args.train_batch_size,
                    micro_train_batch_size,
                ):
                    micro_indices = batch_indices[
                        micro_start:
                        micro_start + micro_train_batch_size
                    ]
                    batch={}
                    for key, value in train_data.items():
                        selected_data = value[micro_indices]
                        batch[key] = selected_data.to(args.device)
                    policy_log_probs = get_response_log_probs(
                        policy,
                        batch["input_ids"],
                        batch["labels"],
                    )
                    token_loss = compute_grpo_clip_loss(
                        advantage=batch["advantages"],
                        policy_log_probs=policy_log_probs,
                        old_log=batch["old_log_probs"],
                        cliprange=args.cliprange,
                    )
                    per_response_loss = (
                        (
                            token_loss * batch["response_mask"]
                        ).sum(dim=1)
                        / args.normalize_constant
                    )
                    loss = per_response_loss.mean()
                    scaled_loss = (
                        loss / args.gradient_accumulation_steps
                    )
                    scaled_loss.backward()

                    loss_sum += loss.item()
                    del (
                        batch,
                        policy_log_probs,
                        token_loss,
                        per_response_loss,
                        loss,
                        scaled_loss,
                    )

                # ---------- 参数更新 ----------
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    max_norm=1.0,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step+=1
                avg_loss = loss_sum / args.gradient_accumulation_steps

                print(
                    f"Rollout {step + 1}/{args.n_grpo_steps} | "
                    f"Epoch {epoch + 1} | "
                    f"Update {global_step} | "
                    f"Loss {avg_loss:.4f} | "
                    f"Reward {reward_metadata['mean_reward']:.4f}"
                )
        del train_data
        torch.cuda.empty_cache()
    # 8. 完整训练结束，保存模型与分词器
    # ==========================================
    policy.config.use_cache = True
    policy.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"训练完成，模型保存到：{save_dir}")


def run_test(args, save_dir):
    save_dir = Path(save_dir)
    output_file = save_dir / "test_results.jsonl"

    # 调用已有的独立 vLLM 测试脚本
    subprocess.run(
        [
            sys.executable,
            str(
                PROJECT_DIR
                / "evaluation/evaluate_zero_shot.py"
            ),
            "--model_path",
            str(save_dir),
            "--data_path",
            args.test_data_path,
            "--output_file",
            str(output_file),
            "--limit",
            str(args.test_limit),
        ],
        check=True,
        cwd=PROJECT_DIR,
    )

    summary = {
        "total": 0,
        "correct": 0,
        "format_error": 0,
        "correct_format_wrong_answer": 0,
    }

    with output_file.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            scores = json.loads(line)["scores"]

            summary["total"] += 1

            if scores.get("reward", 0) == 1:
                summary["correct"] += 1

            if scores.get("format_reward", 0) == 0:
                summary["format_error"] += 1
            elif scores.get("answer_reward", 0) == 0:
                summary["correct_format_wrong_answer"] += 1

    if summary["total"] == 0:
        raise ValueError("测试结果为空")

    summary["accuracy"] = (
        summary["correct"] / summary["total"]
    )

    with (save_dir / "test_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # 对应仓库使用的 vLLM V0 内部接口
    os.environ["VLLM_USE_V1"] = "0"

    # 提前检查路径
    if not Path(args.model_id).is_dir():
        raise FileNotFoundError(
            f"找不到模型目录：{args.model_id}"
        )

    for path in [
        args.train_data_path,
        args.test_data_path,
        PROJECT_DIR / "evaluation/evaluate_zero_shot.py",
    ]:
        if not Path(path).is_file():
            raise FileNotFoundError(f"找不到文件：{path}")

    # 每次运行使用新目录
    run_name = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    save_dir = (
        Path(args.output_dir)
        / f"{run_name}_step_{args.n_grpo_steps}"
    )
    save_dir.mkdir(parents=True, exist_ok=False)

    with (save_dir / "args.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            vars(args),
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 训练进程结束后释放 GPU，再运行独立测试
    context = mp.get_context("spawn")

    training_process = context.Process(
        target=run_grpo_training,
        args=(args, str(save_dir)),
    )

    training_process.start()

    try:
        training_process.join()
    except KeyboardInterrupt:
        training_process.terminate()
        training_process.join()
        raise

    if training_process.exitcode != 0:
        raise RuntimeError(
            "训练失败，未启动测试，请查看上方错误"
        )

    run_test(args, save_dir)
                    
               





