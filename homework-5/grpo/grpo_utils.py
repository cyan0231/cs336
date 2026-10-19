import torch
from typing import Callable, Dict, List, Tuple
from transformers import PreTrainedTokenizer
def compute_group_normalized_rewards(
        reward_fn, #传入回答，真实答案，返回字典
        rollout_response, #产生的批量回答
        true_answer,
        group_size,
        eps,
        normalized_by_std=True #优势是否除以std
):
    raw_reward_list=[]
    for response,answer in zip(rollout_response,true_answer):
        scores=reward_fn(response,answer)
        raw_reward_list.append(scores["reward"])
    raw_reward_list=torch.tensor(raw_reward_list,dtype=torch.float32)
    grouped_rewards=raw_reward_list.view(-1,group_size)
    group_means=grouped_rewards.mean(dim=1,keepdim=True)
    advantage=grouped_rewards-group_means
    if normalized_by_std is True:
        group_std=grouped_rewards.std(dim=1,keepdim=True)
        advantage=advantage/(group_std+eps)
    advantage=advantage.reshape(-1) #变回回答总数 
    metadata = {
        "mean_reward": raw_reward_list.mean().item(),
        "max_reward": raw_reward_list.max().item(),
        "min_reward": raw_reward_list.min().item(),
        "mean_advantage": advantage.mean().item(),
    }
    return advantage,raw_reward_list,metadata



def compute_grpo_clip_loss(
        advantage, #b,1
        policy_log_probs,#b L
        old_log,
        cliprange
):
    advantage=advantage.detach()
    old_log=old_log.detach()
    ratio=torch.exp(policy_log_probs-old_log)
    surr1=ratio*advantage
    surr2=torch.clamp(ratio,1.0-cliprange,1.0+cliprange)*advantage
    loss=-torch.min(surr1,surr2)
    return loss

def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, torch.Tensor]:
    if not prompt_strs or len(prompt_strs) != len(output_strs):
        raise ValueError("数据不能为空，且 prompt 和 response 数量必须一致")

    all_input_ids = []
    all_mask = []
    all_length = []

    for p_str, o_str in zip(prompt_strs, output_strs):
        p_ids = tokenizer.encode(p_str, add_special_tokens=False)
        o_ids = tokenizer.encode(o_str, add_special_tokens=False)

        input_ids = p_ids + o_ids
        all_input_ids.append(input_ids)
        all_length.append(len(input_ids))

        mask = [0] * len(p_ids) + [1] * len(o_ids)
        all_mask.append(mask)

    max_len = max(all_length)
    batch_size = len(prompt_strs)
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_id is None:
        raise ValueError("分词器没有 pad_token_id 或 eos_token_id")

    padded_input_ids = torch.full(
        (batch_size, max_len), pad_id, dtype=torch.long
    )
    padded_mask = torch.zeros(
        (batch_size, max_len), dtype=torch.long
    )

    for i, (ids, m) in enumerate(zip(all_input_ids, all_mask)):
        length = all_length[i]
        padded_input_ids[i, :length] = torch.tensor(ids)
        padded_mask[i, :length] = torch.tensor(m)

    return {
        "input_ids": padded_input_ids[:, :-1].clone(),
        "labels": padded_input_ids[:, 1:].clone(),
        "response_mask": padded_mask[:, 1:].clone(),
    }


