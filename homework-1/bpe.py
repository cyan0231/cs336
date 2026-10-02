import os
from collections import defaultdict, Counter
import regex as re  # type: ignore
import json
def train_bpe(input_path,special_token,vocab_size):
    vocab={i:bytes([i]) for i in range(256)}
    num_merge=vocab_size-256-len(special_token)
    with open(input_path,'r',encoding='utf-8') as f:
        text=f.read()
    if special_token:
        special_regex = "|".join(re.escape(t) for t in special_token)
        parts=re.split(f"({special_regex})",text)
        train_segment=[p for p in parts if p not in special_token]
    else:
        train_segment=[text]
    gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    raw_count=Counter()
    for segment in train_segment:
        words=gpt2_pat.findall(segment)
        for word in words:
            raw_count[tuple(bytes([b]) for b in word.encode('utf-8'))]+=1 #raw_count 存tuple音节的个数，字典
    word_list=[]# 存有哪些音节list
    count_list=[]# 存对应的出现次数
    for k,v in raw_count.items():
        word_list.append(list(k))
        count_list.append(v)
    state=defaultdict(int) #存pair对的出现次数 字典
    indices=defaultdict(set) #存pair对出现的下标 字典
    for i in range(len(word_list)):
        freq=count_list[i]
        for j in range(len(word_list[i])-1):
            p1=word_list[i][j]
            p2=word_list[i][j+1]
            pair=(p1,p2)
            state[pair]+=freq
            indices[pair].add(i)
    merge=[]
    for i in range(num_merge):
        if not state:
            break
        best_pair=max(state.items(),key=lambda x:(x[1],x[0]))[0]
        if state[best_pair]<=0:
            break
        merge.append(best_pair)
        new_token=best_pair[0]+best_pair[1]
        relevant_list=list(indices[best_pair])
        for pos in relevant_list:
            word=word_list[pos]
            freq=count_list[pos]
            j=0
            while j<len(word)-1:
            #for j in range(len(word)-1):
                if (word[j],word[j+1])==best_pair:
                    if j>0:
                        pre_pair=(word[j-1],word[j])
                        state[pre_pair]-=freq
                        if state[pre_pair]==0:
                            del state[pre_pair]
                    if j<len(word)-2:
                        suf_pair=(word[j+1],word[j+2])
                        state[suf_pair]-=freq
                        if state[suf_pair]==0:
                            del state[suf_pair]
                    word[j]=new_token
                    del word[j+1]
                    if j>0:
                        new_pair=(word[j-1],new_token)
                        state[new_pair]+=freq
                        indices[new_pair].add(pos)
                    if j<len(word)-1:
                        new_pair=(new_token,word[j+1])
                        state[new_pair]+=freq
                        indices[new_pair].add(pos)
                else:
                    j+=1
        if best_pair in state: del state[best_pair]
        if best_pair in indices: del indices[best_pair]
    for pair in merge:
        new_id=len(vocab)
        vocab[new_id]=pair[0]+pair[1]
    for s_tok in special_token:
        vocab[len(vocab)]=s_tok.encode('utf-8')
    return vocab,merge
def bytes_to_unicode():
    """
    创建一个映射，将 0-255 字节映射为一组可见的 Unicode 字符。
    这是 GPT-2 源码中的标准做法。
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def save_tokenizer_files(vocab, merges, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 初始化映射表
    byte_encoder = bytes_to_unicode()

    # 词表保存
    # 使用 byte_encoder 将 bytes 转换为可见字符串
    json_vocab = {
        k: "".join(byte_encoder[b] for b in v) 
        for k, v in vocab.items()
    }
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(json_vocab, f, indent=4)
    
    # 合并规则保存
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            # 同样转换 p1 和 p2
            s1 = "".join(byte_encoder[b] for b in p1)
            s2 = "".join(byte_encoder[b] for b in p2)
            f.write(f"{s1} {s2}\n")
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "TinyStoriesV2-GPT4-train.txt")
    vocab_size=10000
    special_tokens = ["<|endoftext|>"]
    output_dir = os.path.join(base_dir, "output")
    vocab,merge=train_bpe(input_path,special_tokens,vocab_size)
    save_tokenizer_files(vocab,merge,output_dir)

if __name__=='__main__':
    main()
            
