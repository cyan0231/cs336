import regex as re  # 使用 regex 而非内置 re，因为它支持 Unicode 类别（如 \p{L}）
from collections.abc import Iterable

class BPETokenizer:
    def __init__(self,vocab,merge,special_tokens):
        self.vocab=vocab
        self.merge=merge
        self.id_to_bytes=vocab
        self.bytes_to_id={v:k for k,v in vocab.items()}
        self.merge_rank={pair:i for i,pair in enumerate(merge)}
        self.special_token=special_tokens or []

        #构造特殊token的正则表达式
        if self.special_token:
            # 关键：必须按照长度从长到短排序（reverse=True）。
            # 这样正则引擎会优先匹配最长的特殊标记，防止重叠标记（如 <|a|><|b|>）被错误拆分。
            sorted_special = sorted(self.special_token, key=len, reverse=True)
            # 使用 re.escape 确保标记中的特殊字符（如 | 或 [ ）被当作普通字符处理
            special_pattern = "|".join(re.escape(t) for t in sorted_special)
            self.special_regex = re.compile(special_pattern)
        else:
            self.special_regex = None


        #构造gpt-2官方分词正则化表达式
        self.gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def encode(self,text):
        if not text:
            return []
        if not self.special_token:
            return self._encode_text_segment(text)
        last_pos=0
        tokens=[]
        for match in self.special_regex.finditer(text):
            pre_text=text[last_pos:match.start()]
            if pre_text:
                tokens.extend(self._encode_text_segment(pre_text))
            last_pos=match.end()
            special_text=text[match.start():match.end()]
            tokens.append(self.bytes_to_id[special_text.encode("utf-8")])
        if last_pos !=len(text):
            end_text=text[last_pos:]
            tokens.extend(self._encode_text_segment(end_text))
        return tokens
    def _encode_text_segment(self,text):
        pre_tokens = self.gpt2_pat.findall(text)
        ids=[]
        for word in pre_tokens:
            byte_parts=[bytes([b]) for b in word.encode('utf-8')]
            while len(byte_parts)>=2:
                min_rank=1e18
                best_part=None
                for pos in range(len(byte_parts)-1):
                    pair=(byte_parts[pos],byte_parts[pos+1])
                    if pair in self.merge_rank:
                        if(self.merge_rank[pair]<min_rank):
                            best_part=pair
                            min_rank=self.merge_rank[pair]
                if best_part is None:
                    break
                new_part=[]
                pos=0
                while pos<len(byte_parts):
                #for pos in range(len(byte_parts)-1):
                    if pos<len(byte_parts)-1 and ((byte_parts[pos],byte_parts[pos+1])==best_part):
                        new_part.append(best_part[0]+best_part[1])
                        pos+=2
                    else :
                        new_part.append(byte_parts[pos])
                        pos+=1

                byte_parts=new_part
            ids.extend(self.bytes_to_id[b] for b in byte_parts)
        return ids
    def decode(self,ids):
        ids=[self.id_to_bytes[idd] for idd in ids]
        full_bytes = b"".join(ids)
        return full_bytes.decode('utf-8',errors='replace')
    def encode_iterable(self,iterable):
        for chunk in iterable:
            yield from self.encode(chunk)




        

