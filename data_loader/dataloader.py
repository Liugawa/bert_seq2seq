# 摘要自动生成的例子
import os
import sys
import codecs
import torch 
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
import pandas as pd
import numpy as np
import random
import json
import time
from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from tokenizer import Tokenizer, load_chinese_base_vocab
else:
    from data_loader.tokenizer import Tokenizer, load_chinese_base_vocab
from utils.load_bert_params import *


def read_corpus(dir_path, vocab_path):
    """
    读原始数据
    """
    sents_src = []
    sents_tgt = []
    word2idx = load_chinese_base_vocab(vocab_path)
    tokenizer = Tokenizer(word2idx)
    finance_path = dir_path + "/财经"
    social_path = dir_path + "/社会"
    for base_path in [finance_path, social_path]:
        for txt in tqdm(os.listdir(base_path)):
            lines = codecs.open(base_path + "/" + txt, encoding="utf-8", mode="rb").readlines()
            title = lines[0].strip()
            if title is '':
                continue
            else:
                content = ""
                for line in lines[1:]:
                    content += line.replace('\u3000', ' ').replace('\xa0', ' ')

                sents_src.append(content.strip('\n'))
                sents_tgt.append(title)
                # print(2333)
    print("新闻共: " + str(len(sents_src)) + "篇")
    return sents_src, sents_tgt

class BertDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self, sents_src, sents_tgt, vocab_path):
        ## 一般init函数是加载所有数据
        super(BertDataset, self).__init__()
        # 读原始数据
        # self.sents_src, self.sents_tgt = read_corpus(poem_corpus_dir)
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt
        self.word2idx = load_chinese_base_vocab(vocab_path)
        self.idx2word = {k: v for v, k in self.word2idx.items()}
        self.tokenizer = Tokenizer(self.word2idx)

    def __getitem__(self, i):
        ## 得到单个数据
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]
        token_ids, token_type_ids = self.tokenizer.encode(src, tgt, max_length=256)
        output = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
        }
        return output

    def __len__(self):

        return len(self.sents_src)

def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.LongTensor(pad_indice)
    
    def torch_unilm_mask(s):
        "使用torch计算unilm mask"
        idxs = torch.cumsum(s, axis=1)
        mask = idxs[:, None, :] <= idxs[:, :, None]
        return mask.float()[:, None]


    token_ids = [data["token_ids"] for data in batch]
    max_length = 256#max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()

    token_type_ids_padded = padding(token_type_ids, max_length)
    attention_mask = torch_unilm_mask(token_type_ids_padded)

    return token_ids_padded, token_type_ids_padded, target_ids_padded, attention_mask



if __name__ == "__main__":
    data_dir = r"/home/ai/yangwei/text_summarization/THUCNews"
    vocab_path = r"/home/ai/yangwei/myResources/chinese_roberta_wwm_base_ext_pytorch/vocab.txt" # roberta模型字典的位置

    sents_src, sents_tgt = read_corpus(data_dir, vocab_path)
    dataset = BertDataset(sents_src, sents_tgt, vocab_path)
    # dataset.__getitem__(0)
    dataloader =  DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    for data in dataloader:
        print(data)
