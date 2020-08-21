# 标题自动生成的例子
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
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
from data_loader.tokenizer import Tokenizer, load_chinese_base_vocab
from utils.load_bert_params import load_bert, load_model_params, load_recent_model
from data_loader.dataloader import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class TitleModelTrainer:
    # 标题生成模型：控制训练过程，保存模型，等
    def __init__(self):
        # 加载数据
        data_dir = r"/home/ai/yangwei/text_summarization/THUCNews"
        self.vocab_path = r"/home/ai/yangwei/myResources/chinese_roberta_wwm_base_ext_pytorch/vocab.txt" # roberta模型字典的位置
        self.sents_src, self.sents_tgt = read_corpus(data_dir, self.vocab_path)
        self.model_name = "roberta" # 选择模型名字
        self.model_path = r"/home/ai/yangwei/myResources/chinese_roberta_wwm_base_ext_pytorch/pytorch_model.bin" # roberta模型位置
        self.model_save_path = "training_files/best_model.bin"
        self.batch_size = 16
        self.lr = 2e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #################################
        # 定义模型
        #################################
        self.bert_model = load_bert(self.vocab_path, model_name=self.model_name)

        #################################
        # 加载预训练的模型参数～
        #################################
        checkpoint = load_model_params(self.bert_model, self.model_path)
        # 将模型发送到计算设备(GPU或CPU)
        self.bert_model.to(self.device)
        # 声明需要优化的参数
        self.optim_parameters = list(self.bert_model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=self.lr, weight_decay=1e-3)
        # 声明自定义的数据加载器
        dataset = BertDataset(self.sents_src, self.sents_tgt, self.vocab_path)
        # dataset.__getitem__(0)
        self.dataloader =  DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

    def train(self, epoch):
        # 一个epoch的训练
        self.bert_model.train()
        self.iteration(epoch, dataloader=self.dataloader, train=True)
    
    def save(self, save_path):
        """
        保存模型
        """
        torch.save(self.bert_model.state_dict(), save_path)
        print("{} saved!".format(save_path))

    def iteration(self, epoch, dataloader, train=True):
        total_loss = []
        start_time = time.time() ## 得到当前时间
        step = 0
        for token_ids, token_type_ids, target_ids, att_mask in dataloader:
            if step % 300 == 0:
                self.bert_model.eval()
                idx = random.randint(0, len(self.sents_src)-1)
                text = self.sents_src[idx]
                title = self.sents_tgt[idx]
                print("Generated Title: ", self.bert_model.generate(text, beam_size=3, device=self.device, is_poem=False))
                print("Real      Title: ", title)
                self.bert_model.train()

            token_ids = token_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            att_mask = att_mask.to(self.device)
            target_ids = target_ids.to(self.device)
            # 因为传入了target标签，因此会计算loss并且返回
            predictions, loss = self.bert_model(token_ids,
                                                token_type_ids,
                                                att_mask,
                                                labels=target_ids,
                                                device=self.device
                                                )
            if step % 50 == 0:
                print("epoch: {}, step: {}, loss: {:.4f}".format(epoch, step, loss.item()))
            # 反向传播
            if train:
                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播, 获取新的梯度
                loss.backward()
                # 用获取的梯度更新模型参数
                self.optimizer.step()

            # 为计算当前epoch的平均loss
            total_loss.append(loss.item())
            step += 1

        end_time = time.time()
        spend_time = end_time - start_time
        # 打印训练信息
        print("epoch is " + str(epoch)+"; loss is " + str(np.mean(total_loss)) + "; spend time is "+ str(spend_time))
        # 保存模型
        self.save(self.model_save_path)

if __name__ == '__main__':

    trainer = TitleModelTrainer()
    train_epoches = 50
    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)

    # nohup python -u main.py > training_files/train-8-20.log 2>&1 &
