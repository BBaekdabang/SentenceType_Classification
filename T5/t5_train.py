import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
from transformers import AutoModel, AutoModelForSequenceClassification, T5ForConditionalGeneration,AutoTokenizer
import pandas as pd
import numpy as np
import os
import gc
import random
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import warnings
import copy
warnings.filterwarnings("ignore")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


tokenizer = AutoTokenizer.from_pretrained('paust/pko-t5-base')

label_list = ['사실형', '대화형', '추론형', '예측형']

label_to_id = {label_list[i]: i for i in range(len(label_list))}

def train():
    global label_list

    tokenizer = AutoTokenizer.from_pretrained('paust/pko-t5-base')
    
    train_data = get_dataset( train_data_path , tokenizer )
    dev_data = get_dataset( dev_data_path , tokenizer )

    train_dataloader = DataLoader(train_data, shuffle=True,
                                batch_size=32,
                                num_workers=8,
                                pin_memory=True,)
    
    dev_dataloader = DataLoader(dev_data, shuffle=True,
                                batch_size=32,
                                num_workers=8,
                                pin_memory=True,)
    
    
    
    print('loading model')
    model = T5_model(len(label_list), len(tokenizer))
    # model.load_state_dict(torch.load(pretrained_path))
    model.to(device)    
    print('end loading')
    
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    #optimizer
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=3e-6,
        eps=1e-8
    )

    #num_epochs
    epochs = 20

    max_grad_norm = 1.0
    total_steps = epochs * len(train_dataloader)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    print('[Training Phase]')
    #Start Epoch
    epoch_step = 0
    
    best_acc= -1
    
    for _ in tqdm(range(epochs), desc="Epoch"):
        epoch_step += 1

        print('\n')
        print('******************************')
        print('Epoch :', epoch_step)
        print('******************************')

        model.train()

        # train
        total_loss = 0

        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            e_input_ids, e_input_mask, d_input_ids,d_attention_mask= batch
            
            labels = d_input_ids
            labels = torch.where(labels!=0,labels,-100)        
        
            model.zero_grad()
            loss, logits = model(e_input_ids, e_input_mask, d_input_ids,d_attention_mask,labels)

            loss.backward()

            total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)

        print("Average train loss: {}".format(avg_train_loss))

        model_saved_path = ''
        torch.save(model.state_dict(), model_saved_path)
        
        if True:

            model.eval()

            pred_list = []
            labels = []
            for batch in dev_dataloader:
                batch = tuple(t.to(device) for t in batch)
                e_input_ids, e_input_mask, d_input_ids,d_attention_mask = batch

                with torch.no_grad():
                    predictions = model.test(e_input_ids, e_input_mask)
                    
                pred_list.extend(tokenizer.batch_decode(predictions, skip_special_tokens=True))
                labels.extend(tokenizer.batch_decode(d_input_ids, skip_special_tokens=True))
                
            data_num = 0
            correct = 0 
            pred_num = 0  

            for true,pred in zip(labels,pred_list):
                true_l = []
                pred_l = []
                for i in label_list:
                    if i in true:
                        true_l.append(i)
                data_num = data_num + len(true_l)
                for i in label_list:
                    if i in pred:
                        pred_l.append(i)
                pred_num = pred_num + len(pred_l)
                for i in true_l:
                    if i in pred_l:
                        correct = correct+1
                        
            print(correct,data_num,pred_num)
            
            print("dev acc = ",correct/data_num)
            if (correct/data_num) > best_acc:
                best_acc = correct/data_num
            
            precision = correct/pred_num
            Recall = correct/data_num
            f1 = 2*precision*Recall/(precision+Recall)
            
            print('F1 Score :', f1)        

    print("training is done")
