# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
from sre_parse import expand_template
import sys 
import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
from model import CodeBERT
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                              RobertaConfig, RobertaModel, RobertaTokenizer)
import random

import time

import sys 
sys.path.append("..") 
from gpu import select_gpu
select_gpu('tmp' + str(time.time()))

logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url

        
def convert_examples_to_features(js,tokenizer,args):
    # label

    """convert examples to token ids"""
    code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
    code_tokens = tokenizer.tokenize(code)[:args.code_length-2]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
    
    nl = ' '.join(js['docstring_tokens']) if 'docstring_tokens' in js and type(js['docstring_tokens']) is list else ' '.join(js['doc'].split()) if 'doc' in js else ' '.join(js['question'].split())
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-2]
    nl_tokens = [tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length      
    
    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, js['url'] if "url" in js else js["retrieval_idx"])



class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        data = []
        with open(file_path) as f:
            if "jsonl" in file_path:
                i = 0
                for line in f:
                    i += 1
                    # if i > 200: break
                    line = line.strip()
                    js = json.loads(line)
                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']
                    data.append(js)
            elif "codebase" in file_path or "code_idx_map" in file_path:
                js = json.load(f)
                for key in js:
                    temp = {}
                    temp['code_tokens'] = key.split()
                    temp["retrieval_idx"] = js[key]
                    temp['doc'] = ""
                    temp['docstring_tokens'] = ""
                    data.append(temp)
            elif "json" in file_path:
                i = 0
                for js in json.load(f):
                    i += 1
                    # if i > 200: break
                    data.append(js) 
                    
        
        for js in data:
            self.examples.append(convert_examples_to_features(js,tokenizer,args))
                
        if "train" in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))                       
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids))
            

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, tokenizer):
    """ Train the model """
    #get training dataset
    train_dataset = TextDataset(tokenizer, args, args.train_data_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * args.num_train_epochs)


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    
    tr_num,tr_loss,best_mrr = 0,0,0 
    for idx in range(args.num_train_epochs): 
        for step,batch in enumerate(train_dataloader):
            #get inputs
            code_inputs = batch[0].to(args.device)    
            nl_inputs = batch[1].to(args.device)
            #get code and nl vectors
            model.train()
            code_vec = model(code_inputs=code_inputs)
            nl_vec = model(nl_inputs=nl_inputs)
            
            #calculate scores and loss
            scores = torch.einsum("ab,cb->ac",nl_vec,code_vec)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(scores, torch.arange(code_inputs.size(0), device=scores.device))
            
            #report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step+1)%100 == 0:
                logger.info("epoch {} step {} lr {} loss {}".format(idx,step+1,round(scheduler.get_last_lr()[0],8), round(tr_loss/tr_num,5)))
                tr_loss = 0
                tr_num = 0
            
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
        #evaluate    
        eval_data_file, codebase_file = args.eval_data_file, args.codebase_file
        result = evaluate(args, model, tokenizer, eval_data_file, codebase_file)
        logger.info("***** Eval results on validation set *****")
        for key in sorted(result.keys()):
            logger.info("%s:  %s = %s", eval_data_file, key, str(round(result[key],3)))
        cur_mrr = result['eval_mrr']

        test_data_file, codebase_file = args.test_data_file, args.codebase_file
        if "AdvTest" in codebase_file: 
            codebase_file = test_data_file
        result = evaluate(args, model, tokenizer, test_data_file, codebase_file)
        logger.info("***** Eval results on test set *****")
        for key in sorted(result.keys()):
            logger.info("%s:  %s = %s", test_data_file, key, str(round(result[key],3)))
        
        #save best model
        if cur_mrr>best_mrr:
            best_mrr = cur_mrr
            logger.info("  "+"*"*20)  
            logger.info("  Best mrr:%s",round(best_mrr,4))
            logger.info("  "+"*"*20)                          

            model_to_save = model.module if hasattr(model,'module') else model
            model_best_state = {key: value.cpu() for key, value in model_to_save.state_dict().items()}
    
    return model_best_state


def evaluate(args, model, tokenizer,file_name,codebase_file):
    query_dataset = TextDataset(tokenizer, args, file_name)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=4)
    
    code_dataset = TextDataset(tokenizer, args, codebase_file)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size,num_workers=4)    
    
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    url_dict = {}
    for idx, example in enumerate(code_dataset.examples): 
        url_dict[example.url] = idx
    
    ground_truth = []
    for example in query_dataset.examples: 
        ground_truth.append(url_dict[example.url])

    model.eval()
    code_vecs = [] 
    nl_vecs = []
    for batch in query_dataloader:  
        nl_inputs = batch[1].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs) 
            nl_vecs.append(nl_vec.cpu()) 

    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)    
        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs)
            code_vecs.append(code_vec.cpu())  # .cpu()
    code_vecs = torch.cat(code_vecs)
    nl_vecs = torch.cat(nl_vecs)

    scores = torch.matmul(nl_vecs,code_vecs.T)
    
    sort_ids = torch.argsort(scores, dim = 1, descending = True)  
    rank_mat = torch.argsort(sort_ids, dim = 1)

    rank_code = rank_mat[list(range(len(ground_truth))), ground_truth] + 1

    inverse_rank_code = 1 / rank_code

    masked_inverse_rank_code = inverse_rank_code * (rank_code <= 1000)

    result = {
        "eval_mrr":float(torch.mean(masked_inverse_rank_code).item())
    }

    return result

                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default="dataset/CSN/python/train.jsonl", type=str, 
                        help="The input training data file (a json file).")
    parser.add_argument("--eval_data_file", default="dataset/CSN/python/valid.jsonl", type=str, 
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default="dataset/CSN/python/test.jsonl", type=str, 
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default="dataset/CSN/python/codebase.jsonl", type=str, 
                        help="An optional input test data file to codebase (a jsonl file).")  
    
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--output_dir", default="saved_model/unixcoder/CSN/python/", type=str, 
                        help="The saved model checkpoint.")

    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")  

    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")         
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--log", default="log/log.log", type=str,
                        help="log_file.")

    parser.add_argument('--seed', type=int, default=123456,
                        help="random seed for initialization")


    #print arguments
    args = parser.parse_args()
    #set logger
    os.makedirs('/'.join(args.log.split('/')[:-1]), exist_ok=True)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO, 
                    filename=args.log, filemode="w" )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)

    #build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained(args.model_name_or_path) 
 
    
    model = CodeBERT(model)
    logger.info("Training/evaluation parameters %s", args)
    
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)  
            
    # Training
    model_best_state = train(args, model, tokenizer)
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, 'model.bin')
    torch.save(model_best_state, output_file)
      
    # Evaluation
    model_to_load = model.module if hasattr(model, 'module') else model  
    model_to_load.load_state_dict(model_best_state)      
    eval_data_file, codebase_file = args.eval_data_file, args.codebase_file
    result = evaluate(args, model, tokenizer, eval_data_file, codebase_file)
    logger.info("***** Eval results on validation set *****")
    for key in sorted(result.keys()):
        logger.info("%s:  %s = %s", eval_data_file, key, str(round(result[key],3)))
            
    model_to_load = model.module if hasattr(model, 'module') else model  
    model_to_load.load_state_dict(model_best_state)      
    test_data_file, codebase_file = args.test_data_file, args.codebase_file
    if "AdvTest" in codebase_file: 
        codebase_file = test_data_file
    result = evaluate(args, model, tokenizer, test_data_file, codebase_file)
    logger.info("***** Eval results on test set *****")
    for key in sorted(result.keys()):
        logger.info("%s:  %s = %s", test_data_file, key, str(round(result[key],3)))


if __name__ == "__main__":
    main()

