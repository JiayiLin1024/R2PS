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
from model import Unixcoder, Unixcoder_RR
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                              RobertaConfig, RobertaModel, RobertaTokenizer)
import random

import time
from gpu import gpu_memory_keeper

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# gpu_memory_keeper('tmp' + str(time.time()))

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
    code_tokens = tokenizer.tokenize(code)[:args.code_length-4]
    code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
    
    nl = ' '.join(js['docstring_tokens']) if 'docstring_tokens' in js and type(js['docstring_tokens']) is list else ' '.join(js['doc'].split()) if 'doc' in js else ' '.join(js['question'].split())
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-4]
    nl_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
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
                    # if i > 1000: break
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
                    
        
        # print(len(data))
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

class TextDataset_Train(Dataset):
    def __init__(self, tokenizer, args, sia_model, num_negs, file_path=None): 
        self.examples = []
        data = []
        with open(file_path) as f:
            if "jsonl" in file_path:
                i = 0
                for line in f:
                    i += 1
                    # if i > 1000: break
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
                    
        
        # print(len(data))
        for js in data:
            self.examples.append(convert_examples_to_features(js,tokenizer,args))


        sia_model.eval()
        self.code_vecs = []
        for i in range(0, len(self.examples), args.eval_batch_size): 
            code_inputs = torch.tensor([example.code_ids for example in self.examples[i: i + args.eval_batch_size]]).to(args.device) 
            with torch.no_grad():
                code_vec = sia_model(code_inputs=code_inputs)
                self.code_vecs.append(code_vec.cpu()) 
        self.code_vecs = torch.cat(self.code_vecs)


        self.nl_vecs = []
        for i in range(0, len(self.examples), args.eval_batch_size): 
            nl_inputs = torch.tensor([example.nl_ids for example in self.examples[i: i + args.eval_batch_size]]).to(args.device) 
            with torch.no_grad():
                nl_vec = sia_model(nl_inputs=nl_inputs)
                self.nl_vecs.append(nl_vec.cpu()) 
        self.nl_vecs = torch.cat(self.nl_vecs)

        self.num_negs = num_negs
        self.neg_begin = int(self.code_vecs.shape[0] * args.lam) + 1
        self.neg_begin = int(self.code_vecs.shape[0] * args.lam) # StaQC
        # self.hard_neg = int(num_negs * args.lam)
        # self.random_neg = num_negs - int(num_negs * args.lam)

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
        nl_vec = self.nl_vecs[i].unsqueeze(dim=0)
        scores = torch.matmul(nl_vec, self.code_vecs.T)
        sort_ids = torch.argsort(scores, dim = 1, descending = True)
        neg_ids_hard = sort_ids[:, self.neg_begin:self.num_negs+self.neg_begin].flatten()
        # neg_ids_random = torch.randperm(self.code_vecs.shape[0])[:self.random_neg]
        code_ids = torch.cat([torch.tensor([i]), neg_ids_hard])
        nl_ids = torch.tensor([i] * (self.num_negs + 1))
        return torch.tensor([self.examples[j].code_ids for j in code_ids]), torch.tensor([self.examples[j].nl_ids for j in nl_ids])

        # return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids))


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, mono_model, sia_model, tokenizer):
    """ Train the model """
    #get training dataset
    train_dataset = TextDataset_Train(tokenizer, args, sia_model, args.num_negs, args.train_data_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(mono_model.parameters(), lr=args.learning_rate, eps=1e-8)
    # num_warmup_steps = 0.1 * len(train_dataloader) * args.num_train_epochs, 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps = len(train_dataloader) * args.num_train_epochs)


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    mono_model.zero_grad()
    tr_num,tr_loss,best_mrr = 0,0,0 

    scaler = torch.cuda.amp.GradScaler()
    for idx in range(args.num_train_epochs): 
        # if idx >= 1: exit()
        for step,batch in enumerate(train_dataloader): 
            if idx >= 0: 
                # get inputs
                optimizer.zero_grad()
                code_inputs = batch[0].to(args.device)
                nl_inputs = batch[1].to(args.device)
                
                code_nl = torch.cat([nl_inputs.reshape([-1, nl_inputs.shape[-1]]), code_inputs.reshape([-1, code_inputs.shape[-1]])],dim=-1)

                mono_model.train()
                with torch.autocast(device_type='cuda', dtype=torch.float16):  
                    code_nl_score = mono_model(code_nl)
                    
                    #calculate scores and loss
                    scores = torch.reshape(code_nl_score, [nl_inputs.shape[0], args.num_negs + 1])
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(scores * args.temperature, torch.zeros(nl_inputs.size(0), dtype=torch.int64, device=scores.device))
                scaler.scale(loss).backward()

                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)
                
                #backward
                torch.nn.utils.clip_grad_norm_(mono_model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # report loss
                tr_loss += loss.item()
                tr_num += 1
                if (step+1)%100 == 0:
                    logger.info("epoch {} step {} lr {} loss {}".format(idx,step+1,round(scheduler.get_last_lr()[0],8), round(tr_loss/tr_num,5)))
                    tr_loss = 0
                    tr_num = 0

            else: 
                # get inputs
                code_inputs = batch[0].to(args.device)
                nl_inputs = batch[1].to(args.device)
                
                code_nl = torch.cat([nl_inputs.reshape([-1, nl_inputs.shape[-1]]), code_inputs.reshape([-1, code_inputs.shape[-1]])],dim=-1)

                mono_model.train()
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):  
                    code_nl_score = mono_model(code_nl)
                    #calculate scores and loss
                    scores = torch.reshape(code_nl_score, [nl_inputs.shape[0], args.num_negs + 1])
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(scores * args.temperature, torch.zeros(nl_inputs.size(0), dtype=torch.int64, device=scores.device))
                
                #report loss
                tr_loss += loss.item()
                tr_num += 1
                if (step+1)%100 == 0:
                    logger.info("epoch {} step {} lr {} loss {}".format(idx,step+1,round(scheduler.get_last_lr()[0],8), round(tr_loss/tr_num,5)))
                    tr_loss = 0
                    tr_num = 0
                
                #backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mono_model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
        #evaluate    
        eval_data_file, codebase_file = args.eval_data_file, args.codebase_file
        result = evaluate(args, mono_model, sia_model, tokenizer, eval_data_file, codebase_file)
        logger.info("***** Eval results on validation set *****")
        for key in sorted(result.keys()):
            logger.info("%s:  %s = %s", eval_data_file, key, str(round(result[key],3)))
        cur_mrr = result['eval_mrr']

        test_data_file, codebase_file = args.test_data_file, args.codebase_file
        if "AdvTest" in codebase_file: 
            codebase_file = test_data_file
        result = evaluate(args, mono_model, sia_model, tokenizer, test_data_file, codebase_file)
        logger.info("***** Eval results on test set *****")
        for key in sorted(result.keys()):
            logger.info("%s:  %s = %s", test_data_file, key, str(round(result[key],3)))
        
        #save best model
        if cur_mrr > best_mrr:
            best_mrr = cur_mrr
            logger.info("  "+"*"*20)  
            logger.info("  Best mrr:%s",round(best_mrr,4))
            logger.info("  "+"*"*20)                          

            model_to_save = mono_model.module if hasattr(mono_model,'module') else mono_model
            model_best_state = {key: value.cpu() for key, value in model_to_save.state_dict().items()}
            
            # logger.info("  "+"*"*20)  
            # logger.info("  Number of layers:%s",len(model_best_state))
            # logger.info("  "+"*"*20)  
    
    return model_best_state



def evaluate(args, mono_model, sia_model, tokenizer, file_name,codebase_file):
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

    mono_model.eval()
    sia_model.eval()

    code_vecs = [] 
    nl_vecs = []
    for batch in query_dataloader:  
        nl_inputs = batch[1].to(args.device)
        with torch.no_grad():
            nl_vec = sia_model(nl_inputs=nl_inputs) 
            nl_vecs.append(nl_vec.cpu()) 

    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)    
        with torch.no_grad():
            code_vec = sia_model(code_inputs=code_inputs)
            code_vecs.append(code_vec.cpu()) 
    code_vecs = torch.cat(code_vecs)
    nl_vecs = torch.cat(nl_vecs)
    scores = torch.matmul(nl_vecs,code_vecs.T)
    sort_ids_ = torch.argsort(scores, dim = 1, descending = True)  
    rank_mat_ = torch.argsort(sort_ids_, dim = 1)

    rank_code_ = rank_mat_[list(range(len(ground_truth))), ground_truth] + 1
    inverse_rank_code_ = 1 / rank_code_
    masked_inverse_rank_code_ = inverse_rank_code_ * (rank_code_ <= 1000)
    logger.info("eval_mrr of siasime model: %s", float(torch.mean(masked_inverse_rank_code_).item()))
    
    rerank_code_ids = sort_ids_[:, :args.atK].reshape(-1)
    rerank_query_ids = torch.arange(scores.shape[0]).unsqueeze(1).repeat(1, args.atK).reshape(-1)
    rerank_scores = []
    for i in range(0, rerank_code_ids.shape[0], args.eval_batch_size):
        code_inputs = torch.stack([code_dataset[j][0] for j in rerank_code_ids[i:i+args.eval_batch_size]]).to(args.device)
        nl_inputs = torch.stack([query_dataset[j][1] for j in rerank_query_ids[i:i+args.eval_batch_size]]).to(args.device)
        with torch.no_grad():
            code_nl = torch.cat([nl_inputs, code_inputs],dim=1)
            code_nl_score = mono_model(code_nl).cpu()
            rerank_scores.append(code_nl_score.squeeze())
            # scores[rerank_query_ids[i:i+args.eval_batch_size], rerank_code_ids[i:i+args.eval_batch_size]] = code_nl_score.reshape(-1) + 100
            if 'cosqa' in args.train_data_file: 
                scores[rerank_query_ids[i:i+args.eval_batch_size], rerank_code_ids[i:i+args.eval_batch_size]] += code_nl_score.reshape(-1) + 100
            else: 
                scores[rerank_query_ids[i:i+args.eval_batch_size], rerank_code_ids[i:i+args.eval_batch_size]] = code_nl_score.reshape(-1) + 100


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
    parser.add_argument("--train_data_file", default="dataset/CSN/ruby/train.jsonl", type=str, 
                        help="The input training data file (a json file).")
    parser.add_argument("--eval_data_file", default="dataset/CSN/ruby/valid.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default="dataset/CSN/ruby/test.jsonl", type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default="dataset/CSN/ruby/codebase.jsonl", type=str, 
                        help="An optional input test data file to codebase (a jsonl file).") 
    
    parser.add_argument("--model_name_or_path", default="microsoft/unixcoder-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--output_dir_sia_model", default="saved_model/unixcoder/CSN/ruby/", type=str, 
                        help="The saved model checkpoint.")
    parser.add_argument("--output_dir_mono_model", default="saved_model/unixcoder_RR_test/CSN/ruby/", type=str, 
                        help="The saved model checkpoint.")

    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
     
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size for training.")   
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--num_negs", default=31, type=int, 
                        help="Number of hard negative samples for each positive sample.")
    parser.add_argument("--atK", default=5, type=int,
                        help="For top K, ranking by UniModel. ")   
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")         
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--temperature", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--log", default="log/log_.log", type=str,
                        help="log_file.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--with_training', action='store_true', )
    parser.add_argument('--lam', default=0, type=float)

    #print arguments
    args = parser.parse_args()
    #set log
    os.makedirs('/'.join(args.log.split('/')[:-1]), exist_ok=True)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO,
                    filename=args.log, filemode="w"  )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)

    #build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained(args.model_name_or_path) 
 
    sia_model = Unixcoder(RobertaModel.from_pretrained(args.model_name_or_path) )
    mono_model = Unixcoder_RR(RobertaModel.from_pretrained(args.model_name_or_path) )

    logger.info("Training/evaluation parameters %s", args)
    
    sia_model.to(args.device)
    mono_model.to(args.device)
    if args.n_gpu > 1:
        sia_model = torch.nn.DataParallel(sia_model)  
        mono_model = torch.nn.DataParallel(mono_model) 

    # siamese model load 
    model_to_load_sia = sia_model.module if hasattr(sia_model, 'module') else sia_model  
    model_to_load_sia.load_state_dict(torch.load(os.path.join(args.output_dir_sia_model, "model.bin")))

    # mono model load
    # model_to_load_mono = mono_model.module if hasattr(mono_model, 'module') else mono_model  
    # model_to_load_mono.load_state_dict(torch.load(os.path.join("saved_model/unixcoder_rr/StaQC/", "model.bin")))
            
    # Training 
    if args.with_training: 
    # if 1: 
        # print("*******training*******")
        model_best_state_mono = train(args, mono_model, sia_model, tokenizer)
        os.makedirs(args.output_dir_mono_model, exist_ok=True)
        output_file_mono = os.path.join(args.output_dir_mono_model, 'model.bin')
        torch.save(model_best_state_mono, output_file_mono)
      
    # Evaluation
    # mono model load 
    model_to_load_mono = mono_model.module if hasattr(mono_model, 'module') else mono_model  
    model_to_load_mono.load_state_dict(torch.load(os.path.join(args.output_dir_mono_model, "model.bin")))
    eval_data_file, codebase_file = args.eval_data_file, args.codebase_file
    result = evaluate(args, mono_model, sia_model, tokenizer, eval_data_file, codebase_file)
    logger.info("***** Eval results on validation set *****")
    for key in sorted(result.keys()):
        logger.info("%s:  %s = %s", eval_data_file, key, str(round(result[key],3)))


    test_data_file, codebase_file = args.test_data_file, args.codebase_file
    if "AdvTest" in codebase_file: 
        codebase_file = test_data_file
    result = evaluate(args, mono_model, sia_model, tokenizer, test_data_file, codebase_file)
    logger.info("***** Eval results on test set *****")
    for key in sorted(result.keys()):
        logger.info("%s:  %s = %s", test_data_file, key, str(round(result[key],3)))



if __name__ == "__main__":
    main()

