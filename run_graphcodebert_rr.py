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
from model import GraphCodeBERT, GraphCodeBERT_RR
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                              RobertaConfig, RobertaModel, RobertaTokenizer)
import random

logger = logging.getLogger(__name__)

import time
from gpu import gpu_memory_keeper
# gpu_memory_keeper('tmp' + str(time.time()))


from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript
}

#load parsers
parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser

#remove comments, tokenize code and extract dataflow                                        
def extract_dataflow(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg



class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,                 
                 nl_tokens,
                 nl_ids,
                 url,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.position_idx=position_idx
        self.dfg_to_code=dfg_to_code
        self.dfg_to_dfg=dfg_to_dfg        
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url=url
        
        
def convert_examples_to_features(js,tokenizer,args):
    #code
    parser=parsers[args.lang]
    code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
    #extract data flow
    code_tokens,dfg=extract_dataflow(code,parser,args.lang)
    code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
    ori2cur_pos={}
    ori2cur_pos[-1]=(0,0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
    code_tokens=[y for x in code_tokens for y in x]  
    #truncating
    code_tokens=code_tokens[:args.code_length+args.data_flow_length-2-min(len(dfg),args.data_flow_length)]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
    dfg=dfg[:args.code_length+args.data_flow_length-len(code_tokens)]
    code_tokens+=[x[0] for x in dfg]
    position_idx+=[0 for x in dfg]
    code_ids+=[tokenizer.unk_token_id for x in dfg]
    padding_length=args.code_length+args.data_flow_length-len(code_ids)
    position_idx+=[tokenizer.pad_token_id]*padding_length
    code_ids+=[tokenizer.pad_token_id]*padding_length    
    #reindex
    reverse_index={}
    for idx,x in enumerate(dfg):
        reverse_index[x[1]]=idx
    for idx,x in enumerate(dfg):
        dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
    dfg_to_dfg=[x[-1] for x in dfg]
    dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
    length=len([tokenizer.cls_token])
    dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]        
    #nl
    nl=' '.join(js['docstring_tokens']) if 'docstring_tokens' in js and type(js['docstring_tokens']) is list else ' '.join(js['doc'].split()) if 'doc' in js else ' '.join(js['question'].split())
    nl_tokens=tokenizer.tokenize(nl)[:args.nl_length-2]
    nl_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids+=[tokenizer.pad_token_id]*padding_length    
    
    return InputFeatures(code_tokens,code_ids,position_idx,dfg_to_code,dfg_to_dfg,nl_tokens,nl_ids,js['url'] if "url" in js else js["retrieval_idx"])

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.args=args
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

    def __getitem__(self, item): 
        #calculate graph-guided masked function
        attn_mask=np.zeros((self.args.code_length+self.args.data_flow_length,
                            self.args.code_length+self.args.data_flow_length),dtype=np.bool_)
        #calculate begin index of node and max length of input
        node_index=sum([i>1 for i in self.examples[item].position_idx])
        max_length=sum([i!=1 for i in self.examples[item].position_idx])
        #sequence can attend to sequence
        attn_mask[:node_index,:node_index]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(self.examples[item].code_ids):
            if i in [0,2]:
                attn_mask[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_code):
            if a<node_index and b<node_index:
                attn_mask[idx+node_index,a:b]=True
                attn_mask[a:b,idx+node_index]=True
        #nodes attend to adjacent nodes 
        for idx,nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a+node_index<len(self.examples[item].position_idx):
                    attn_mask[idx+node_index,a+node_index]=True  
                    
        return (torch.tensor(self.examples[item].code_ids),
              torch.tensor(attn_mask),
              torch.tensor(self.examples[item].position_idx), 
              torch.tensor(self.examples[item].nl_ids))


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
    train_dataset = TextDataset(tokenizer, args, args.train_data_file)
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
    for idx in range(args.num_train_epochs): 
        for step,batch in enumerate(train_dataloader):
            # break
            #get inputs
            code_inputs = batch[0].to(args.device)  
            attn_mask = batch[1].to(args.device)
            position_idx = batch[2].to(args.device)
            nl_inputs = batch[3].to(args.device)

            col = torch.arange(nl_inputs.shape[0])[:,None]
            query_idx = col.repeat(1, args.num_negs + 1).reshape(-1)
            row = torch.arange(args.num_negs + 1)[None, :]
            code_idx = ((col + row) % nl_inputs.shape[0]).reshape(-1)
            
            code_nl = torch.cat([nl_inputs[query_idx], code_inputs[code_idx]],dim=1)

            mono_model.train()
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
            # if step > 1: break
            
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
        nl_inputs = batch[3].to(args.device)
        with torch.no_grad():
            nl_vec = sia_model(nl_inputs=nl_inputs) 
            nl_vecs.append(nl_vec.cpu()) 

    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)    
        attn_mask = batch[1].to(args.device)
        position_idx =batch[2].to(args.device) 
        with torch.no_grad():
            code_vec= sia_model(code_inputs=code_inputs, attn_mask=attn_mask,position_idx=position_idx)
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
        attn_mask = torch.stack([code_dataset[j][1] for j in rerank_code_ids[i:i+args.eval_batch_size]]).to(args.device)
        position_idx = torch.stack([code_dataset[j][2] for j in rerank_code_ids[i:i+args.eval_batch_size]]).to(args.device)
        nl_inputs = torch.stack([query_dataset[j][3] for j in rerank_query_ids[i:i+args.eval_batch_size]]).to(args.device)
        with torch.no_grad():
            code_nl = torch.cat([nl_inputs, code_inputs],dim=1)
            code_nl_score = mono_model(code_nl).cpu()
            rerank_scores.append(code_nl_score.squeeze())
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
    
    parser.add_argument("--model_name_or_path", default="microsoft/graphcodebert-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--output_dir_sia_model", default="saved_model/graphcodebert/CSN/ruby/", type=str, 
                        help="The saved model checkpoint.")
    parser.add_argument("--output_dir_mono_model", default="saved_model/graphcodebert/CSN/ruby/", type=str, 
                        help="The saved model checkpoint.")

    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.") 
    parser.add_argument("--lang", default='ruby', type=str,
                        help="language.") 
     
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")   
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--num_negs", default=4, type=int, 
                        help="Number of hard negative samples for each positive sample.")
    parser.add_argument("--atK", default=5, type=int,
                        help="For top K, ranking by UniModel. ")   
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")         
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--temperature", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--log", default="log/log.log", type=str,
                        help="log_file.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

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
 
    sia_model = GraphCodeBERT(RobertaModel.from_pretrained(args.model_name_or_path) )
    mono_model = GraphCodeBERT_RR(RobertaModel.from_pretrained(args.model_name_or_path) )

    logger.info("Training/evaluation parameters %s", args)
    
    sia_model.to(args.device)
    mono_model.to(args.device)
    if args.n_gpu > 1:
        sia_model = torch.nn.DataParallel(sia_model)  
        mono_model = torch.nn.DataParallel(mono_model) 

    # siamese model load 
    model_to_load_sia = sia_model.module if hasattr(sia_model, 'module') else sia_model  
    model_to_load_sia.load_state_dict(torch.load(os.path.join(args.output_dir_sia_model, "model.bin")))
            
    # Training
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

