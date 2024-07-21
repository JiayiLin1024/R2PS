#!/bin/bash

for dataset in CSN # CSN cosqa StaQC AdvTest

do
    if [ $dataset == CSN ]
    then
        langs=(ruby) # ruby javascript go python java php
    else
        langs=(python)
    fi

    for lang in ${langs[@]}
    do

        if [ $dataset == CSN ]
        then
            output_dir_sia_model=saved_model/unixcoder/$dataset/$lang/
            output_dir_mono_model=saved_model/unixcoder_rr_hard/$dataset/$lang/
            train_data_file=dataset/$dataset/$lang/train.jsonl
            eval_data_file=dataset/$dataset/$lang/valid.jsonl
            test_data_file=dataset/$dataset/$lang/test.jsonl
            codebase_file=dataset/$dataset/$lang/codebase.jsonl
            temperature=20
            lam=0.0
        elif [ $dataset == cosqa ]
        then
            output_dir_sia_model=saved_model/unixcoder/$dataset/
            output_dir_mono_model=saved_model/unixcoder_rr_hard/$dataset/
            train_data_file=dataset/$dataset/cosqa-retrieval-train-19604.json
            eval_data_file=dataset/$dataset/cosqa-retrieval-dev-500.json
            test_data_file=dataset/$dataset/cosqa-retrieval-test-500.json
            codebase_file=dataset/$dataset/code_idx_map.txt
            temperature=20
            lam=0.0
        elif [ $dataset == StaQC ]
        then
            output_dir_sia_model=saved_model/unixcoder/$dataset/
            output_dir_mono_model=saved_model/unixcoder_rr_hard/$dataset/
            train_data_file=dataset/StackOverflow-Question-Code-Dataset/train.jsonl
            eval_data_file=dataset/StackOverflow-Question-Code-Dataset/valid.jsonl
            test_data_file=dataset/StackOverflow-Question-Code-Dataset/test.jsonl
            codebase_file=dataset/StackOverflow-Question-Code-Dataset/codebase.jsonl
            temperature=40
            lam=0.004
        elif [ $dataset == AdvTest ]
        then
            output_dir_sia_model=saved_model/unixcoder/$dataset/
            output_dir_mono_model=saved_model/unixcoder_rr_hard/CSN/python/
            train_data_file=dataset/AdvTest/train.jsonl
            eval_data_file=dataset/AdvTest/valid.jsonl
            test_data_file=dataset/AdvTest/test.jsonl
            codebase_file=dataset/AdvTest/valid.jsonl
            temperature=40
            lam=0.0
        fi

        
        # for lam in 0.003 0.004 0.005 0.006 0.007 0.008 0.009 # 0.002 0.003 
        # do 
        python run_unixcoder_rr_hard.py \
            --output_dir_sia_model $output_dir_sia_model \
            --output_dir_mono_model $output_dir_mono_model \
            --model_name_or_path microsoft/unixcoder-base \
            --train_data_file $train_data_file \
            --eval_data_file $eval_data_file \
            --test_data_file $test_data_file \
            --codebase_file $codebase_file \
            --num_train_epochs 10 \
            --code_length 256 \
            --nl_length 128 \
            --train_batch_size 32 \
            --eval_batch_size 64 \
            --learning_rate 2e-5 \
            --temperature $temperature \
            --seed 123456 \
            --num_negs 31 \
            --atK 10 \
            --lam $lam \
            --log log/unixcoder_rr_hard/${dataset}_${lang}_final_test.log \
            --with_training
        # done

        # sleep 5m

    done

done
