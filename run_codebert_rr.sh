#!/bin/bash

for dataset in CSN # cosqa StaQC AdvTest # cosqa StaQC AdvTest # 

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
            output_dir_sia_model=saved_model/codebert/$dataset/$lang/
            output_dir_mono_model=saved_model/codebert_rr/$dataset/$lang/
            train_data_file=dataset/$dataset/$lang/train.jsonl
            eval_data_file=dataset/$dataset/$lang/valid.jsonl
            test_data_file=dataset/$dataset/$lang/test.jsonl
            codebase_file=dataset/$dataset/$lang/codebase.jsonl
            temperature=20
        elif [ $dataset == cosqa ]
        then
            output_dir_sia_model=saved_model/codebert/$dataset/
            output_dir_mono_model=saved_model/codebert_rr/$dataset/
            train_data_file=dataset/$dataset/cosqa-retrieval-train-19604.json
            eval_data_file=dataset/$dataset/cosqa-retrieval-dev-500.json
            test_data_file=dataset/$dataset/cosqa-retrieval-test-500.json
            codebase_file=dataset/$dataset/code_idx_map.txt
            temperature=20
        elif [ $dataset == StaQC ]
        then
            output_dir_sia_model=saved_model/codebert/$dataset/
            output_dir_mono_model=saved_model/codebert_rr/$dataset/
            train_data_file=dataset/StackOverflow-Question-Code-Dataset/train.jsonl
            eval_data_file=dataset/StackOverflow-Question-Code-Dataset/valid.jsonl
            test_data_file=dataset/StackOverflow-Question-Code-Dataset/test.jsonl
            codebase_file=dataset/StackOverflow-Question-Code-Dataset/codebase.jsonl
            temperature=40
        elif [ $dataset == AdvTest ]
        then
            output_dir_sia_model=saved_model/codebert/$dataset/
            output_dir_mono_model=saved_model/codebert_rr/$dataset/
            train_data_file=dataset/AdvTest/train.jsonl
            eval_data_file=dataset/AdvTest/valid.jsonl
            test_data_file=dataset/AdvTest/test.jsonl
            codebase_file=dataset/AdvTest/valid.jsonl
            temperature=40
        fi

        for alpha in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
        do
        python run_codebert_rr.py \
            --output_dir_sia_model $output_dir_sia_model \
            --output_dir_mono_model $output_dir_mono_model \
            --model_name_or_path microsoft/codebert-base \
            --train_data_file $train_data_file \
            --eval_data_file $eval_data_file \
            --test_data_file $test_data_file \
            --codebase_file $codebase_file \
            --num_train_epochs 10 \
            --code_length 256 \
            --nl_length 128 \
            --train_batch_size 40 \
            --eval_batch_size 64 \
            --learning_rate 2e-5 \
            --temperature $temperature \
            --seed 123456 \
            --num_negs 31 \
            --atK 10 \
            --alpha $alpha \
            --log log_sum/codebert_rr/${dataset}_${lang}_eval_sum_${alpha}.log 
        # sleep 5m
        done

    done

done
