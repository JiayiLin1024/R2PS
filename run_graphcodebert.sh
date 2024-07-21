#!/bin/bash

for dataset in AdvTest # CSN cosqa StaQC AdvTest 

do
    if [ $dataset == CSN ]
    then
        langs=(ruby php) # javascript go python java 
    else
        langs=(python)
    fi

    for lang in ${langs[@]}
    do

        if [ $dataset == CSN ]
        then
        output_dir=saved_model/graphcodebert/$dataset/$lang/
        train_data_file=dataset/$dataset/$lang/train.jsonl
        eval_data_file=dataset/$dataset/$lang/valid.jsonl
        test_data_file=dataset/$dataset/$lang/test.jsonl
        codebase_file=dataset/$dataset/$lang/codebase.jsonl
        temperature=20
        elif [ $dataset == cosqa ]
        then
        output_dir=saved_model/graphcodebert/$dataset/
        train_data_file=dataset/$dataset/cosqa-retrieval-train-19604.json
        eval_data_file=dataset/$dataset/cosqa-retrieval-dev-500.json
        test_data_file=dataset/$dataset/cosqa-retrieval-test-500.json
        codebase_file=dataset/$dataset/code_idx_map.txt
        temperature=20
        elif [ $dataset == StaQC ]
        then
        output_dir=saved_model/graphcodebert/$dataset/
        train_data_file=dataset/StackOverflow-Question-Code-Dataset/train.jsonl
        eval_data_file=dataset/StackOverflow-Question-Code-Dataset/valid.jsonl
        test_data_file=dataset/StackOverflow-Question-Code-Dataset/test.jsonl
        codebase_file=dataset/StackOverflow-Question-Code-Dataset/codebase.jsonl
        temperature=40
        elif [ $dataset == AdvTest ]
        then
        output_dir=saved_model/graphcodebert/$dataset/
        train_data_file=dataset/AdvTest/train.jsonl
        eval_data_file=dataset/AdvTest/valid.jsonl
        test_data_file=dataset/AdvTest/test.jsonl
        codebase_file=dataset/AdvTest/valid.jsonl
        temperature=40
        fi


        python run_graphcodebert.py \
            --output_dir $output_dir \
            --model_name_or_path microsoft/graphcodebert-base \
            --lang $lang \
            --train_data_file $train_data_file \
            --eval_data_file $eval_data_file \
            --test_data_file $test_data_file \
            --codebase_file $codebase_file \
            --num_train_epochs 10 \
            --code_length 256 \
            --nl_length 128 \
            --data_flow_length 64 \
            --train_batch_size 64 \
            --eval_batch_size 64 \
            --learning_rate 2e-5 \
            --temperature $temperature \
            --seed 123456 \
            --log log/graphcodebert/${dataset}_${lang}.log &

        sleep 5m

    done

done
