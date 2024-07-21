# R2PS for code search


## 1. Download dataset

- CSN dataset

```bash
cd dataset
wget https://github.com/microsoft/CodeBERT/raw/master/GraphCodeBERT/codesearch/dataset.zip
unzip dataset.zip && rm -r dataset.zip && mv dataset CSN && cd CSN
bash run.sh 
cd ../..
```

- CoSQA dataset

```bash
cd dataset
mkdir cosqa && cd cosqa
wget https://github.com/Jun-jie-Huang/CoCLR/raw/main/data/search/code_idx_map.txt
wget https://github.com/Jun-jie-Huang/CoCLR/raw/main/data/search/cosqa-retrieval-dev-500.json
wget https://github.com/Jun-jie-Huang/CoCLR/raw/main/data/search/cosqa-retrieval-test-500.json
wget https://github.com/Jun-jie-Huang/CoCLR/raw/main/data/search/cosqa-retrieval-train-19604.json
cd ../..
```

- AdvTest dataset

```bash
cd dataset
wget https://github.com/microsoft/CodeXGLUE/raw/main/Text-Code/NL-code-search-Adv/dataset.zip
unzip dataset.zip && rm -r dataset.zip && mv dataset AdvTest && cd AdvTest
wget https://zenodo.org/record/7857872/files/python.zip
unzip python.zip && python preprocess.py && rm -r python && rm -r *.pkl && rm python.zip
cd ../..
```

- [StaQC dataset](https://github.com/LittleYUYU/StackOverflow-Question-Code-Dataset)

```bash
cd dataset/StackOverflow-Question-Code-Dataset
```



## 2. Dependency 
```bash
pip install torch
pip install transformers
```

## 3. Training and Evaualtion 

#### (a) CodeBERT+RR

```bash
run_codebert_rr.sh
```

#### (b) GraphCodeBERT+RR

```bash
run_graphcodebert_rr.sh
```
#### (c) UniXcoder+RR

```bash
run_unixcoder_rr.sh
```
#### (d) UniXcoder+R2PS

```bash
run_unixcoder_rr_hard.sh
```
## 4. Results (MRR)

| model  | CSN | CoSQA | StaQC | AdvTest |
| --- | --- | --- | --- | --- |
| SyncoBERT |  74.0  |  -   |  -  |  38.1  |
| CodeRetriever |  75.3  | 69.6    | -   | 43.0   |
| CodeBERT | 70.3   | 66.9    | 21.4   |  34.9  |
| CodeBERT+RR | 75.9   | 68.3    | 26.2   |  42.0  |
| GraphCodeBERT | 71.7   |  68.7   |  20.5  |  39.1  |
| GraphCodeBERT+RR | 77.0   | 69.8    |  24.3  | 45.4   |
| UniXcoder | 74.4   |  69.6   | 25.8   | 42.5   |
| UniXcoder+RR | 77.5   |  71.5   |  29.0  | 45.4   |
| UniXcoder+R2PS |  79.2  |  72.3   | 29.4   |  47.0  |








