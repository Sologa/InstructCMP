# InstructCMP: Length Control in Sentence Compression through Instruction-based Large Language Models
* Accepted to ACL 2024 Findings Long
* Authors: Juseon-Do, Jingun Kwon, Hidetaka Kamigaito, Manabu Okumura
* Paper Link: [InstructCMP](https://aclanthology.org/2024.findings-acl.532/)


## InstructCMP Dataset
Dataset folder has the following structure:
```
InstructCMP
├── dataset folder
│   ├── Google
│   │   ├──google_test.jsonl
│   │   ├──google_valid.jsonl
│   │   └──google_train.jsonl
|   |
│   ├── Broadcast
│   │   └──broadcast_test.jsonl
|   |
│   ├── BNC
│   │   └──bnc_test.jsonl
|   |
│   └── DUC2004
│       └──duc2004_test.jsonl
|
├── src
│   ├── evaluate_utils
│   │   evaluate_functions.py
|   |
│   ├── inference_utils
│   │   └──functions.py
|   |
│   └── utils
|      └──templates.py
|
└── run.py
```

## Run

```
$ cd InstructCMP
$ python src/run.py --model_size "13" \
                    --batch_size 10 \
                    --data_name:str "Google" \
                    --split "test"
```

# Evaluation
The metrics used in this work are in [evaluation_metrics](https://github.com/JuseonDo/InstructCMP/evaluation).

```python
post_processed_outputs = generated_output_post_processing(generated_text)
result = evaluate(targets, sources, post_processed_outputs)
```

# Contact
If you have any questions about this work, please contact **Juseon-Do** using the following email addresses: **dojuseon@gmail.com** or **doju00@naver.com**. 

