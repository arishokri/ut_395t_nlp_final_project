### NLI Task

| setup | dataset | model         | runime on 3090 | accuracy |
| ----- | ------- | ------------- | -------------- | -------- |
| Base  | SNLI    | Electra-small | 50m            | 80.0% ?? |

### QA Task

| setup | dataset            | model         | runime on 3090 | EM    | F1    |
| ----- | ------------------ | ------------- | -------------- | ----- | ----- |
| Base  | SQuAD              | Electra-small | 24m:22s        | 78.20 | 86.24 |
| Base  | emrQA<sup>[1]<sup> | Electra-small |                |       |       |

### References:

1. emrQA: A Large Corpus for Question Answering on Electronic Medical Records: [Article](https://arxiv.org/abs/1809.00732); [Dataset](https://huggingface.co/datasets/Eladio/emrqa-msquad) needs verification
