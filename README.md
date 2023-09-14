# Cross Domain Acronym Disambiguation

## Task and Motivation
Task: Acronym Disambiguation <br>
Motivation: Identification of ambiguous acronyms is a major challenge in information retrieval and analysis. 

According to [Liu et al](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2244212/), almost 81% acronyms used in MEDLINE are ambiguous.

<img width="814" alt="Screenshot 2023-09-14 at 4 59 04 PM" src="https://github.com/abhibha1807/Cross-Domain-Acronym-Disambiguation/assets/34295094/16d5b3b0-375f-43d8-88d3-da830615e2e1">


## Dataset
Our [dataset](https://github.com/amirveyseh/AAAI-21-SDU-shared-task-2-AD) is publicly available as part of the AAAI-21 shared task on Acronym Disambiguation. 

## How to run

### Baseline:
To implement baseline: `run /baseline/main.py`

### Transfer learning approach
For BERT: ` run /TransferLearning/BERT_train.py `
For UlmFit:  ` run /TransferLearning/UlmFit.ipynb ` 

### Results

<img width="848" alt="Screenshot 2023-09-14 at 4 56 03 PM" src="https://github.com/abhibha1807/Cross-Domain-Acronym-Disambiguation/assets/34295094/2428b9b0-eedc-49bc-b2ec-c41e216acd37">



For more information refer to this [pdf](https://github.com/abhibha1807/Cross-Domain-Acronym-Disambiguation/blob/main/ppt.pdf)
