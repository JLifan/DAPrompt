# DAPrompt: Implementation Instructions

This repository provides the code for implementing and reproducing the proposed method.


---


## Environment Setup

We recommend using **Python 3.9**.

Install dependencies:
```bash
pip install -r requirements.txt
```



## Project Structure
```
DAPrompt/
├── data/
|   ├── download_original_data.py           # download the raw dataset
|   ├── generate_fewshot_tasks_graph.py     # generate graph-level few-shot tasks from the raw dataset
|   ├── generate_fewshot_tasks_node.py      # generate node-level few-shot tasks from the raw dataset
├── weights/                                # pretrained weights
├── backbone.py                             # base backbone model
├── main.py                                 # main experiment entry points
├── pretraining.py                          # implementation of the pretraining method
├── prompt.py                               # implementation of the prompt module
├── utils.py                                # utility functions
├── requirements.txt                        # python dependencies
└── README.md                               # this file
```


## Running Experiments

All experiments are conducted through main.py, which contains the full workflow including both pretraining and prompt-based fine-tuning.

Different experiments require different configurations, such as the dataset name, dataset path, and the pre-trained model checkpoint.

```bash
# Example: running a 5-shot experiment on the Cornell dataset using the pretrained model
python main.py --dataset_name Cornell --data_path ./data/node_level/Cornell_5shot.pt --pre_save_path ./weights/DAPrompt_GCN_Cornell_K5_best.pt
```











