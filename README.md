# Advanced Natural Language Processing class: Kaggle competition


## Installation

```bash	
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## Train RoBERTa or MultilingualBERT

Select the right configuration in `train_hf_models/config.py`.

To start training, launch from the root folder of the git:
```bash
python ./train_hf_models/train.py
```

Before predicting, ensure you have at least one checkpoint in the output path given in the config. The latest checkpoint will be used. Then, just launch:
```bash
python ./train_hf_models/test.py
```
