# Dynamic Debiasing of Multi-Hop Fact Verification via Counterfactual Reasoning 
## CRFV Framework
![CRFV](image/Fig1.png)
## 📦 Download

To replicate the results presented in the paper, please download the dataset and pre-trained models from [this shared folder](https://drive.google.com/drive/folders/1gU_92qYOAVcifI2H8bcxBzf4VPZtPINc?usp=sharing).

Then organize them in the following structure:

```
CRFV
├── data
├── pretrained_models
├── outputs
├── data_load_utils.py
├── models.py
├── train_CRFV_fever_2way.py
├── train_CRFV_politihop_2way.py
├── train_CRFV_politihop_3way.py
├── train.sh
└── utils.py
```
## 🛠️ Environment

Our experiments are conducted using PyTorch 2.0.0 on a single NVIDIA RTX 4090D GPU (24 GB). 
Install dependencies with:
```bash
pip install python=3.8.10
pip install torch==2.0.0
pip install torch-geometric
pip install pytorch-pretrained-bert
pip install texar-pytorch
pip install scikit-learn
pip install pandas


