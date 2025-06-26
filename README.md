# FluAdaX
## Introduction
FluAdaX is a deep learning framework designed to quantify host adaptation patterns in Influenza A Virus (IAV) using nucleotide sequences. The base module of FluAdaX is a transformer-style model with multiple attention layers. Moving average equipped gated attention (MEGA) is employed as the backbone of FluAdaX to efficiently process extremely long sequences, in terms of computational and information extraction performance. The whole dataset of nucleotide sequences of the IAV was partitioned into training, validation, and test sets at an 8:1:1 ratio according to the collection timeline within each host category. The outputs of FluAdaX are processed with a softmax function to generate a set of probability values (confidence level β) corresponding to the host species. 
## Overview
Three types of models are developed using the FluAdaX framework:
1. **FluAdaX-Genome (FluAdaX-G)**
   - Aim: Predict host adaptation of IAV strains across human, swine, avian, canine, and equine 
   - Input: Concatenated alignment-free nucleotide sequences of all 8 IAV segments (~13kb per strain)
2. **FluAdaX-Segment (FluAdaX-S)**
   - Aim: Predict host adaptation of IAV segments across human, swine, avian, canine, and equine
   - Input: Alignment-free nucleotide sequences of individual IAV gene segments
3. **FluAdaX-AIV**
   - Aim: Host discrimination in avian-to-human transmission for each segment.
   - Inputs: Alignment of nucleotide sequences of individual IAV gene segments
   - Note: FluAdaX-AIV models were developed for each gene segments
## Environment Installation
We recommend to use linux and conda for the environment management.

### Step1: install python 3.8
1) download anaconda3
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
```
2) install conda
```bash
sh Anaconda3-2022.05-Linux-x86_64.sh
```
3) create a virtual environment: python=3.8
```bash
conda create -n FluAdaX python=3.8
```
4) activate FluAdaX
```bash
conda activate FluAdaX
```
### Step2: install other requirements
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Models and Tokenizer Files
Download required model files and tokenizer:
1. **Models**:
   - FluAdaX-G: `model/trained_FluAdaX_G/`
   - FluAdaX-S: `model/trained_FluAdaX_S/`
2. **Tokenizer Files** (place in inference directory):
   - [`inference/bio_tokenizer.py`](path/to/inference/bio_tokenizer.py)
   - [`inference/vocab.txt`](path/to/inference/vocab.txt)

## Inference
### 1. FluAdaX-G (Whole Genome Model)
Run host origin prediction:
```bash
python FluAdaX_G_infer.py
```
#### Input
Create test.csv file with alignment-free nucleotide sequences of whole genome. 
**Example**: inference/FluAdaX_G/test_example.csv
#### Output
##### 1.**.xlsx file** containing:
   - Probability distribution over 5 hosts: human, swine, avian, canine, equine.
   - Predicted host type.
##### 2. **Acc**: 
   - Consistency accuracy between prediction and input host label (if input host is provided)

### 2. FluAdaX-S (Segment-Specific Model)
Run segment-level prediction:
```bash
python FluAdaX_S_infer.py
```
#### Input
Create test.csv file with alignment-free nucleotide segment sequences. 
**Example**: inference/FluAdaX_S/test_example.csv
#### Output
##### 1.**.xlsx file** containing:
   - Probability distribution over 5 hosts: human, swine, avian, canine, equine.
   - Predicted host type.
##### 2. **Acc**: 
   - Consistency accuracy between prediction and input host label (if input host is provided)
