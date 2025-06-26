# FluAdaX
## Introduction
FluAdaX is a deep learning framework designed to quantify host adaptation patterns in Influenza A Virus (IAV) using nucleotide sequences. The base module of FluAdaX is a transformer-style model with multiple attention layers. Moving average equipped gated attention (MEGA) is employed as the backbone of FluAdaX to efficiently process extremely long sequences, in terms of computational and information extraction performance. The whole dataset of nucleotide sequences of the IAV was partitioned into training, validation, and test sets at an 8:1:1 ratio according to the collection timeline within each host category. The outputs of FluAdaX are processed with a softmax function to generate a set of probability values (confidence level β) corresponding to the host species. 
## Overview
Three types of models are developed using the FluAdaX framework:
1. FluAdaX-Genome (FluAdaX-G)
Input: Concatenated alignment-free nucleotide sequences of all 8 IAV segments (~13kb per strain)
2. FluAdaX-Segment (FluAdaX-S)
Input: Alignment-free nucleotide sequences of individual IAV gene segments
3.FluAdaX-AIV
Inputs: Alignment of nucleotide sequences of individual IAV gene segments
## Environment Installation
We recommend to use linux and conda for the environment management.
Step1: install python 3.8
1) download anaconda3
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
2) install conda
sh Anaconda3-2022.05-Linux-x86_64.sh
3) create a virtual environment: python=3.8
conda create -n FluAdaX python=3.8
4) activate FluAdaX
conda activate FluAdaX
Step2: install other requirements
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
## Inference
### 1. FluAdaX-G (Whole Genome Model)
Run host origin prediction using FluAdaX-G:
```bash
python FluAdaX_G_infer.py
Prepare test.csv file as input. For format reference, see: inference/FluAdaX_G/test_example.csv
Output file contains is in .xlsx format containing probability distribution over 5 hosts: human, swine, avian, canine, equine; and predicted host type. Consistency accuracy between prediction and input host label is also provided.
3. FluAdaX-S
Run host origin prediction using FluAdaX-S:
```bash
python FluAdaX_S_infer.py
Prepare test.csv file as input. For format reference, see: inference/FluAdaX_S/test_example.csv
Output file contains is in .xlsx format containing probability distribution over 5 hosts: human, swine, avian, canine, equine; and predicted host type. Consistency accuracy between prediction and input host label is also provided.
