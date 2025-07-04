# FluAdaX
## Introduction
FluAdaX is a deep learning framework designed to quantify host adaptation patterns in Influenza A Virus (IAV) using nucleotide sequences. The base module of FluAdaX is a transformer-style model with multiple attention layers. Moving average equipped gated attention (MEGA) is employed as the backbone of FluAdaX to efficiently process extremely long sequences, in terms of computational and information extraction performance. The whole dataset of nucleotide sequences of the IAV was partitioned into training, validation, and test sets at an 8:1:1 ratio according to the collection timeline within each host category. The outputs of FluAdaX are processed with a softmax function to generate a set of probability values (confidence level β) corresponding to the host species. 
## 1. Overview
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
## 2. Environment Installation
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

## 3. Models and Tokenizer Files
Download required model files and tokenizer:
1. **Models**:
   - FluAdaX-G: `model/trained_FluAdaX_G/`
   - FluAdaX-S: `model/trained_FluAdaX_S/`
2. **Tokenizer Files** (place in inference directory):
   - [`inference/bio_tokenizer.py`](path/to/inference/bio_tokenizer.py)
   - [`inference/vocab.txt`](path/to/inference/vocab.txt)

## 4. Inference
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

## 5. Risk Assessment

Run risk assessment:

**⚠️Code is coming soon!!!**

### Assessment details
#### 1.Host origin prediction:
We examine the top 2 predicted host probabilities.
- Single dominant probability (>0.8) → prefers host with highest probability 
- Two significant probabilities (both ≥0.2) → prefers non-human host with highest probability

#### 2. Spillover score calculation:
>Note: The spillover score should only be used for evaluation when the collection host is  explicitly known to be one of these five host types: Human, Swine, Avian, Canine, or Equine.
- The spillover score is calculated from the predicted host origin
- The score range of each host origin is provided in λsp_range.txt
- A higher value indicating a greater risk of spillover from the original host

#### 3. Adaptability score calculation:
- The adaptability score is calculated from the predicted host origin to target host X
- The score range of each host origin is provided in λad(X)_range.txt
- A higher value indicating a greater risk of adaptation to the target host X

####  Input
Case_result.xlsx file generated using FluAdaX-G

####  Output
**Case_prediction.xlsx file** containing:
- Host_origin
- Spillover score λsp.
- Adaptability score targeting five hosts: λad(human), λad(swine), λad(avian), λad(canine), λad(equine).

## Data availability
The datasets can be access at Zenodo (https://doi.org/10.5281/zenodo.15803258)

##  Citation
If you use FluAdaX in your research, please cite the following:
```bash
@article {Yang2025.06.17.660059,
	author = {Yang, Jiaying and Fang, Pan and Liang, Jianqiang and Chen, Yihao and Yang, Lei and Zhu, Wenfei and Shi, Mang and Du, Xiangjun and Pu, Juan and Wang, Dayan and Xue, Guirong and Li, Zhaorong and Shu, Yuelong},
	title = {Using Artificial Intelligence to Assess Cross-Species Transmission Potential of Influenza A Virus},
	elocation-id = {2025.06.17.660059},
	year = {2025},
	doi = {10.1101/2025.06.17.660059},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Influenza A viruses (IAVs) pose pandemic threats through cross-species transmission, yet predicting their adaptive evolution remains challenging. We introduced Influenza A virus Adaptability to host X (FluAdaX), a deep learning framework that integrates a moving average-equipped gated attention mechanism using full-genome sequences. FluAdaX demonstrated robust host classification performance across endemic IAV strains, and outperformed traditional models in detecting avian-to-human transmission. Spillover score and adaptability score were introduced to evaluate host shift risk, which prioritized variants with elevated human adaptation potential, such as H7N9, H9N2 avian IAVs, and H1N1 swine IAVs. Besides HA and NA genes, PB2 and NS genes were found critical for zoonosis. Potential molecular markers associated with avian/human tropism were identified across PB2 and NS genes using XGBoost. FluAdaX provided a dynamic framework to decode IAV host adaptation, enabling real-time risk assessment of cross-species transmission of emerging IAV variants.Competing Interest StatementThe authors have declared no competing interest.the National Key Research and Development Program of China, 2021YFC2300100Non-profit Central Research Institute Fund of Chinese Academy of Medical Sciences, 2022-RC310-02the National Nature Science Foundation of China, 81961128002, 82341118},
	URL = {https://www.biorxiv.org/content/early/2025/06/27/2025.06.17.660059},
	eprint = {https://www.biorxiv.org/content/early/2025/06/27/2025.06.17.660059.full.pdf},
	journal = {bioRxiv}
}
```
