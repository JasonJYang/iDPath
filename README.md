# iDPath
This is our PyTorch implementation for the paper (Preprint: https://www.researchsquare.com/article/rs-1269212/v1):
> Deep Learning Can Identify Explainable Reasoning Paths of Mechanism of Drug Action for Drug Repurposing from Multilayer Biological Network

# Introduction
iDPath is an interpretable deep learning-based path-reasoning framework to identify potential drugs for the treatment of diseases, by capturing the mechanism of drug action (MODA) based on simulating the paths from drugs to diseases in the human body.

# Environment Requirement
The code has been tested running under Python 3.7. The required package are as follows:
* pytorch == 1.6.0
* numpy == 1.19.1
* sklearn == 0.23.2
* networkx == 2.5
* pandas == 1.1.2

# Installation
1. To install the required packages for running iDPath, please use the following    command first. If you meet any problems when installing pytorch, please refer to [pytorch official website](https://pytorch.org/)
```bash
pip install -r requirements.txt
```

2. You may need to download the following files to run iDPath
* Download the the [shortest paths](https://drive.google.com/file/d/11pHbXWHsRIfPmMBDNXdwVyxn2opP0a4s/view?usp=sharing) between all the targets of drugs and diseases and put two files (`disease_path_dict.pkl` and `drug_path_dict.pkl`) under the folder `data/path`.
* Download the [processed data](https://drive.google.com/file/d/1UWijysxx4qHtfI4CY5nRo4ew4iGSZhzo/view?usp=sharing) and put these files under the folder `data/processed`.

# Example to Run the Codes
* Train. 
```bash
python train.py --config config/config.json 
```
* Test. When the training is finished, you will get a file that records the parameters for the best model, remember its location (such as `saved/models/iDPath/0117_164440/model_best.pth`) and use it for testing.
```bash
python test.py --config config/config.json --resume saved/models/iDPath/0117_164440/model_best.pth
```

# Dataset
Datasets used in the paper:
* [RegNetwork](http://www.regnetworkweb.org/home.jsp)
* [STRING](https://version-10-5.string-db.org/cgi/download.pl?species_text=Homo+sapiens)
* [STITCH](http://stitch.embl.de/cgi/download.pl?UserId=zHfYv4tsZAzR&sessionId=tPYL1GXyX6xd&species_text=Homo+sapiens)
* [DrugBank](https://go.drugbank.com/releases/latest#full)
* [TTD](http://db.idrblab.net/ttd/)
* [DisGeNet](https://www.disgenet.org/home/)
* [HMDD](https://www.cuilab.cn/hmdd)
* [Cheng et al.' PPI dataset](https://www.nature.com/articles/s41467-019-09186-x#data-availability)
  
# License
Distributed under the GPL-2.0 License License. See `LICENSE` for more information.

# Contact
Jiannan Yang - jiannan.yang@my.cityu.edu.hk
