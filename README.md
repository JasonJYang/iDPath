# iDPath
This is the PyTorch implementation for our paper published on Briefings in Bioinformatics:
> [Deep Learning Identifies Explainable Reasoning Paths of Mechanism of Drug Action for Drug Repurposing from Multilayer Biological Network](https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbac469/6809964?login=true)

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
* Download the [test data](https://drive.google.com/file/d/1WeG75vYUbNlP96kc6IHvbAedTRGG57i5/view?usp=sharing) and put these files under the folder `data/test`.

# Example to Run the Codes
## Train 
```bash
python train.py --config config/config.json 
```
## Test 
When the training is finished, you will get a file that records the parameters for the best model, remember its location (such as `saved/models/iDPath/0117_164440/model_best.pth`) and use it for testing.
```bash
python test.py --config config/config.json --resume saved/models/iDPath/0117_164440/model_best.pth
```
## Inference
1. Data. To make an inference on the new drug-disease pair, you need to prepare a csv file named `test.csv` under the folder `data/test` with the following fields, where the drug is denoted by its PubChem CID and disease is denoted by its ICD10 code. Note that if your input drugs or diseases cannot be found in our dataset, the corresponding pairs will be ignored.
```
drug_pubchemcid	    disease_icd10
CID-132971	        ICD10-C61
```
2. Pre-trained model. You can use your own pre-trained model or use our prepared one [`model_best.pth`](https://drive.google.com/file/d/1WeG75vYUbNlP96kc6IHvbAedTRGG57i5/view?usp=sharing) and put the `config.json` and `model_best.pth` to the folder `data/test`.
3. Run. We provide an argument `K` in the `inference_config.json` to control the output of the number of top-k critical paths identified by iDPath. Please use the following command to run the inference.
```bash
python inference.py --resume data/test/model_best.pth --config config/inference_config.json
``` 
4. Result. After the inference is done, you will get a file named `result.csv` under the folder `saved/models/iDPath/xxxx_xxxxxx` (where `xxxx_xxxxxx` is your runing time as the runing id). The `result.csv` contains the predicted probability of therapeutic effect and top-k critical paths of your input drug-disease pairs.

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
