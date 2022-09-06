# iDPath for IC50 prediction
Please download the code for IC50 prediction from [Google Drive](https://drive.google.com/file/d/1WeG75vYUbNlP96kc6IHvbAedTRGG57i5/view?usp=sharing).
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
1. Data. To make an inference on the new drug-cellline pair, you need to prepare a csv file named `test.csv` under the folder `data/test` with the following fields, where the drug is denoted by its PubChem CID and cellline is denoted by its name. Note that if your input drugs or celllines cannot be found in our dataset, the corresponding pairs will be ignored.
```
drug_pubchemcid	    CellLine
CID-24360	        CellLine-A673
```
2. Pre-trained model. You need to use the pre-trained model and put the `config.json` and `model_best.pth` to the folder `data/test`.
3. Run. We provide an argument `K` in the `inference_config.json` to control the output of the number of top-k critical paths identified by iDPath. Please use the following command to run the inference.
```bash
python inference.py --resume data/test/model_best.pth --config config/inference_config.json
``` 
4. Result. After the inference is done, you will get a file named `result.csv` under the folder `saved/models/iDPath/xxxx_xxxxxx` (where `xxxx_xxxxxx` is your runing time as the runing id). The `result.csv` contains the predicted IC50 values and top-k critical paths of your input drug-cellline pairs.
  
# License
Distributed under the GPL-2.0 License License. See `LICENSE` for more information.

# Contact
Jiannan Yang - jiannan.yang@my.cityu.edu.hk
