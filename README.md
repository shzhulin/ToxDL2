# ToxDL 2.0: Protein toxicity prediction based on pretrained language model with graph neural networks
<br>
Assessing the potential toxicity of proteins is crucial for therapeutic proteins, facilitating accurate protein toxicity prediction for both therapeutic and agricultural applications. Traditional experimental methods for toxicity evaluation are time-consuming, expensive, and labor-intensive, highlighting the requirement for efficient computational approaches. Recent advancements in language models and deep learning have significantly improved protein toxicity prediction, yet current models often lack the ability to integrate spatial and structural information, which is crucial for accurate toxicity assessment.
<br>
<br>
In this study, we present ToxDL 2.0, a novel multimodal deep learning model for protein toxicity prediction that integrates both evolutionary and structural information derived from the pretrained language model and AlphaFold2. ToxDL 2.0 consists of three key modules: (1) a Graph Convolutional Network (GCN) module for generating protein graph embeddings based on AlphaFold2-predicted structures, (2) a domain embedding module for capturing protein domain representations, and (3) a dense module that combines these embeddings to predict toxicity.

# Dependency:
python              	3.9 <br>
torch					1.13.1 <br>
torch_geometric			2.5.3 <br>
fair-esm				2.0.0 <br>
graphein				1.7.6 <br>
networkx				3.2.1 <br>
gensim              	4.3.2 <br>
biopython				1.82 <br>
biotite					0.40.0 <br>
logomaker           	0.8 <br>
scikit-learn         	1.0.2 <br>
scipy               	1.10.1 <br>
numpy               	1.26.4 <br>
pandas              	1.5.3 <br>
matplotlib          	3.8.4 <br>
joblib              	1.4.2 <br>

# OS Requirements
This package is supported for *Linux* operating systems. The package has been tested on the following systems:
<br>
Linux: Ubuntu 20.04 
# Files
The project contains the following files and directories:
<br>
```checkpoints/```: Directory to store trained model checkpoints.
<br>
```data/```: Directory for storing datasets.
<br>
```predictions/```: Directory for storing prediction results.
<br>
```src/dataset.py```: Dataset loading and preprocessing.
<br>
```src/train_ToxDL2.py```: Training pipeline.
<br>
```src/predict_ToxDL2.py```: Script for making toxicity predictions on new data.
<br>
```src/model.py```: Implementation of network architecture of ToxDL 2.0.
<br>
```src/utils.py```: General utility functions.
<br> 

# Demo
## Run ToxDL 2.0 for training
You first make empty directory "checkpoints/" and "predictions/", then directly run the below command to run ToxDL 2.0 model with default hyperparamters:
<br>
```python train_ToxDL2.py```
<br>
The above command will output performance metrics in F1-score, MCC, auROC and auPRC.
<br>
You can also specity the hyperparamters by modifying the file "parameters/test_000.py"
<br>
The trained model is saved at the directory "checkpoints/".
<br>
The output file is saved at the directory "predictions/".
## Run ToxDL 2.0 for prediction
Please execute the following command directly if you can provide the PDB file.
<br>
```python predict_ToxDL2.py```
<br>
If you do not have a PDB file, you can use AlphaFold2 to predict the protein structure.

# Web service
You can also predict toxicity score for new proteins using the online web service at http://www.csbio.sjtu.edu.cn/bioinf/ToxDL2/. 
