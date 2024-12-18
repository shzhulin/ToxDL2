import torch
import os.path
from parameters.test_000 import device
import numpy as np
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import f1_score, precision_recall_curve, matthews_corrcoef
from pathlib import Path
from dataset import pdb_to_graph
from torch_geometric.data import Data
from gensim.models import Word2Vec


def calc_metrics_for_test(labels, preds):
    auroc = roc_auc_score(labels, preds)
    precision, recall, thresholds = precision_recall_curve(labels, preds)
    auprc = auc(recall, precision)
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    f1score = f1_score(labels, preds, average='binary')
    mcc = matthews_corrcoef(labels, preds)
    return auroc, auprc, f1score, mcc


def evaluation_result_obtained(predicted_results_save_path):
    with open(predicted_results_save_path) as fp:
        allLines = fp.readlines()
        num_test_proteins = len(allLines) // 3
        preds = np.zeros(num_test_proteins, dtype=np.float32)
        labels = np.zeros(num_test_proteins, dtype=np.int32)
        for index in range(num_test_proteins):
            labels[index] = float(allLines[index * 3 + 1].strip())
            preds[index] = float(allLines[index * 3 + 2].strip())
    auROC, auPRC, Fmax, mcc = calc_metrics_for_test(labels, preds)
    print('F1:', Fmax, 'MCC', mcc, 'auROC:', auROC, 'auPRC', auPRC)


def data_loader_result_write(model_best_save_path, test_loader, test_predicted_save_path):
    net_trained = torch.load(model_best_save_path)
    net_trained.eval()
    with open(test_predicted_save_path, 'w') as fp:
        for data in test_loader:
            data = data.to(device)
            data_copy = data.clone().detach()
            pred = net_trained.forward(data_copy).squeeze().to(torch.float32)
            for protein_name, protein_length, protein_label, protein_predict in zip(data.name, data.length, data.y,
                                                                                    pred):
                print(protein_name, file=fp)
                print(str(protein_label.item()), file=fp)
                print(str(protein_predict.item()), file=fp)


def pytorch_model_store(net, model_best_save_path):
    model_best_save_dir = os.path.dirname(model_best_save_path)
    if not os.path.exists(model_best_save_dir):
        os.makedirs(model_best_save_dir)
    torch.save(net, model_best_save_path)


def load_ToxDL2_model(model_path):
    model = torch.load(model_path)
    return model


def get_domain_vector(protein_domains, domain2vector_model_path):
    domain2vector_model = Word2Vec.load(domain2vector_model_path)
    domain_embeddings = [domain2vector_model.wv[domain]
                         for domain in protein_domains if domain in domain2vector_model.wv]
    if domain_embeddings:
        return np.expand_dims(np.mean(domain_embeddings, axis=0), axis=0)
    else:
        return np.expand_dims(np.zeros(domain2vector_model.vector_size), axis=0)


def obtain_protein_feature(pdb_data_path, protein_domains, domain2vector_model_path):
    # Create a Data object for the current protein
    protein_node_feat, protein_edge_index, protein_name, protein_sequence = pdb_to_graph(Path(pdb_data_path))
    protein_length = len(protein_sequence)
    domain_vector = get_domain_vector(protein_domains, domain2vector_model_path)
    # domain_vector = np.zeros([1, 256])
    # unknown tested protein label information
    y = -1
    data_item = Data(
        x=protein_node_feat,
        edge_index=protein_edge_index,
        name=protein_name,
        sequence=protein_sequence,
        length=protein_length,
        vector=domain_vector,
        y=torch.tensor(float(y), dtype=torch.float),
    )
    return data_item
