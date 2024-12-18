import torch
from parameters.test_000 import device
from utils import load_ToxDL2_model, obtain_protein_feature


def run_predict():
    pdb_test_file = "../data/pdb_test/P0DQR5.pdb"
    protein_domains = ['IPR004169']
    domain2vector_model_path = "../checkpoints/protein_domain_embeddings.model"
    protein_feature = obtain_protein_feature(pdb_test_file, protein_domains, domain2vector_model_path)
    print(f"Loaded {pdb_test_file}")

    model_path = '../checkpoints/ToxDL2_model.pth'
    model = load_ToxDL2_model(model_path)
    model = model.to('cuda')
    model.eval()

    print(protein_feature)
    with torch.no_grad():
        protein_feature = protein_feature.to(device)
        prediction = model.forward(protein_feature)
        print(protein_feature.name + f"\tPrediction: {prediction.item()}")


if __name__ == '__main__':
    run_predict()
