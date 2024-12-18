import math
import os
from pathlib import Path
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from parameters.test_000 import DATA, batch_size, device
import esm

import warnings
warnings.filterwarnings('ignore')

# Load ESM model
protein_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
protein_model = protein_model.to(device)
protein_model.eval()

AMINOACID = 'ACDEFGHIKLMNPQRSTVWY'
aa_codes = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'LYS': 'K',
    'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

# Generate a mapping from amino acids to numbers for one-hot encoding
aa_to_num = dict(
    (aa, i)
    for i, aa in enumerate(
        [
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLU",
            "GLN",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
            "UNK",
        ]
    )
)


def graph_node_load(pdb_ID, seq):
    seq_feat = torch.load(DATA / "esm_data" / f'{pdb_ID}_esm2.ts')
    return seq_feat


def pretrain_protein(data):
    """
    Pretrain protein function.
    """
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = protein_model(batch_tokens.to(device), repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    feat = token_representations.squeeze(0)[1:len(data[0][1])+1]
    return feat


def graph_node_obtain(pdb_ID, seq):
    """
    Graph node save function.
    """
    if len(seq) > 1022:
        seq_feat = []
        for i in range(len(seq)//1022):
            data = [(pdb_ID, seq[i*1022:(i+1)*1022])]
            seq_feat.append(pretrain_protein(data))
        data = [(pdb_ID, seq[(i+1)*1022:])]
        seq_feat.append(pretrain_protein(data))
        seq_feat = torch.cat(seq_feat, dim=0)
    else:
        data = [(pdb_ID, seq)]
        seq_feat = pretrain_protein(data)
    seq_feat = seq_feat.cpu()
    return seq_feat


def pdb_to_graph(pdb_file_path, max_dist=8.0):
    # read in the PDB file by looking for the Calpha atoms and extract their amino acid and coordinates based on the
    # positioning in the PDB file
    pdb_ID = pdb_file_path.name[:-4]
    residues = []
    with open(pdb_file_path, "r") as protein:
        for line in protein:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                residues.append(
                    (
                        line[17:20].strip(),
                        float(line[30:38].strip()),
                        float(line[38:46].strip()),
                        float(line[46:54].strip()),
                    )
                )
    # Finally compute the node features based on the amino acids in the protein
    seq = ''.join([aa_codes[res[0]] for res in residues])
    # node_feat = graph_node_load(pdb_ID, seq)
    node_feat = graph_node_obtain(pdb_ID, seq)

    # compute the edges of the protein by iterating over all pairs of amino acids and computing their distance
    edges = []
    for i in range(len(residues)):
        res = residues[i]
        for j in range(i + 1, len(residues)):
            tmp = residues[j]
            if math.dist(res[1:4], tmp[1:4]) <= max_dist:
                edges.append((i, j))
                edges.append((j, i))

    # store the edges in the PyTorch Geometric format
    edges = torch.tensor(edges, dtype=torch.long).T
    return node_feat, edges, pdb_ID, seq


class ToxProteinDataset(InMemoryDataset):
    def __init__(self, folder_name, file_index, transform=None, pre_transform=None, pre_filter=None):
        self.folder_name = folder_name
        super().__init__(root=folder_name, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[file_index])

    @property
    def processed_file_names(self):
        return ["train.pt", "val.pt", "test.pt", "independent.pt"]

    def torch_save_pt(self, data_type):
        protein_graphs = dict(
            [
                (filename[:-4], pdb_to_graph(Path(self.folder_name) / "pdb_data" / data_type / filename))
                for filename in os.listdir(Path(self.folder_name) / "pdb_data" / data_type)
            ]
        )

        with open(Path(self.folder_name) / "domain_data" / (data_type + ".domain")) as fp:
            data_list = []
            data_num = 0
            data_list_index = 0
            for line in fp:
                line = line.strip()
                if line.startswith('>'):
                    if data_num > 0:
                        data_list.append(
                            Data(
                                x=protein_node_feat,
                                edge_index=protein_edge_index,
                                name=protein_name,
                                sequence=protein_sequence,
                                length=protein_length,
                                vector=domain_vector,
                                y=torch.tensor(float(y), dtype=torch.float),
                            )
                        )
                        data_list_index += 1
                    protein_name = line[1:]
                    data_num += 1
                    protein_node_feat, protein_edge_index, protein_name, protein_sequence = protein_graphs[protein_name]
                    if protein_node_feat is None:
                        print("protein_node_feat error:\t" + line)
                elif sum([item in AMINOACID for item in line]) == len(line):
                    protein_sequence = line
                    protein_length = len(line)
                elif len(line) == 1 and line in ['0', '1']:
                    y = int(line)
                elif len(line.split(',')) in [256, 269]:
                    domain_vector = [float(item) for item in line.split(',')]
                else:
                    print("input data error:\t" + line)
            if data_num - 1 == data_list_index:
                data_list.append(
                    Data(
                        x=protein_node_feat,
                        edge_index=protein_edge_index,
                        name=protein_name,
                        sequence=protein_sequence,
                        length=protein_length,
                        vector=domain_vector,
                        y=torch.tensor(float(y), dtype=torch.float),
                    )
                )
            else:
                print("data num error:\t" + line)
            if data_type == 'train':
                # then split the data and store them for later reuse without running the preprocessing pipeline
                train_data, train_slices = self.collate(data_list)
                torch.save((train_data, train_slices), self.processed_paths[0])
            if data_type == 'valid':
                val_data, val_slices = self.collate(data_list)
                torch.save((val_data, val_slices), self.processed_paths[1])
            if data_type == 'test':
                test_data, test_slices = self.collate(data_list)
                torch.save((test_data, test_slices), self.processed_paths[2])
            if data_type == 'independent':
                test_data, test_slices = self.collate(data_list)
                torch.save((test_data, test_slices), self.processed_paths[3])

    def process(self):
        for data_type in ['train', 'valid', 'test', 'independent']:
            print(data_type)
            self.torch_save_pt(data_type)


class ToxProteinDataModule:
    def __init__(self, folder_name):
        self.train = ToxProteinDataset(folder_name, 0)
        self.val = ToxProteinDataset(folder_name, 1)
        self.test = ToxProteinDataset(folder_name, 2)
        self.independent = ToxProteinDataset(folder_name, 3)

    def train_dataloader(self):
        print("Load Training Set !")
        return DataLoader(self.train, batch_size=batch_size, shuffle=True)

    def valid_dataloader(self):
        print("Load Validation Set !")
        return DataLoader(self.val, batch_size=batch_size, shuffle=True)

    def test_dataloader(self):
        print("Load Test Set !")
        return DataLoader(self.test, batch_size=batch_size, shuffle=True)

    def independent_dataloader(self):
        print("Load Independent Test Set !")
        return DataLoader(self.independent, batch_size=batch_size, shuffle=True)
