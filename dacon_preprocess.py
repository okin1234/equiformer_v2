import numpy as np
import pickle
from ase.io import read
import torch
from torch_geometric.data import Data
import os
from tqdm import tqdm

class dacon_data_preprocess():
    def __init__(self, train_file, test_file, save_dataset_path):
        self.train_file = train_file
        self.test_file = test_file
        print("Read train data using ase...")
        self.train_structures = read(train_file, index=':')
        print("Read test data using ase...")
        self.test_structures = read(test_file, index=':')
        self.save_dataset_path = save_dataset_path
    
    def data_parse(self, ase_atom, idx):
        pos = torch.tensor(ase_atom.get_positions(), dtype=torch.float32) ### pos
        cell = torch.tensor(ase_atom.get_cell(), dtype=torch.float32).unsqueeze(0) ### cell [1, 3, 3]
        atomic_numbers = torch.tensor(ase_atom.get_atomic_numbers(), dtype=torch.float32) ### atomic_numbers
        natoms = ase_atom.get_global_number_of_atoms() ### natoms
        tags = torch.tensor(ase_atom.get_tags(), dtype=torch.int64) ### tags
        energy = ase_atom.get_total_energy() ### y
        forces = torch.tensor(ase_atom.get_forces(), dtype=torch.float32) ### forces
        fixed = torch.zeros(natoms, dtype=torch.float32) ### fixed
        sid = idx
        fid = idx
        
        graph_data = Data(pos=pos, cell=cell, atomic_numbers=atomic_numbers, natoms=natoms, tags=tags, y=energy, forces=forces, fixed=fixed, sid=sid, fid=fid)
        return graph_data
    
    def process(self):
        print("Start train data preprocess...")
        train_data = []
        for idx, ase_atom in enumerate(tqdm(self.train_structures)):
            train_data.append(self.data_parse(ase_atom, idx))
        
        print("Start test data preprocess...")
        test_data = []
        for idx, ase_atom in enumerate(tqdm(self.test_structures)):
            test_data.append(self.data_parse(ase_atom, idx))
        
        os.makedirs(self.save_dataset_path, exist_ok=True)
        print("Save train data")
        with open(os.path.join(self.save_dataset_path, 'train.pkl'), 'wb') as f:
            pickle.dump(train_data, f)
        print("Save test data")
        with open(os.path.join(self.save_dataset_path, 'test.pkl'), 'wb') as f:
            pickle.dump(test_data, f)
        print("Done")
        
if __name__ == '__main__':
    save_dataset_path = "datasets/samsung_dacon_2023"
    train_file = "../dataset/train.xyz"
    test_file = "../dataset/test.xyz"
    dacon_data_preprocess(train_file, test_file, save_dataset_path).process()