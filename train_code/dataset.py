from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import utils
import numpy as np
import torch
import random
from rdkit import Chem
from scipy.spatial import distance_matrix
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import pickle
import os



def get_atom_feature(m):
    n = m.GetNumAtoms()
    H = []
    for i in range(n):
        H.append(utils.atom_feature(m, i, None, None))
    H = np.array(H)
    return H + 0


class Dataset(Dataset):

    def __init__(self, keys, data_dir, wild_dir):  #加了特征
        self.keys = keys
        self.data_dir = data_dir
        self.wild_dir = wild_dir


    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index][0]
        pssm_features = self.keys[index][1]
        labels = self.keys[index][4]

        zone = self.keys[index][5]


        if zone == 1:
            trans_wild_features = self.keys[index][2]
            trans_mut_features = self.keys[index][3]
            mol_w = Chem.MolFromPDBFile('./' + self.wild_dir + '/' + key + '_wild.pdb')
            mol_m = Chem.MolFromPDBFile('./' + self.data_dir + '/' + key + '_mutation.pdb')
        elif zone == 2:
            mol_m = Chem.MolFromPDBFile('./' + self.wild_dir + '/' + key + '_wild.pdb')
            mol_w = Chem.MolFromPDBFile('./' + self.data_dir + '/' + key + '_mutation.pdb')
            labels = -labels
            trans_wild_features = self.keys[index][3]
            trans_mut_features = self.keys[index][2]
        if mol_m != None and mol_w != None:
            H1 = get_atom_feature(mol_m)
            H2 = get_atom_feature(mol_w)
            labels = labels
            sample = {'H1': H1, \
                      'H2': H2, \
                      'pssm_features': pssm_features,\
                      'trans_wild_features': trans_wild_features,\
                      'trans_mut_features': trans_mut_features,\
                      'labels': labels
                      }
            return sample
        else:
            print(key)
            return 0


def collate_fn(batch):
    max_natoms1 = max([len(item['H1']) for item in batch if item is not None])
    max_natoms2 = max([len(item['H2']) for item in batch if item is not None])
    H1 = np.zeros((len(batch), max_natoms1, 30))
    H2 = np.zeros((len(batch), max_natoms2, 30))
    features = np.zeros((len(batch), 380, 1))
    wild_features = np.zeros((len(batch), 768, 1))
    mut_features = np.zeros((len(batch), 768, 1))
    labels = []
    for i in range(len(batch)):
        natom1 = len(batch[i]['H1'])
        natom2 = len(batch[i]['H2'])
        H1[i, :natom1] = batch[i]['H1']
        H2[i, :natom2] = batch[i]['H2']

        temp = batch[i]['pssm_features'].unsqueeze(0)
        temp = temp.permute(1, 0)
        features[i, :380, :1] = temp

        temp = batch[i]['trans_wild_features'].unsqueeze(0)
        temp = temp.permute(1, 0)
        wild_features[i, :768, :1] = temp

        temp = batch[i]['trans_mut_features'].unsqueeze(0)
        temp = temp.permute(1, 0)
        mut_features[i, :768, :1] = temp

        labels.append(batch[i]['labels'])

    H1 = torch.from_numpy(H1).float()
    H2 = torch.from_numpy(H2).float()
    features = torch.from_numpy(features).float()
    wild_features = torch.from_numpy(wild_features).float()
    mut_features = torch.from_numpy(mut_features).float()

    return H1, H2, labels, features, wild_features, mut_features
