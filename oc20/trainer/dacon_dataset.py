"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


    1. Modify pyg2 data transformation:
        Since S2EF dataset is generated with PyG2, there is no need to convert again
        and the original transformation can result in nothing in Data.

"""

import bisect
import logging
import math
import pickle
import random
import warnings
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
import torch_geometric

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
#from ocpmodels.common.utils import pyg2_data_transform

import pickle

def pyg2_data_transform(data: Data):
    # if we're on the new pyg (2.0 or later), we need to convert the data to the new format
    if torch_geometric.__version__ >= "2.0":
        if '_store' not in data.__dict__:
            return Data(**{k: v for k, v in data.__dict__.items() if v is not None})

    return data


@registry.register_dataset("samsung_dacon_2023")
class DaconDataset(Dataset):
    r"""Dataset class to load from Samsung Dacon 2023 competition data.

    Args:
            config (dict): Dataset configuration
            transform (callable, optional): Data transform function.
                    (default: :obj:`None`)
    """

    def __init__(self, config, transform=None):
        super(DaconDataset, self).__init__()
        self.config = config

        self.path = Path(self.config["src"])
        self.metadata_path = self.path.parent / "metadata.npz"
        self.datas = pickle.load(open(self.path, 'rb'))
        self.num_samples = len(self.datas)
        


        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        datapoint = self.datas[idx]
        data_object = pyg2_data_transform(datapoint)

        if self.transform is not None:
            data_object = self.transform(data_object)

        return data_object


def data_list_collater(data_list, otf_graph=False):
    batch = Batch.from_data_list(data_list)

    if not otf_graph:
        try:
            n_neighbors = []
            for i, data in enumerate(data_list):
                n_index = data.edge_index[1, :]
                n_neighbors.append(n_index.shape[0])
            batch.neighbors = torch.tensor(n_neighbors)
        except NotImplementedError:
            logging.warning(
                "LMDB does not contain edge index information, set otf_graph=True"
            )

    return batch
