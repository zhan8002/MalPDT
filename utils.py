import copy
import array
import torch
import torch.nn as nn
import sys
import os
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from secml_malware.models.malconv import MalConv, AvastNet, FireEye
import argparse
import copy
import pefile
import lief
import random
import string
import mmap

class ExeDataset(Dataset):
    def __init__(self, fp_list, data_path, label_list, first_n_byte=2 ** 20):
        self.fp_list = fp_list
        self.data_path = data_path
        self.label_list = label_list
        self.first_n_byte = first_n_byte

    def __len__(self):
        return len(self.fp_list)

    def __getitem__(self, idx):
        try:
            with open(self.data_path+self.fp_list[idx],'rb') as f:
                tmp = [i for i in f.read()[:self.first_n_byte]]
                tmp = tmp+[256]*(self.first_n_byte-len(tmp))
        except:
            with open(self.data_path+self.fp_list[idx].lower(),'rb') as f:
                tmp = [i for i in f.read()[:self.first_n_byte]]
                tmp = tmp+[256]*(self.first_n_byte-len(tmp))

        return np.array(tmp),np.array([self.label_list[idx]]), idx


def binary_to_bytez(binary, dos_stub=False, imports=False, overlay=False, relocations=False,
                      resources=False, tls=False):

    # Write modified binary to disk
    builder = lief.PE.Builder(binary)
    builder.build_imports(imports)
    builder.build()

    bytez = array.array("B", builder.get_build()).tobytes()
    return bytez

def feature_extract(bytez, input_length):
    b = np.ones((input_length,), dtype=np.int16) * 256
    bytez = np.frombuffer(bytez[: input_length], dtype=np.uint8)
    b[: len(bytez)] = bytez
    return b