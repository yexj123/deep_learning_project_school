"""
    Copyright (C) 2022 Francesca Meneghello
    contact: meneghello@dei.unipd.it
    ... [License Text] ...
"""

import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

def convert_to_number(lab, csi_label_dict):
    lab_num = np.argwhere(np.asarray(csi_label_dict) == lab)[0][0]
    return lab_num

def create_windows(csi_list, labels_list, sample_length, stride_length):
    csi_matrix_stride = []
    labels_stride = []
    for i in range(len(labels_list)):
        csi_i = csi_list[i]
        label_i = labels_list[i]
        len_csi = csi_i.shape[1]
        for ii in range(0, len_csi - sample_length, stride_length):
            csi_matrix_stride.append(csi_i[:, ii:ii+sample_length])
            labels_stride.append(label_i)
    return csi_matrix_stride, labels_stride

def create_windows_antennas(csi_list, labels_list, sample_length, stride_length, remove_mean=False):
    csi_matrix_stride = []
    labels_stride = []
    for i in range(len(labels_list)):
        csi_i = csi_list[i]
        label_i = labels_list[i]
        len_csi = csi_i.shape[2]
        for ii in range(0, len_csi - sample_length, stride_length):
            csi_wind = csi_i[:, :, ii:ii + sample_length, ...]
            if remove_mean:
                csi_mean = np.mean(csi_wind, axis=2, keepdims=True)
                csi_wind = csi_wind - csi_mean
            csi_matrix_stride.append(csi_wind)
            labels_stride.append(label_i)
    return csi_matrix_stride, labels_stride

def expand_antennas(file_names, labels, num_antennas):
    file_names_expanded = [item for item in file_names for _ in range(num_antennas)]
    labels_expanded = [item for item in labels for _ in range(num_antennas)]
    stream_ant = np.tile(np.arange(num_antennas), len(labels))
    return file_names_expanded, labels_expanded, stream_ant


# ==========================================
# PyTorch Dataset Definition
# ==========================================

class CSIDataset(Dataset):
    """
    A custom PyTorch Dataset handling standard, randomized, and single-stream loading.
    """
    def __init__(self, csi_matrix_files, labels_stride, mode='standard', stream_ant=None):
        self.csi_matrix_files = csi_matrix_files
        self.labels_stride = labels_stride
        self.mode = mode
        self.stream_ant = stream_ant

    def __len__(self):
        return len(self.labels_stride)

    def __getitem__(self, idx):
        csi_file = self.csi_matrix_files[idx]
        label = self.labels_stride[idx]

        # Handle byte decoding if necessary
        if isinstance(csi_file, (bytes, bytearray)):
            csi_file = csi_file.decode()

        # Load the pickle file
        with open(csi_file, "rb") as fp:
            matrix_csi = pickle.load(fp)

        # Process based on the loading mode
        if self.mode == 'single':
            stream_a = self.stream_ant[idx]
            matrix_csi_single = matrix_csi[stream_a, ...].T
            if len(matrix_csi_single.shape) < 3:
                matrix_csi_single = np.expand_dims(matrix_csi_single, axis=-1)
            
            # Cast to PyTorch Float Tensor
            tensor_csi = torch.tensor(matrix_csi_single, dtype=torch.float32)

        else:
            if self.mode == 'randomized_antennas':
                stream_order = np.random.permutation(matrix_csi.shape[2])
                matrix_csi = matrix_csi[:, :, stream_order]
            
            # Convert to PyTorch Float Tensor and apply equivalent of tf.transpose(perm=[2, 1, 0])
            tensor_csi = torch.tensor(matrix_csi, dtype=torch.float32).permute(2, 1, 0)

        # Convert label to PyTorch Long Tensor (Standard for classification)
        tensor_label = torch.tensor(label, dtype=torch.long)

        return tensor_csi, tensor_label


# ==========================================
# DataLoader Wrappers
# ==========================================
# Note: input_shape, cache_file, and repeat variables are ignored as PyTorch 
# manages these concepts differently (e.g., repeating is handled by looping epochs)

def create_dataset(csi_matrix_files, labels_stride, input_shape=None, batch_size=32, 
                   shuffle=True, cache_file=None, prefetch=True, repeat=True, num_workers=4):
    
    dataset = CSIDataset(csi_matrix_files, labels_stride, mode='standard')
    
    # pin_memory=True acts similarly to TF's prefetch by staging data to the GPU faster
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                            num_workers=num_workers, pin_memory=prefetch)
    return dataloader


def create_dataset_randomized_antennas(csi_matrix_files, labels_stride, input_shape=None, batch_size=32, 
                                       shuffle=True, cache_file=None, prefetch=True, repeat=True, num_workers=4):
    
    dataset = CSIDataset(csi_matrix_files, labels_stride, mode='randomized_antennas')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                            num_workers=num_workers, pin_memory=prefetch)
    return dataloader


def create_dataset_single(csi_matrix_files, labels_stride, stream_ant, input_shape=None, batch_size=32, 
                          shuffle=True, cache_file=None, prefetch=True, repeat=True, num_workers=4):
    
    dataset = CSIDataset(csi_matrix_files, labels_stride, mode='single', stream_ant=list(stream_ant))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                            num_workers=num_workers, pin_memory=prefetch)
    return dataloader