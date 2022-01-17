from numpy.lib.function_base import select
import torch
import numpy as np
from scipy import sparse
from torch.utils.data import Dataset

class PathDataset(Dataset):
    def __init__(self, drug_disease_array, total_path_dict, type_dict, 
                       max_path_length=8, max_path_num=8, rng=None):
        self.drug_disease_array = drug_disease_array
        self.total_path_dict = total_path_dict
        self.type_dict = type_dict
        self.max_path_length = max_path_length
        self.max_path_num = max_path_num
        self.rng = rng

    def __len__(self):
        return len(self.drug_disease_array)

    def __getitem__(self, index):
        drug, disease, label = self.drug_disease_array[index]
        path_list = self.total_path_dict[tuple([drug, disease, label])]
        path_array_list = []
        type_array_list = []
        lengths_list = []
        mask_list = []
        for path in path_list:
            path = path[:self.max_path_length]
            pad_num = max(0, self.max_path_length - len(path))
            path_array_list.append(path + [0]*pad_num)
            type_array_list.append([self.type_dict[n] for n in path]+[0]*pad_num)
            lengths_list.append(len(path))
            mask_list.append([1]*len(path)+[0]*pad_num)
        replace = len(path_array_list) < self.max_path_num
        select_idx_list = [idx for idx in self.rng.choice(len(path_array_list), size=self.max_path_num, replace=replace)]
        path_array = np.array([path_array_list[idx] for idx in select_idx_list])
        type_array = np.array([type_array_list[idx] for idx in select_idx_list])
        lengths_array = np.array([lengths_list[idx] for idx in select_idx_list])
        mask_array = np.array([mask_list[idx] for idx in select_idx_list])

        path_feature = torch.from_numpy(path_array).type(torch.LongTensor)
        type_feature = torch.from_numpy(type_array).type(torch.LongTensor)
        label = torch.from_numpy(np.array([label])).type(torch.FloatTensor)
        lengths = torch.from_numpy(lengths_array).type(torch.LongTensor)
        mask = torch.from_numpy(mask_array).type(torch.ByteTensor)

        return drug, disease, path_feature, type_feature, lengths, mask, label

