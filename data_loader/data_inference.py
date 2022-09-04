import os
import pickle
import torch
import random
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from scipy import sparse
from pathlib import Path
from base import BaseDataLoader
from data_loader.dataset import Dataset

class PathDataLoader(BaseDataLoader):
    def __init__(self, data_dir, drug_disease_pd_dir, max_path_length=8, max_path_num=8, random_state=0):
        random.seed(0)
        self.data_dir = Path(data_dir)
        self.drug_disease_pd_dir = drug_disease_pd_dir
        self.max_path_length = max_path_length
        self.max_path_num = max_path_num
        self.seed = random_state
        self.rng = np.random.RandomState(self.seed)

        self.graph = self._data_loader()
        self.node_num = self.graph.number_of_nodes()
        self.type_dict = self._get_type_dict()
        self._load_path_dict()

        self.drug_cid2index_dict, self.disease_icd2index_dict = self._get_drug_disease_map_dict()

    def get_node_num(self):
        return self.node_num

    def get_type_num(self):
        return 4

    def get_node_map_dict(self):
        node_map_pd = pd.read_csv(self.data_dir.joinpath('network_raw', 'mapping_file', 'node_map.csv'))
        node_map_dict = {row['map']: row['node'] for _, row in node_map_pd.iterrows()}
        node_map_dict[0] = 0
        return node_map_dict
    
    def get_sparse_adj(self):
        def adj_normalize(mx):
            """Row-normalize sparse matrix"""
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sparse.diags(r_inv)
            mx = r_mat_inv.dot(mx)
            return mx
        def sparse_mx_to_torch_sparse_tensor(sparse_mx):
            """Convert a scipy sparse matrix to a torch sparse tensor."""
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
            indices = torch.from_numpy(
                np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
            values = torch.from_numpy(sparse_mx.data)
            shape = torch.Size(sparse_mx.shape)
            return torch.sparse.FloatTensor(indices, values, shape)

        print('Get sparse adjacency matrix in csr format.')
        # csr matrix, note that if there is a link from node A to B, then the nonzero value in the adjacency matrix is (A, B)
        # where A is the row number and B is the column number
        csr_adjmatrix = nx.adjacency_matrix(self.graph, nodelist=sorted(list(range(1, self.node_num+1))))

        # add virtual node (index is 0)
        virtual_col = sparse.csr_matrix(np.zeros([self.node_num, 1]))
        csr_adjmatrix = sparse.hstack([virtual_col, csr_adjmatrix])
        virtual_row = sparse.csr_matrix(np.zeros([1, self.node_num+1]))
        csr_adjmatrix = sparse.vstack([virtual_row, csr_adjmatrix])

        row_num, col_num = csr_adjmatrix.shape
        print('{} edges among {} possible pairs.'.format(csr_adjmatrix.getnnz(), row_num*col_num))

        adj = csr_adjmatrix.tocoo()
        adj = adj_normalize(adj + sparse.eye(row_num))
        adj_tensor = sparse_mx_to_torch_sparse_tensor(adj)
        return adj_tensor

    def _data_loader(self):
        print('Load graph and other basic data...')
        if self.data_dir.joinpath('processed', 'graph.pkl').is_file():
            graph = pickle.load(self.data_dir.joinpath('processed', 'graph.pkl').open('rb'))
        else:
            graph_dataset = Dataset(data_dir=os.path.join(self.data_dir, 'network_raw'))
            graph = graph_dataset.get_network()
            with self.data_dir.joinpath('processed', 'graph.pkl').open('wb') as f:
                pickle.dump(graph, f)

        return graph

    def _get_type_dict(self):
        type_mapping_dict = {'gene': 1, 'chemical': 2, 'drug': 3, 'disease': 4}
        if self.data_dir.joinpath('processed', 'type.csv').is_file():
            type_pd = pd.read_csv(self.data_dir.joinpath('processed', 'type.csv'))
            type_dict = {row['node']: type_mapping_dict[row['type']] for idx, row in type_pd.iterrows()}
        else:
            gene_map_pd = pd.read_csv(self.data_dir.joinpath('network_raw', 'mapping_file', 'gene_map.csv'))
            chemical_map_pd = pd.read_csv(self.data_dir.joinpath('network_raw', 'mapping_file', 'chemical_map.csv'))
            drug_map_pd = pd.read_csv(self.data_dir.joinpath('network_raw', 'mapping_file', 'total_drug_map.csv'))
            disease_map_pd = pd.read_csv(self.data_dir.joinpath('network_raw', 'mapping_file', 'disease_map.csv'))
            chemical_nodrug_list = list(set(chemical_map_pd['map']) - set(drug_map_pd['map']))

            gene_map_dict = {node: 'gene' for node in list(gene_map_pd['map'])}
            chemical_map_dict = {node: 'chemical' for node in chemical_nodrug_list}
            drug_map_dict = {node: 'drug' for node in list(drug_map_pd['map'])}
            disease_map_dict = {node: 'disease' for node in list(disease_map_pd['map'])}
            type_dict = {**gene_map_dict, **chemical_map_dict, **drug_map_dict, **disease_map_dict}

            type_pd = pd.DataFrame({'node': list(type_dict.keys()),
                                    'type': list(type_dict.values())})
            type_pd.to_csv(self.data_dir.joinpath('processed', 'type.csv'), index=False)

            type_dict = {node: type_mapping_dict[type_string] for node, type_string in type_dict.items()}

        return type_dict

    def _load_path_dict(self):
        print('Load path_dict.pkl ...')
        with self.data_dir.joinpath('path', 'drug_path_dict.pkl').open('rb') as f:
            drug_path_dict = pickle.load(f)
        with self.data_dir.joinpath('path', 'disease_path_dict.pkl').open('rb') as f:
            disease_path_dict = pickle.load(f)
        with self.data_dir.joinpath('test', 'test_path_dict.pkl').open('rb') as f:
            path_dict = pickle.load(f)

        # the columns are drug_pubchemcid and target_entrez
        drug_target_pd = pd.read_csv(self.data_dir.joinpath('network_raw', 'drug_target.csv'))
        # the columns are disease_icd10 and target_entrez
        disease_target_pd = pd.read_csv(self.data_dir.joinpath('network_raw', 'disease_target.csv'))

        self.drug_path_dict = drug_path_dict
        self.disease_path_dict = disease_path_dict
        self.path_dict = path_dict

        self.drug_target_pd = drug_target_pd
        self.disease_target_pd = disease_target_pd

    def _get_drug_disease_map_dict(self):
        drug_map_pd = pd.read_csv(self.data_dir.joinpath('network_raw', 'mapping_file', 'total_drug_map.csv'))
        disease_map_pd = pd.read_csv(self.data_dir.joinpath('network_raw', 'mapping_file', 'disease_map.csv'))

        drug_cid2index_dict = {row['drug']: row['map'] for _, row in drug_map_pd.iterrows() 
            if row['map'] in list(self.drug_path_dict.keys())}
        disease_icd2index_dict = {row['disease']: row['map'] for _, row in disease_map_pd.iterrows()
            if row['map'] in list(self.disease_path_dict.keys())}
        drug_cid2index_dict = {row['drug']: row['map'] for _, row in drug_map_pd.iterrows()}
        disease_icd2index_dict = {row['disease']: row['map'] for _, row in disease_map_pd.iterrows()}

        return drug_cid2index_dict, disease_icd2index_dict

    def create_data(self):
        drug_disease_pd = pd.read_csv(self.data_dir.joinpath('test', self.drug_disease_pd_dir))

        print('Start processing your input data...')
        if len(drug_disease_pd) == 1:
            for _, row in drug_disease_pd.iterrows():
                drug = row['drug_pubchemcid']
                disease = row['disease_icd10']
                if drug not in self.drug_cid2index_dict:
                    print(f'Input {drug} not in our dataset!')
                if disease not in self.disease_icd2index_dict:
                    print(f"Input {disease} not in our dataset!")
                return

        total_path_array, total_type_array = [], []
        total_lengths_array, total_mask_array = [], []
        drug_used, disease_used = [], []
        drug_index_list, disease_index_list = [], []
        for idx, row in tqdm(drug_disease_pd.iterrows()):
            drug, disease = row['drug_pubchemcid'], row['disease_icd10']
            
            if drug not in self.drug_cid2index_dict:
                print(f'Input {drug} not in our dataset!')
                if disease not in self.disease_icd2index_dict:
                    print(f"Input {disease} not in our dataset!")
                continue
            if disease not in self.disease_icd2index_dict:
                print(f'Input {disease} not in our dataset!')
                if drug not in self.drug_cid2index_dict:
                    print(f"Input {drug} not in our dataset!")
                continue
            
            drug_index = self.drug_cid2index_dict[drug]
            disease_index = self.disease_icd2index_dict[disease]
            drug_index_list.append(drug_index)
            disease_index_list.append(disease_index)

            if tuple([drug_index, disease_index]) in self.path_dict:
                path_array = self.path_dict[tuple([drug_index, disease_index])]
                path_list = []
                for p in path_array.tolist():
                    path_list.append([n for n in p if n != 0])
            else:
                drug_target_list = list(set(self.drug_target_pd[self.drug_target_pd['drug_pubchemcid']==drug_index]['target_entrez']))
                disease_target_list = list(set(self.disease_target_pd[self.disease_target_pd['disease_icd10']==disease_index]['target_entrez']))

                if len(self.drug_path_dict[drug_index]) == 0 or len(self.disease_path_dict[disease_index]) == 0:
                    print(f'Cannot find path for {drug}-{disease}')
                    continue
                drug_path_list = [self.drug_path_dict[drug_index][t]+[disease_index] \
                    for t in disease_target_list if t in self.drug_path_dict[drug_index]]
                disease_path_list = [self.disease_path_dict[disease_index][t]+[drug_index] \
                    for t in drug_target_list if t in self.disease_path_dict[disease_index]]
                disease_path_list = [path[::-1] for path in disease_path_list]
                # all path starts with drug and ends with disease
                path_list = drug_path_list + disease_path_list
                if len(path_list) == 0:
                    print(f'Cannot find enough path for {drug}-{disease}')
                    continue
                
            '''Sample path'''
            path_array_list, type_array_list, lengths, mask = [], [], [], []
            for path in path_list:
                path = path[: self.max_path_length]
                pad_num = max(0, self.max_path_length - len(path))
                path_array_list.append(path + [0]*pad_num)
                type_array_list.append([self.type_dict[n] for n in path]+[0]*pad_num)
                lengths.append(len(path))
                mask.append([1]*len(path)+[0]*pad_num)
            if tuple([drug_index, disease_index]) not in self.path_dict:
                replace = len(path_array_list) < self.max_path_num
                select_idx_list = [idx for idx in self.rng.choice(len(path_array_list), size=self.max_path_num, replace=replace)]
            else:
                select_idx_list = list(range(self.max_path_num))
            path_array = np.array([[path_array_list[idx] for idx in select_idx_list]]) # shape: [1, path_num, path_length]
            type_array = np.array([[type_array_list[idx] for idx in select_idx_list]]) # shape: [1, path_num, path_length]
            lengths_array = np.array([lengths[idx] for idx in select_idx_list])
            mask_array = np.array([mask[idx] for idx in select_idx_list])

            total_path_array.append(path_array)
            total_type_array.append(type_array)
            total_lengths_array.append(lengths_array)
            total_mask_array.append(mask_array)

            drug_used.append(drug)
            disease_used.append(disease)
        
        path_feature = torch.from_numpy(np.concatenate(total_path_array, axis=0)).type(torch.LongTensor)
        type_feature = torch.from_numpy(np.concatenate(total_type_array, axis=0)).type(torch.LongTensor)
        lengths = torch.from_numpy(np.concatenate(total_lengths_array)).type(torch.LongTensor)
        mask = torch.from_numpy(np.concatenate(total_mask_array)).type(torch.ByteTensor)
            
        return drug_index_list, disease_index_list, path_feature, type_feature, lengths, mask, drug_used, disease_used