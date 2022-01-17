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
from data_loader.path_dataset import PathDataset

class PathDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, 
                       max_path_length=8, max_path_num=8, random_state=0, recreate=False, use_disease_seed=False,
                       shuffle=True, validation_split=0.1, test_split=0.2, num_workers=1, training=True):
        random.seed(0)
        self.data_dir = Path(data_dir)
        self.max_path_length = max_path_length
        self.max_path_num = max_path_num
        self.random_state = random_state
        self.recreate = recreate
        self.use_disease_seed = use_disease_seed

        self.rng = np.random.RandomState(random_state)

        self.graph = self._data_loader()
        self.node_num = self.graph.number_of_nodes()
        self.type_dict = self._get_type_dict()
        self.path_dict = self._load_path_dict()
        self.dataset = self._create_dataset()

        super().__init__(self.dataset, batch_size, shuffle, validation_split, test_split, num_workers)

    def get_node_num(self):
        return self.node_num

    def get_type_num(self):
        return 4
    
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

    def _negative_sampling(self, positive_drug_disease_pd):
        print('Negative sampling...')
        drug_list = list(set(positive_drug_disease_pd['drug_pubchemcid']))
        disease_list = list(set(positive_drug_disease_pd['disease_icd10']))
        negative_drug_list, negative_disease_list = [], []
        
        if self.use_disease_seed:
            print('Use disease as seed.')
            for disease in disease_list:
                positive_drug_list = list(positive_drug_disease_pd[positive_drug_disease_pd['disease_icd10']==disease]['drug_pubchemcid'])
                drug_left_list = list(set(drug_list) - set(positive_drug_list))
                # random select the drugs with the same number of that of positive drugs
                negative_drug_list += random.sample(drug_left_list, min(len(positive_drug_list), len(drug_left_list)))
                negative_disease_list += [disease] * min(len(positive_drug_list), len(drug_left_list))
        else:
            print('Use drug as seed.')
            for drug in drug_list:
                positive_disease_list = list(positive_drug_disease_pd[positive_drug_disease_pd['drug_pubchemcid']==drug]['disease_icd10'])
                disease_left_list = list(set(disease_list) - set(positive_disease_list))
                # random select the diseases with the same number of that of positive diseases
                negative_disease_list += random.sample(disease_left_list, min(len(positive_disease_list), len(disease_left_list)))
                negative_drug_list += [drug] * min(len(positive_disease_list), len(disease_left_list))

        negative_pd = pd.DataFrame({'drug_pubchemcid': negative_drug_list, 
                                    'disease_icd10': negative_disease_list})
        print('For {0} drugs, {1} negative samples are generated.'.format(len(drug_list), len(negative_disease_list)))
        return negative_pd

    def _load_path_dict(self):
        if not self.recreate and self.data_dir.joinpath('processed', 'path_dict.pkl').is_file():
            print('Load existing path_dict.pkl ...')
            with self.data_dir.joinpath('processed', 'path_dict.pkl').open('rb') as f:
                path_dict = pickle.load(f)
        else:
            print('Start creating path_dict ...')
            print('Load drug_path_dict.pkl ...')
            with self.data_dir.joinpath('path', 'drug_path_dict.pkl').open('rb') as f:
                drug_path_dict = pickle.load(f)
            print('Load disease_path_dict.pkl ...')
            with self.data_dir.joinpath('path', 'disease_path_dict.pkl').open('rb') as f:
                disease_path_dict = pickle.load(f)
            # the columns are drug_pubchemcid, disease_icd10
            positive_drug_disease_pd = pd.read_csv(self.data_dir.joinpath('network_raw', 'drug_disease.csv'))
            # the columns are drug_pubchemcid and target_entrez
            drug_target_pd = pd.read_csv(self.data_dir.joinpath('network_raw', 'drug_target.csv'))
            # the columns are disease_icd10 and target_entrez
            disease_target_pd = pd.read_csv(self.data_dir.joinpath('network_raw', 'disease_target.csv'))
            # the columns are drug_pubchemcid, disease_icd10
            negative_drug_disease_pd = self._negative_sampling(positive_drug_disease_pd)
            # create path_dict
            path_dict, drug_disease_pd = self._create_path(positive_drug_disease_pd, negative_drug_disease_pd, 
                                                           drug_path_dict, disease_path_dict,
                                                           drug_target_pd, disease_target_pd)
            # save
            drug_disease_pd.to_csv(self.data_dir.joinpath('processed', 'total_drug_disease.csv'), index=False)
            with self.data_dir.joinpath('processed', 'path_dict.pkl').open('wb') as f:
                pickle.dump(path_dict, f)

        return path_dict

    def _create_path(self, positive_drug_disease_pd, negative_drug_disease_pd, 
                           drug_path_dict, disease_path_dict,
                           drug_target_pd, disease_target_pd):
        print('Create all the shortest paths between drugs and diseases...')
        positive_drug_disease_pd['label'] = [1] * len(positive_drug_disease_pd)
        negative_drug_disease_pd['label'] = [0] * len(negative_drug_disease_pd)
        drug_disease_pd = pd.concat([positive_drug_disease_pd, negative_drug_disease_pd])
        
        path_dict = dict()
        for idx, row in tqdm(drug_disease_pd.iterrows()):
            drug, disease, label = row['drug_pubchemcid'], row['disease_icd10'], row['label']
            drug_target_list = list(set(drug_target_pd[drug_target_pd['drug_pubchemcid']==drug]['target_entrez']))
            disease_target_list = list(set(disease_target_pd[disease_target_pd['disease_icd10']==disease]['target_entrez']))

            if len(drug_path_dict[drug]) == 0 or len(disease_path_dict[disease]) == 0:
                continue
            drug_path_list = [drug_path_dict[drug][t]+[disease] for t in disease_target_list if t in drug_path_dict[drug]]
            disease_path_list = [disease_path_dict[disease][t]+[drug] for t in drug_target_list if t in disease_path_dict[disease]]
            disease_path_list = [path[::-1] for path in disease_path_list]
            # all path starts with drug and ends with disease
            path_list = drug_path_list + disease_path_list
            if len(path_list) == 0:
                continue
            path_dict[tuple([drug, disease, label])] = path_list
        return path_dict, drug_disease_pd        

    def _create_dataset(self):
        print('Creating tensor dataset...') 
        drug_disease_array = list(self.path_dict.keys())

        dataset = PathDataset(drug_disease_array=drug_disease_array,
                              total_path_dict=self.path_dict,
                              type_dict=self.type_dict,
                              max_path_length=self.max_path_length,
                              max_path_num=self.max_path_num,
                              rng=self.rng)
        return dataset

    def create_path_for_repurposing(self, disease, total_test_drug):
        total_path_array, total_type_array, label = [], [], []
        total_lengths_array, total_mask_array = [], []
        drug_used = []
        for drug in total_test_drug:
            '''Find all the path'''
            drug_target_list = list(set(self.drug_target_pd[self.drug_target_pd['drug_pubchemcid']==drug]['target_entrez']))
            disease_target_list = list(set(self.disease_target_pd[self.disease_target_pd['disease_icd10']==disease]['target_entrez']))
            if len(self.drug_path_dict[drug]) == 0 or len(self.disease_path_dict[disease]) == 0:
                continue
            drug_path_list = [self.drug_path_dict[drug][t]+[disease] for t in disease_target_list if t in self.drug_path_dict[drug]]
            disease_path_list = [self.disease_path_dict[disease][t]+[drug] for t in drug_target_list if t in self.disease_path_dict[disease]]
            disease_path_list = [path[::-1] for path in disease_path_list]
            # all path starts with drug and ends with disease
            path_list = drug_path_list + disease_path_list
            
            '''Sample path'''
            path_array_list, type_array_list, lengths, mask = [], [], [], []
            for path in path_list:
                path = path[: self.max_path_length]
                pad_num = max(0, self.max_path_length - len(path))
                path_array_list.append(path + [0]*pad_num)
                type_array_list.append([self.type_dict[n] for n in path]+[0]*pad_num)
                lengths.append(len(path))
                mask.append([1]*len(path)+[0]*pad_num)
            replace = len(path_array_list) < self.max_path_num
            select_idx_list = [idx for idx in self.rng.choice(len(path_array_list), size=self.max_path_num, replace=replace)]
            path_array = np.array([[path_array_list[idx] for idx in select_idx_list]]) # shape: [1, path_num, path_length]
            type_array = np.array([[type_array_list[idx] for idx in select_idx_list]]) # shape: [1, path_num, path_length]
            lengths_array = np.array([lengths[idx] for idx in select_idx_list])
            mask_array = np.array([mask[idx] for idx in select_idx_list])

            total_path_array.append(path_array)
            total_type_array.append(type_array)
            total_lengths_array.append(lengths_array)
            total_mask_array.append(mask_array)
            if drug in self.disease2drug_dict[disease]:
                label.append([1])
            else:
                label.append([0])
            
            drug_used.append(drug)
        
        path_feature = torch.from_numpy(np.concatenate(total_path_array, axis=0)).type(torch.LongTensor)
        type_feature = torch.from_numpy(np.concatenate(total_type_array, axis=0)).type(torch.LongTensor)
        label = torch.from_numpy(np.array(label)).type(torch.FloatTensor)
        lengths = torch.from_numpy(np.concatenate(total_lengths_array)).type(torch.LongTensor)
        mask = torch.from_numpy(np.concatenate(total_mask_array)).type(torch.ByteTensor)

        return path_feature, type_feature, lengths, mask, label, drug_used

    def get_recommendation_data(self):
        '''Get the unique drug and disease in the test dataset'''
        drug_disease_array = list(self.path_dict.keys())
        test_drug_disease_array = [drug_disease_array[idx] for idx in self.test_idx]
        print('{} records in test dataset'.format(len(test_drug_disease_array)))
        total_test_drug, total_test_disease = [], []
        disease2drug_dict = dict()
        for drug, disease, label in test_drug_disease_array:
            if label == 0:
                continue
            total_test_drug.append(drug)
            total_test_disease.append(disease)
            if disease not in disease2drug_dict:
                disease2drug_dict[disease] = [drug]
            else:
                disease2drug_dict[disease].append(drug)
        total_test_drug = list(set(total_test_drug))
        total_test_disease = list(set(total_test_disease))
        self.disease2drug_dict = {disease: list(set(drug_list)) for disease, drug_list in disease2drug_dict.items()}
        '''Prepare the dataset'''
        print('Start creating path_dict for test dataset...')
        print('Load drug_path_dict.pkl ...')
        with self.data_dir.joinpath('path', 'drug_path_dict.pkl').open('rb') as f:
            self.drug_path_dict = pickle.load(f)
        print('Load disease_path_dict.pkl ...')
        with self.data_dir.joinpath('path', 'disease_path_dict.pkl').open('rb') as f:
            self.disease_path_dict = pickle.load(f)
        # the columns are drug_pubchemcid and target_entrez
        self.drug_target_pd = pd.read_csv(self.data_dir.joinpath('network_raw', 'drug_target.csv'))
        # the columns are disease_icd10 and target_entrez
        self.disease_target_pd = pd.read_csv(self.data_dir.joinpath('network_raw', 'disease_target.csv'))

        return total_test_drug, total_test_disease
        




