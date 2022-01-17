import os
import pandas as pd
import networkx as nx

class Dataset(object):
    def __init__(self, data_dir, undirected=False):
        self.data_dir = data_dir
        self.undirected = undirected
        self.data_dict = self._data_loading()
        self.graph = self._network_building()

    def get_network(self):
        return self.graph

    def get_disease_drug_associations(self):
        return self.data_dict['drug_disease']

    def _data_loading(self):
        '''Network'''
        # gene regulatory network with columns as from_entrez and target_entrez
        grn_pd = pd.read_csv(os.path.join(self.data_dir, 'grn.csv'))
        # protein-protein interaction network with columns as protein1_entrez and protein2_entrez
        ppi_pd = pd.read_csv(os.path.join(self.data_dir, 'ppi.csv'))
        # protein-chemical interaction network with columns as chemical_pubchemcid and protein_entrez
        pci_pd = pd.read_csv(os.path.join(self.data_dir, 'pci.csv'))
        # chemical-chemical interaction network with columns as chemical1_pubchemcid and chemical2_pubchemcid
        cci_pd = pd.read_csv(os.path.join(self.data_dir, 'cci.csv'))
        
        '''Drug and Disease Info'''
        # the columns are target_entrez and disease_icd10
        disease_target_pd = pd.read_csv(os.path.join(self.data_dir, 'disease_target.csv'))
        # the columns are drug_pubchemcid and target_entrez
        drug_target_pd = pd.read_csv(os.path.join(self.data_dir, 'drug_target.csv'))
        # the columns are drug_pubchemcid and disease_icd10
        drug_disease_pd = pd.read_csv(os.path.join(self.data_dir, 'drug_disease.csv'))

        '''data dict'''
        data_dict = {'grn': grn_pd, 'ppi': ppi_pd, 'pci': pci_pd, 'cci': cci_pd,
                     'disease_target': disease_target_pd, 'drug_target': drug_target_pd,
                     'drug_disease': drug_disease_pd}
        return data_dict

    def display_data_statistics(self):
        print('{} nodes and {} edges in the directed biology network'.format(
            len(self.chemical_list)+len(self.gene_list), len(self.data_dict['grn'])+len(self.data_dict['ppi'])+\
                len(self.data_dict['pci'])+len(self.data_dict['cci'])))
        print('{} drugs and {} diseases in our dataset and there {} associations between them'.format(
            len(self.drug_list), len(self.disease_list), len(self.data_dict['drug_disease'])))
        print('Drugs have {} targets and disease have {} targets'.format(
            len(self.data_dict['drug_target']), len(self.data_dict['disease_target'])))
    
    def _df_column_switch(self, df_name):
        df_copy = self.data_dict[df_name].copy()
        df_copy.columns = ['from', 'target'] 
        df_switch = self.data_dict[df_name].copy()
        df_switch.columns = ['target', 'from']
        df_concat = pd.concat([df_copy, df_switch])
        df_concat.drop_duplicates(subset=['from', 'target'], inplace=True)
        return df_concat

    def _network_building(self):
        ppi_directed = self._df_column_switch(df_name='ppi')
        pci_directed = self._df_column_switch(df_name='pci')
        cci_directed = self._df_column_switch(df_name='cci')
        # the direction in grn is from-gene -> target-gene
        grn_directed = self.data_dict['grn'].copy()
        grn_directed.columns = ['from', 'target']

        if self.undirected:
            print('Creat undirected graph ...')
            drug_target = self._df_column_switch(df_name='drug_target')
            disease_target = self._df_column_switch(df_name='disease_target')
        else:
            print('Creat directed graph ...')
            # the direction in drug-target network is drug -> target
            drug_target = self.data_dict['drug_target'].copy()
            drug_target.columns = ['from', 'target']
            # here the direction in disease-target network should be disease -> target
            disease_target = self.data_dict['disease_target'].copy()
            disease_target.columns = ['target', 'from']
        
        graph_directed = pd.concat([ppi_directed, pci_directed, cci_directed, grn_directed,
                                    drug_target, disease_target])
        graph_directed.drop_duplicates(subset=['from', 'target'], inplace=True)

        graph_nx = nx.from_pandas_edgelist(graph_directed, source='from', target='target',
                                           create_using=nx.DiGraph())
        
        return graph_nx
