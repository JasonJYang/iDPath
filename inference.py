import os
import pickle
import argparse
import torch
import numpy as np
import data_loader.data_inference as module_data
from model.model import iDPath as module_arch
from parse_config import ConfigParser
from utils.get_path import process_one_dataset

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    adj = data_loader.get_sparse_adj()
    node_num = data_loader.get_node_num()
    type_num = data_loader.get_type_num()
    node_map_dict = data_loader.get_node_map_dict()

    # build model architecture, then print to console
    model = module_arch(node_num=node_num,
                        type_num=type_num,
                        adj=adj,
                        emb_dim=config['arch']['args']['emb_dim'],
                        gcn_layersize=config['arch']['args']['gcn_layersize'],
                        dropout=config['arch']['args']['dropout'])
    logger.info(model)

    # load trained model
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    drug_index_list, disease_index_list, path_feature, type_feature, \
        lengths, mask, drug_used, disease_used = data_loader.create_data()
    result_dict = {}
    with torch.no_grad():
        path_feature, type_feature, mask = path_feature.to(device), type_feature.to(device), mask.to(device)
        output, node_weight_normalized, path_weight_normalized = model(path_feature, type_feature, lengths, mask, gcn=False)
        y_pred = torch.sigmoid(output)

        # for saving
        result_dict['input'] = path_feature.cpu().detach().numpy()
        result_dict['output'] = y_pred.cpu().detach().numpy()
        result_dict['node_weight'] = node_weight_normalized.cpu().detach().numpy()
        result_dict['path_weight'] = path_weight_normalized.cpu().detach().numpy()
        result_dict['drug'] = np.array(drug_index_list)
        result_dict['disease'] = np.array(disease_index_list)
    
    with open(os.path.join(config.save_dir, 'result.pkl'), 'wb') as f:
        pickle.dump(result_dict, f)
    
    output_df = process_one_dataset(node_map_dict=node_map_dict, data=result_dict, K=config['K'])
    output_df.to_csv(os.path.join(config.save_dir, 'result.csv'), index=False)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-drug', default=None, type=str,
                      help='PubChem CID of the input drug, such as CID-2244 for aspirin.')
    args.add_argument('-disease', default=None, type=str,
                      help='ICD10 of the input disease, such as ICD10-C61 for prostate cancer.')

    config = ConfigParser.from_args(args)
    main(config)