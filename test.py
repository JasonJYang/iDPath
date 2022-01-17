import os
import pickle
import argparse
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
from tqdm import tqdm
from model.rec_metric import test_one_disease
from model.model import iDPath as module_arch
from parse_config import ConfigParser

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
    valid_data_loader = data_loader.split_dataset(valid=True)
    test_data_loader = data_loader.split_dataset(test=True)
    adj = data_loader.get_sparse_adj()
    node_num = data_loader.get_node_num()
    type_num = data_loader.get_type_num()

    # build model architecture, then print to console
    model = module_arch(node_num=node_num,
                        type_num=type_num,
                        adj=adj,
                        emb_dim=config['arch']['args']['emb_dim'],
                        gcn_layersize=config['arch']['args']['gcn_layersize'],
                        dropout=config['arch']['args']['dropout'])
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

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

    def inference(temp_data_loader, display_string, save_file_name):
        total_loss = 0.0
        total_metrics = torch.zeros(len(metrics))
        save_dict = {'input': [], 'output': [], 'true_label': [],
                     'node_weight': [], 'path_weight': [],
                     'drug': [], 'disease': []}
        with torch.no_grad():
            for batch_idx, (drug, disease, path_feature, type_feature, lengths, mask, target) in enumerate(temp_data_loader):
                path_feature, type_feature = path_feature.to(device), type_feature.to(device)
                mask, target = mask.to(device), target.to(device)

                output, node_weight_normalized, path_weight_normalized = model(path_feature, type_feature, lengths, mask, gcn=False)
                loss = criterion(output, target)

                batch_size = path_feature.shape[0]
                total_loss += loss.item() * batch_size
                y_pred = torch.sigmoid(output)
                y_pred = y_pred.cpu().detach().numpy()
                y_true = target.cpu().detach().numpy()
                for i, metric in enumerate(metrics):
                    total_metrics[i] += metric(y_pred, y_true) * batch_size

                # for saving
                save_dict['input'].append(path_feature.cpu().detach().numpy())
                save_dict['output'].append(y_pred)
                save_dict['true_label'].append(y_true)
                save_dict['node_weight'].append(node_weight_normalized.cpu().detach().numpy())
                save_dict['path_weight'].append(path_weight_normalized.cpu().detach().numpy())
                save_dict['drug'].append(drug)
                save_dict['disease'].append(disease)
    
        logger.info(display_string)
        n_samples = len(temp_data_loader.sampler)
        log = {'loss': total_loss / n_samples}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples \
                for i, met in enumerate(metrics)})
        logger.info(log)

        logger.info('Save predictions...')
        with open(os.path.join(config.save_dir, save_file_name), 'wb') as f:
            pickle.dump(save_dict, f)

    Ks = config['Ks']
    def drug_repurposing():
        result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
                  'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}
        total_test_drug, total_test_disease = data_loader.get_recommendation_data()

        y_pred_list, rating_dict_list, pos_test_drugs_list = [], [], []
        drug_used_list, disease_used = [], []
        for disease in tqdm(total_test_disease):
            try:
                path_feature, type_feature, lengths, mask, target, drug_used = data_loader.create_path_for_repurposing(
                    disease=disease, total_test_drug=total_test_drug)
                path_feature, type_feature, target = path_feature.to(device), type_feature.to(device), target.to(device)
                mask = mask.to(device)
                output, _, _ = model(path_feature, type_feature, lengths, mask, gcn=False)
                y_pred = torch.sigmoid(output)
            except:
                continue

            y_pred = y_pred.cpu().detach().numpy()
            rating_dict = {drug_used[idx]: y_pred[idx] for idx in range(len(drug_used))}
            pos_test_drugs = data_loader.disease2drug_dict[disease]
            
            disease_used.append(disease)
            drug_used_list.append(drug_used)
            y_pred_list.append(y_pred)
            rating_dict_list.append(rating_dict)
            pos_test_drugs_list.append(pos_test_drugs)

            disease_metrics = test_one_disease(rating_dict=rating_dict, 
                test_drugs=drug_used, pos_test_drugs=pos_test_drugs, Ks=Ks)

            for metric_name, metric_array in result.items():
                result[metric_name] += disease_metrics[metric_name]

        for metric_name, metric_array in result.items():
            result[metric_name] = result[metric_name] / len(disease_used)

        logger.info('{} diesease have been tested.'.format(len(disease_used)))
        for idx, K in enumerate(Ks):
            log = {metric_name: metric_array[idx] for metric_name, metric_array in result.items() if metric_name != 'auc'}
            logger.info('Top {0} Predictions.'.format(K))
            logger.info(log)

        logger.info('Save...')
        result_dict = {'y_pred': y_pred_list, 'disease': disease_used, 'drug': drug_used_list,
                       'rating_dict': rating_dict_list, 'pos_test_drugs': pos_test_drugs_list}
        with open(os.path.join(config.save_dir, 'result_dict.pkl'), 'wb') as f:
            pickle.dump(result_dict, f)

    inference(temp_data_loader=data_loader, display_string='Train dataset', save_file_name='train_save_dict.pkl')
    inference(temp_data_loader=valid_data_loader, display_string='Valid dataset', save_file_name='valid_save_dict.pkl')
    inference(temp_data_loader=test_data_loader, display_string='Test dataset', save_file_name='test_save_dict.pkl')
    drug_repurposing()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)

    
        