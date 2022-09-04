import numpy as np
import pandas as pd
from tqdm import tqdm

def get_path_node_weight(node_map_dict, data, index):
    drug = node_map_dict[data['drug'][index]]
    disease = node_map_dict[data['disease'][index]]
    
    path_array = data['input'][index]
    path_weight_raw = data['path_weight'][index]
    path_weight_list = [['path_weight']+['node'+str(i) for i in range(1,9)]]
    for path_idx, path in enumerate(path_array):
        path_weight_list.append([path_weight_raw[path_idx]]+[node_map_dict[n] for n in path])
    path_weight_df = pd.DataFrame(path_weight_list)
    path_weight_df.columns = path_weight_df.iloc[0]
    path_weight_df = path_weight_df[1:]
    
    node_weight_raw = data['node_weight'][index]
    node_weight_list = [['node_weight']+['node'+str(i) for i in range(1,9)]]
    for path_idx, n_w_list in enumerate(node_weight_raw):
        node_list = [node_map_dict[n] for n in path_array[path_idx]]
        valid_node_index = [idx for idx, n in enumerate(node_list) if n != 0 and n != drug and n != disease]
        valid_node_weight_sum = sum(n_w_list[valid_node_index])
        valid_node_weight = [w/valid_node_weight_sum for w in n_w_list[valid_node_index]]
        n_w_result = [0] + valid_node_weight + [0] * (7-len(valid_node_weight))
        node_weight_list.append(['node_weight']+list(n_w_result))
    node_weight_df = pd.DataFrame(node_weight_list)
    node_weight_df.columns = node_weight_df.iloc[0]
    node_weight_df = node_weight_df[1: ]

    return path_weight_df, node_weight_df

def get_topk_path(path_weight_df, node_weight_df, K):
    path_weight_df = path_weight_df.sort_values(by=['path_weight'], ascending=False)
    path_weight_df.drop_duplicates(inplace=True)
    path_topk_pd = path_weight_df[: K]
    keep_index = list(path_topk_pd.index)

    node_topk_pd = node_weight_df.loc[keep_index].reset_index(drop=True)
    path_topk_pd = path_topk_pd.reset_index(drop=True)

    topk_str_list = []
    path_list = path_topk_pd.values.tolist()
    node_list = node_topk_pd.values.tolist()
    for i in range(len(path_list)):
        path = [n for n in path_list[i] if type(n) == str]
        node_weight = [v for v in node_list[i][1: ] if v !=0]
        path_nodeweight_str_list = [path[0]] + \
            [path[i+1]+'({:.4f})'.format(node_weight[i]) for i in range(len(node_weight))] + [path[-1]]
        topk_str_list.append(' -> '.join(path_nodeweight_str_list))
    return topk_str_list

def process_one_dataset(node_map_dict, data, K=5):    
    # top-k critical paths
    topk_path_column_names = ['path-'+str(i) for i in range(1, K+1)]
    topk_path_dict = {'path-'+str(i): [] for i in range(1, K+1)}
    for i in tqdm(range(len(data['disease']))):
        path_weight_df, node_weight_df = get_path_node_weight(node_map_dict=node_map_dict, data=data, index=i)
        topk_str_list = get_topk_path(path_weight_df=path_weight_df,
                                      node_weight_df=node_weight_df,
                                      K=K)
        for i in range(K):
            if i >= len(topk_str_list):
                topk_path_dict[topk_path_column_names[i]].append(None)
            else:
                topk_path_dict[topk_path_column_names[i]].append(topk_str_list[i])
    
    df_dict = {'drug': data['drug'].tolist(),
               'disease': data['disease'].tolist(),
               'prediction': data['output'].flatten().tolist()}
    df_dict.update(topk_path_dict)
    output_df = pd.DataFrame(df_dict)
    output_df['drug'] = output_df['drug'].map(node_map_dict)
    output_df['disease'] = output_df['disease'].map(node_map_dict)
    return output_df