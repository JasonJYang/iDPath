import heapq
import numpy as np
from sklearn import metrics

def get_auc(drug_score_dict, pos_test_drugs):
    drug_score_list = sorted(drug_score_dict.items(), key=lambda kv: kv[1]) # sort from small to large
    drug_score_list.reverse() # from large to small
    drug_sort = [item[0] for item in drug_score_list]
    posterior = [item[1] for item in drug_score_list] # score for each drug

    r = []
    for drug in drug_sort:
        if drug in pos_test_drugs:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.roc_auc_score(y_true=r, y_score=posterior)
    return auc


def ranklist_by_sorted(pos_test_drugs, test_drugs, rating_dict, Ks):
    drug_score_dict = {drug: rating_dict[drug] for drug in test_drugs}
    K_max = max(Ks) # max k value
    # heapq.nlargest is used to find the n largest velements in a dataset
    K_max_drug_score = heapq.nlargest(K_max, drug_score_dict, key=drug_score_dict.get)
    r = []
    for i in K_max_drug_score:
        if i in pos_test_drugs:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(drug_score_dict=drug_score_dict, pos_test_drugs=pos_test_drugs)
    return r, auc


def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)

def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num

def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.

def get_metrics(pos_test_drugs, r, auc, Ks):
    rec_precision, rec_recall, ndcg, hit_ratio = [], [], [], []
    for K in Ks:
        rec_precision.append(precision_at_k(r, K))
        rec_recall.append(recall_at_k(r, K, len(pos_test_drugs)))
        ndcg.append(ndcg_at_k(r, K))
        hit_ratio.append(hit_at_k(r, K))
    return {'precision': np.array(rec_precision), 'recall': np.array(rec_recall),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_disease(rating_dict, test_drugs, pos_test_drugs, Ks):
    '''
    rating_dict: key is the drug and the value is the corresponding prediction value
    test_drugs: total drugs in the test dataset
    pos_test_drugs: for one disease, the positive drugs in the test dataset
    Ks: a list of K values
    '''
    r, auc = ranklist_by_sorted(pos_test_drugs=pos_test_drugs, test_drugs=test_drugs,
                                rating_dict=rating_dict, Ks=Ks)
    return get_metrics(pos_test_drugs=pos_test_drugs, r=r, auc=auc, Ks=Ks)

