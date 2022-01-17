from sklearn import metrics

def accuracy(y_pred, y_true):
    """
    Note that the input y_pred and y_true are both numpy array.
    Args:
        y_pred (numpy.array): 
        y_true (numpy.array): [description]
    """
    return metrics.accuracy_score(y_pred=y_pred.round(), y_true=y_true)

def precision(y_pred, y_true):
    """
    Note that the input y_pred and y_true are both numpy array.
    Args:
        y_pred (numpy.array): 
        y_true (numpy.array): [description]
    """
    return metrics.precision_score(y_pred=y_pred.round(), y_true=y_true)

def recall(y_pred, y_true):
    """
    Note that the input y_pred and y_true are both numpy array.
    Args:
        y_pred ([type]): [description]
        y_true ([type]): [description]

    Returns:
        [type]: [description]
    """
    return metrics.recall_score(y_pred=y_pred.round(), y_true=y_true)

def roc_auc(y_pred, y_true):
    """
    The values of y_pred can be decimicals, within 0 and 1.
    Args:
        y_pred ([type]): [description]
        y_true ([type]): [description]

    Returns:
        [type]: [description]
    """
    return metrics.roc_auc_score(y_score=y_pred, y_true=y_true)

def pr_auc(y_pred, y_true):
    return metrics.average_precision_score(y_score=y_pred, y_true=y_true)

def f1_score(y_pred, y_true):
    return metrics.f1_score(y_pred=y_pred.round(), y_true=y_true)

def precision_recall_curve(y_pred, y_true):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true=y_true, probas_pred=y_pred)
    prc_dict = {'precision': precision, 'recall': recall, 'thresholds': thresholds}
    return prc_dict

def roc_curve(y_pred, y_true):
    fpr, tpr, thresholds = metrics.roc_curve(y_score=y_pred, y_true=y_true, pos_label=1)
    roc_dict = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
    return roc_dict