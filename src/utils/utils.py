from collections import defaultdict
import pandas as pd

def parse_metrics(metrics:dict)->pd.DataFrame:
    """
    Helper to parse training/validation metrics from log
    """
    # remove metrics from sanity check
    results  = defaultdict(list)
    for res in metrics[1:]:
        #val metrics
        results['val_loss'].append(float(res['val_loss'].cpu()))
        results['val_acc'].append(float(res['val_acc'].cpu()))
        results['val_f1'].append(float(res['val_f1'].cpu()))
        results['val_precision'].append(float(res['val_precision'].cpu()))
        results['val_recall'].append(float(res['val_recall'].cpu()))
        results['val_prauc'].append(float(res['val_prauc'].cpu()))
        results['val_rocauc'].append(float(res['val_rocauc'].cpu()))
        #train metrics
        results['train_loss'].append(float(res['train_loss'].cpu()))
        results['train_acc'].append(float(res['train_acc'].cpu()))
        results['train_f1'].append(float(res['train_f1'].cpu()))
        results['train_precision'].append(float(res['train_precision'].cpu()))
        results['train_recall'].append(float(res['train_recall'].cpu()))
        results['train_prauc'].append(float(res['train_prauc'].cpu()))
        results['train_rocauc'].append(float(res['train_rocauc'].cpu()))
    return pd.DataFrame(results)