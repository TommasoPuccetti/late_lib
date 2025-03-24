import pandas as pd
from config import *


def store_sota_results(acc, rec, prec, f1, fpr, tn, fp, fn, tp):
    sota_results = {
        "accuracy": acc,
        "recall": rec,
        "precision": prec,
        "f1-score": f1,
        "fpr": fpr,
        "tn": tn,
        "fp":fp,
        "fn": fn,
        "tp": tp
    }

    print("Accuracy: {}, \nRecall: {}, \nPrecision: {}, \nF1-Score: {}".format(acc, rec, prec, f1))
    print("TP: {}, \nFP: {}, \nTN: {}, \nFN: {}".format(tp, fp, tn, fn))
    print(PRINT_SEPARATOR)

    return sota_results


def init_sequence_results_dict():
    return pd.DataFrame(columns=[
        'start_idx_attack', 'end_idx_attack', 'attack_duration', 'time_to_detect',
        'idx_detection_abs', 'idx_detection_rel', 'attack_len', 'attack_type',
        'pr', 'rec', 'fpr', 'tn', 'fp', 'fn', 'tp', 'target_fpr', 'detected'])

def store_sequence_results(df, latency_seq_res, seq_sota_eval, y_test_atk, test_multi, desired_fpr):
    """
    Stores attack detection results into a DataFrame and returns the updated DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to store results.
        latency_seq_res (dict): Latency sequence results.
        seq_sota_eval (dict): Evaluation metrics.
        y_test_atk (list): Attack labels for the test set.
        test_multi (list): Attack types.
        desired_fpr (float): Target false positive rate.

    Returns:
        pd.DataFrame: Updated DataFrame with stored results.
    """
    
    new_row = {
        'start_idx_attack': latency_seq_res['atk_start_idx'],
        'end_idx_attack': latency_seq_res['atk_end_idx'],
        'attack_duration': latency_seq_res['atk_time'],
        'time_to_detect': latency_seq_res['det_time'],
        'idx_detection_abs': latency_seq_res['det_idx_abs'],
        'idx_detection_rel': latency_seq_res['det_idx_rel'],
        'attack_len': len(y_test_atk),
        'attack_type': test_multi[latency_seq_res['atk_start_idx']],
        'pr': seq_sota_eval['precision'],
        'rec': seq_sota_eval['recall'],
        'fpr': seq_sota_eval['fpr'],
        'fp': seq_sota_eval['fp'],
        'fn': seq_sota_eval['fn'],
        'tp': seq_sota_eval['tp'],
        'tn': seq_sota_eval['tn'],
        'target_fpr': desired_fpr,
        'detected': latency_seq_res['det']
    }
    
    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)