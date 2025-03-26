"""

This module evaluates an Intrusion Detection model by computing  standard classification metrics and the time an attack remains undetected. The evaluation includes the following key metrics:

- **False Positive Rate (FPR):** The proportion of normal instances misclassified as attacks.
- **Attack Latency (Δl):** The time taken to detect each attack sequence.
- **Sequence Detection Rate (SDR):** The number of detected attack sequences over the total attack sequences.

### Evaluation Approach
To measure the average latency (ΔL) and SDR at different FPR levels, the evaluation keeps track of:

1. **Initial Data Point of Each Attack Sequence:** Marks the start of each attack.
2. **Positions of Data Points Labeled as Anomalous:** Tracks where the model detects anomalies.
3. **First Correctly Classified Anomalous Data Point in Each Sequence:** Determines when an attack is first detected.

These metrics are only meaningful if the dataset consists of sequences containing normal and anomalous operations.

### Main Evaluation Functions

#### eval_sota()
Evaluates the detector using standard classification metrics, including confusion matrices.
(See implementation: [[evaluator.py#eval_sota]])

#### plot_curves()
Plots precision-recall and ROC curves to visualize model performance.
(See implementation: [[evaluator.py#plot_curves]])

#### eval_fpr_latency()
Computes attack latency for each attack sequence at different FPR thresholds.
(See implementation: [[evaluator.py#eval_fpr_latency]])

#### avg_fpr_latency()
Calculates the average attack latency per attack type and overall.
(See implementation: [[evaluator.py#avg_fpr_latency]])

#### summary_fpr_latency()
Summarizes the average attack latency for different attack types and overall performance.
(See implementation: [[evaluator.py#summary_fpr_latency]])
"""

import os
import numpy as np
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
from config import *
from loader import PathManager
import warnings
import results_handler as rh 

warnings.filterwarnings('ignore')


class Evaluator:
    def __init__(self, paths: PathManager):
        self.overall = {}
        self.results_p = paths.results_p
        
    # === eval_sota ===
    def eval_sota(self, test_y, preds):
        
        """
        Computes classification metrics for evaluating an intrusion detection model.

        #### Args:
        ----------
        - `test_y (np.ndarray)`: Ground truth labels (0 for normal, 1 for anomalous).
        - `preds (np.ndarray)`: Predicted labels from the model (0 for normal, 1 for anomalous).

        #### Returns:
        ----------
        dict: A dictionary containing the following metrics:
            
        - `accuracy (float)`: Overall classification accuracy.
        - `recall (float)`: Proportion of actual positives correctly identified.
        - `precision (float)`: Proportion of predicted positives that are actually positive.
        - `f1_score (float)`: Harmonic mean of precision and recall.
        - `fpr (float)`: False positive rate (FP / (FP + TN)).
        - `tn, fp, fn, tp (int)`: Confusion matrix values.
        """ 
    
        fp, fn, tp (int): confusion matrix values.
        acc = sklearn.metrics.accuracy_score(test_y, preds)
        rec = sklearn.metrics.recall_score(test_y, preds)
        prec = sklearn.metrics.precision_score(test_y, preds)
        f1 = sklearn.metrics.f1_score(test_y, preds)
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(test_y, preds).ravel()
        fpr = fp / fp + tn
        
        sota_results = rh.store_sota_results(acc, rec, prec, f1, fpr, tn, fp, fn, tp)
        self.overall = sota_results 
        
        return sota_results
    
    def plot_roc(self, test_y, preds_proba, results_p=None):
        if results_p == None:
            results_p = self.results_p
        plt.figure(dpi=400)
        fpr, tpr, _ = metrics.roc_curve(test_y,  preds_proba[:,1])
        plt.plot(fpr, tpr)
        plt.ylabel('Recall')
        plt.xlabel('False Positive Rate')
        plt.savefig(os.path.join(results_p, "roc_curve,pdf"), format='pdf', bbox_inches='tight')
        plt.show()

    def plot_precision_recall(self, test_y, preds_proba, results_p=None):
        if results_p == None:
            results_p = self.results_p
        plt.figure(dpi=400)
        fpr, tpr, _ = metrics.precision_recall_curve(test_y,  preds_proba[:,1])
        plt.plot(fpr, tpr)
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.savefig(os.path.join(results_p, "precision_recall_curve,pdf"), format='pdf', bbox_inches='tight')
        plt.show()

    def plot_curves(self, test_y, preds_proba, results_p=None):
        if results_p == None:
            results_p = self.results_p
        self.plot_precision_recall(test_y, preds_proba, results_p)
        self.plot_roc(test_y, preds_proba, results_p)

    def check_if_out_path_is_given(self, results_p):
        #if the path is not provided by argument take the one in object param.
        if results_p == None:
            results_p = self.results_p
        return results_p    

    def bin_preds_for_given_fpr(self, test_y, preds_proba, desired_fprs, verbose=False):
        #compute the roc curve using model prediction probabilities 
        #select the index of the fpr to consider, find the thresholds for the desired FPRs, and compute binary predictions based on FPR thresholds 
        fpr, tpr, thresholds = metrics.roc_curve(test_y, preds_proba[:,1])
        fpr_indexes = [np.argmax(fpr > val) for val in desired_fprs] 
        fpr_thresholds = thresholds[fpr_indexes]
        bin_preds_fpr = [(preds_proba > val).astype(int)[:, 1] for val in fpr_thresholds]

        if verbose:
            print(bin_preds_fpr)
        
        return bin_preds_fpr

    def atk_sequence_from_seq_idxs(self, test_y, bin_pred, seq, last):
        seq_y = test_y[seq]
        seq_preds = np.array(bin_pred[last: last + seq_y.shape[0]])
        y_test_atk = np.where(seq_y == 1)[0]
        if len(y_test_atk) == 0:
            last += len(seq_y)

        return seq_y, seq_preds, y_test_atk, last

    def eval_sequence_latency(self, seq, y_test_atk, test_timestamp, seq_preds):
        # Compute attack timing
        attack_start_idx = seq[y_test_atk[0]]
        attack_end_idx = seq[y_test_atk[-1]]
        attack_time = test_timestamp[attack_end_idx] - test_timestamp[attack_start_idx]
    
        # Detect first attack occurrence
        if 1 in seq_preds[y_test_atk]:
            index_rel = np.where(seq_preds[y_test_atk] == 1)[0][0]
            index_abs = seq[y_test_atk[index_rel]]
            detection_time = test_timestamp[index_abs] - test_timestamp[attack_start_idx]
            detected = 1
        else:
            index_rel = len(seq) - 1
            index_abs = seq[y_test_atk[index_rel]]
            detection_time = attack_time  # If undetected, assign full attack time
            detected = 0

        latency_seq_res = {
            "atk_start_idx": attack_start_idx,
            "atk_end_idx": attack_end_idx,
            "atk_time": attack_time,
            "det_idx_rel": index_rel,
            "det_idx_abs": index_abs,
            "det_time": detection_time,
            "det": detected
        }

        return latency_seq_res

    def eval_all_attack_sequences(self, test_y, test_multi, test_timestamp, test_seq, bin_pred, desired_fpr, results_p, verbose):
        sequences_results = rh.init_sequence_results_dict()
        last = 0  
        for i, seq in enumerate(test_seq):
            seq_y, seq_preds, y_test_atk, last = self.atk_sequence_from_seq_idxs(test_y, bin_pred, seq, last)
            seq_sota_eval = self.eval_sota(seq_y, seq_preds)
            latency_seq_res = self.eval_sequence_latency(seq, y_test_atk, test_timestamp, seq_preds)
            sequences_results = rh.store_sequence_results(sequences_results, latency_seq_res, seq_sota_eval, y_test_atk, test_multi, desired_fpr)
            last += len(seq_y)
        if verbose: 
            sequences_results.to_csv(os.path.join(results_p,  str(desired_fpr) + '.csv'), index=None)
        return sequences_results
        
    def eval_fpr_latency(self, test_y, test_multi, test_timestamp, test_seq, preds_proba, desired_fprs=DESIRED_FPRS, results_p=None, verbose=False):
        
        results_p = self.check_if_out_path_is_given(results_p)

        bin_preds_fpr = self.bin_preds_for_given_fpr(test_y, preds_proba, desired_fprs, verbose)

        sequences_results_fprs = []
        for bin_pred, des_fpr in zip(bin_preds_fpr, desired_fprs):
            sequences_results = self.eval_all_attack_sequences(test_y, test_multi, test_timestamp, test_seq, bin_pred, des_fpr, results_p, verbose)
            sequences_results_fprs.append(sequences_results)
            
        return sequences_results_fprs

    def avg_fpr_latency(self, sequences_results, results_p=None):
        #if the path is not provided by argument take the one in object param.
        if results_p == None:
            results_p = self.results_p
            
        for df in sequences_results: 
            num_seq = df.shape[0]
            # calculate time_to_detect (attack latency) for all the detected sequences 
            df_detect = df[df['detected'] != 0]
            df_detect['time_to_detect'] = pd.to_timedelta(df['time_to_detect']).dt.total_seconds()
            #calculate sequence detection rate
            grouped_df = df_detect.groupby('attack_type')
            grouped_df_det = df_detect.groupby('attack_type').size().reset_index(name='count_det')
            grouped_df_tot = df.groupby('attack_type').size().reset_index(name='count_tot')

            detection_rate_df = pd.merge(grouped_df_det, grouped_df_tot, on='attack_type', how='outer')
            detection_rate_df['count_ratio'] = detection_rate_df['count_det'] / detection_rate_df['count_tot']
            target_fpr = str(df['target_fpr'].unique()[0])
            detection_rate_df['target_fpr'] = target_fpr

            avg_result_df = rh.store_results_for_attack_type(grouped_df)
            all_results_df = rh.store_overall_results(num_seq, target_fpr, df_detect)
            rh.all_latency_results_to_excel(results_p, target_fpr, df, df_detect, avg_result_df, detection_rate_df, all_results_df)
    
    def summary_fpr_latency(self, results_p=None):
        results_p = self.check_if_out_path_is_given(results_p)
        files = os.listdir(results_p)
        xlsx_files = [file for file in files if file.endswith('.xlsx')]
    
        df_out = pd.DataFrame()
        rows_fpr = []
        rows_sdr = []
        for file in xlsx_files:
            df_fpr = pd.read_excel(os.path.join(results_p, file) , sheet_name='avg_results_for_attack_type')
            df_sdr = pd.read_excel(os.path.join(results_p, file) , sheet_name='detection_rate_for_attack_type')
            target_fpr = df_sdr['target_fpr'].unique()[0]
            
            df_fpr_out = df_fpr.set_index('attack_type_').T
            selected_row = df_fpr_out.loc['time_to_detect_mean']
            selected_row = selected_row.to_frame().T
            selected_row['target_fpr'] = [target_fpr]
            rows_fpr.append(selected_row)

            df_sdr_out = df_sdr.set_index('attack_type').T
            selected_row = df_sdr_out.loc['count_ratio']
            selected_row = selected_row.to_frame().T
            selected_row['target_fpr'] = [target_fpr]
            rows_sdr.append(selected_row)

        df_fpr_out = pd.concat(rows_fpr, ignore_index=True)
        df_sdr_out = pd.concat(rows_sdr, ignore_index=True)
        print(df_fpr_out)
        print(df_sdr_out)
        rh.summary_fpr_latency_sdr_to_excel(results_p, df_fpr_out, df_sdr_out)



