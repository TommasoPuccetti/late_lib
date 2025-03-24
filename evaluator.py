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
    """ Evaluate the performance of a detector given prediction probabilities and labels.
     Evaluator can perform:
     - overall evaluation: it returns the evaluation calculated by the  
     Attributes:
         results_p: data output path to store the results of the evaluation.
    """
    def __init__(self, paths: PathManager):
        self.overall = {}
        self.results_p = paths.results_p

    def eval_sota(self, test_y, preds):
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
        if len(y_test_atk) == 0:  # No attacks in this sequence
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
            
        for fpr_result in sequences_results: 
            print("---------------------------------")
            print(len(sequences_results))
            print("---------------------------------")
            print(fpr_result)
            df = fpr_result
            # calculate time_to_detect (attack latency) for all the detected sequences 
            df_detect = df[df['detected'] != 0]
            df_detect['time_to_detect'] = pd.to_timedelta(df['time_to_detect']).dt.total_seconds()
            
            #calculate sequence detection rate
            grouped_df = df_detect.groupby('attack_type')
            print(grouped_df)
            
            grouped_df_det = df_detect.groupby('attack_type').size().reset_index(name='count_det')
            grouped_df_tot = df.groupby('attack_type').size().reset_index(name='count_tot')
            print(grouped_df_tot)

            detection_rate_df = pd.merge(grouped_df_det, grouped_df_tot, on='attack_type', how='outer')
            detection_rate_df['count_ratio'] = detection_rate_df['count_det'] / detection_rate_df['count_tot']
            target_fpr = str(df['target_fpr'].unique()[0])
            detection_rate_df['target_fpr'] = target_fpr
            #print(detection_rate_df)
        
            # Calculate average, min, max, and std for each group
            avg_result_df = grouped_df.agg({
                'attack_len': ['mean', 'min', 'max', 'std'],
                'fpr': 'mean',
                'pr': 'mean',
                'rec': 'mean',
                'time_to_detect': ['mean', 'min', 'max', 'std'],
                'idx_detection_rel': ['mean', 'min', 'max', 'std']
            }).reset_index()

    
            
            # Calculate aggregated statistics
            result_dict = {
                'detected_sequences': len(df_detect),
                'percent_detected_sequences':(df.shape[0] - len(df_detect))/len(df_detect),
                'avg_time_to_detect': df_detect['time_to_detect'].mean(),
                'std_time_to_detect': df_detect['time_to_detect'].std(),
                'min_time_to_detect': df_detect['time_to_detect'].min(),
                'max_time_to_detect': df_detect['time_to_detect'].max(),
                'avg_idx_detection_rel': df_detect['idx_detection_rel'].mean(),
                'std_idx_detection_rel': df_detect['idx_detection_rel'].std(),
                'min_idx_detection_rel': df_detect['idx_detection_rel'].min(),
                'max_idx_detection_rel': df_detect['idx_detection_rel'].max(),
                'target_fpr':df['target_fpr']}
            
            all_result_df = pd.DataFrame([result_dict])
        
            avg_result_df.columns = ['_'.join(col) for col in avg_result_df.columns]
            
            # Save the result and another DataFrame in the same Excel file
            with pd.ExcelWriter(os.path.join(results_p,  target_fpr + '_all.xlsx'), engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='sequences_results')
                df_detect.to_excel(writer, index=False, sheet_name='detected_sequences_results')
                avg_result_df.to_excel(writer, index=False, sheet_name='avg_results_for_attacks_type')
                detection_rate_df.to_excel(writer, index=False, sheet_name='attacks_detection_rate')
                all_result_df.to_excel(writer, index=False, sheet_name='all_summary')

    
    def summary_fpr_latency(self, results_p=None):
        #if the path is not provided by argument take the one in object param.
        if results_p == None:
            results_p = self.results_p
        
        files = os.listdir(results_p)
        print(results_p)

        xlsx_files = [file for file in files if file.endswith('.xlsx')]
        print(xlsx_files)
        print("########################################################################")
        df_out = pd.DataFrame()
        rows_fpr = []
        rows_sdr = []
        for file in xlsx_files:
            print('---------------------------')
            df_fpr = pd.read_excel(os.path.join(results_p, file) , sheet_name='avg_results_for_attacks_type')
            df_sdr = pd.read_excel(os.path.join(results_p, file) , sheet_name='attacks_detection_rate')
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

        # Save the result and another DataFrame in the same Excel file
        with pd.ExcelWriter(os.path.join(results_p,  'final/final_results.xlsx'), engine='xlsxwriter') as writer:
            df_fpr_out.to_excel(writer, index=False, sheet_name='fpr_latency_tradeoff')
            df_sdr_out.to_excel(writer, index=False, sheet_name='fpr_sdr_tradeoff')


