import os
import numpy as np
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
from config import *
from loader import PathManager


class Evaluator:
    """Evaluate performance of a detector given prediction probabilities and labels.
     Evaluator can perform:
     - overall evaluation: it returns the evaluation calculated by the  
     Attributes:
         results_p: data output path to store the results of the evaluation.
    """
    def __init__(self, paths: PathManager):
        self.overall = {}
        self.results_p = paths.results_p
    
    def eval_overall(self, test_y, preds):
        acc = sklearn.metrics.accuracy_score(test_y, preds)
        rec = sklearn.metrics.recall_score(test_y, preds)
        prec = sklearn.metrics.precision_score(test_y, preds)
        f1 = sklearn.metrics.f1_score(test_y, preds)
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(test_y, preds).ravel()
        overall = {
            "accuracy": acc,
            "recall": rec,
            "precision": prec,
            "f1-score": f1,
            "tn": tn,
            "fp":fp,
            "fn": fn,
            "tp": tp
        }
        self.overall = overall 
        print("Accuracy: {}, \nRecall: {}, \nPrecision: {}, \nF1-Score: {}".format(acc, rec, prec, f1))
        print("TP: {}, \nFP: {}, \nTN: {}, \nFN: {}".format(tp, fp, tn, fn))
        print(PRINT_SEPARATOR)
        return overall
    
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
                    
    def eval_fpr_latency(self, test_y, test_multi, test_timestamp, test_seq, preds_proba, desired_fpr=DESIRED_FPR, results_p=None, verbose=False):
        
        #if the path is not provided by argument take the one in object param.
        if results_p == None:
            results_p = self.results_p
        
        #compute the roc curve using the model prediction probabilities     
        fpr, tpr, thresholds = metrics.roc_curve(test_y, preds_proba[:,1])
        fpr_indexes = []
        #select the index of the fpr to consider 
        for val in desired_fpr:
            index = np.argmax(fpr > val)
            fpr_indexes.append(index)
        #find the threshold to obtain such fpr
        fpr_thresholds = thresholds[fpr_indexes] 
        bin_preds_fpr = []
        for val in fpr_thresholds:
            bin_pred = (preds_proba > val).astype(int)[:, 1]
            bin_preds_fpr.append(bin_pred)
            if verbose: print(bin_preds_fpr)

        sequences_results = []
        #store fpr values with relative binary prediction obtaine from using a specific threshold
        for val, des_fpr in zip(bin_preds_fpr, DESIRED_FPR):
            bin_pred = val
        #REFACTOR TILL HERE IN DEDICATED FUNCTION FOR FPR DEFINITION 

            fpr_results = pd.DataFrame(columns=['start_idx_attack', 'end_idx_attack', 'attack_duration', 'time_to_detect', 'idx_detection_abs', 'idx_detection_rel', 'attack_len', 'attack_type','fpr','tpr','pr','rec','tn', 'fp','fn','tp','target_fpr', 'detected'])
            last = 0
            i = 0
    
            for seq in test_seq:
                seq_y = test_y[seq]
                seq_preds = bin_pred[last: last + seq_y.shape[0]]
                seq_preds = np.array(seq_preds)
    
                seq_check = test_y[last: last + seq_y.shape[0]]
                y_test_atk= np.where(test_y[seq] == 1)[0]
                y_test_norm= np.where(test_y[seq] ==  0)[0]
    
                conf_matrix = sklearn.metrics.confusion_matrix(seq_y, seq_preds)
                #TODO CASES WHERE SOME METRICS IN CONF MATRIX ARE NOT AVAILABLE TRY CATCH
                tn, fp, fn, tp = conf_matrix.ravel()
                pr = sklearn.metrics.precision_score(seq_y, seq_preds)
                rec = sklearn.metrics.recall_score(seq_y, seq_preds)
        
                preds_1 = seq_preds[y_test_atk]
        
                date_time_0 = test_timestamp[seq[y_test_atk[0]]]
                date_time_last = test_timestamp[seq[y_test_atk[len(y_test_atk)-1]]]
                attack_time = date_time_last - date_time_0
        
                if 1 in seq_preds[y_test_atk]:
                    index_rel = np.where(seq_preds[y_test_atk] == 1)[0][0]
                    index = seq[y_test_atk[index_rel]]
        
                    date_time_index = test_timestamp[index]
                    detection_time = date_time_index - date_time_0
                    detected = 1
                else:
                    index_rel = len(seq) - 1
                    index = seq[y_test_atk[index_rel]]
                    #TODO ADD A TIME DELTA TO LATENCY WHEN SEQUENCE IS NOT DETECTED
                    detection_time = attack_time
                    detected = 0
            
                fpr_results.loc[i] = pd.Series({'start_idx_attack': seq[y_test_atk[0]], 'end_idx_attack': seq[y_test_atk[len(y_test_atk)-1]],
                                   'attack_duration': attack_time , 'time_to_detect': detection_time, 'idx_detection_abs': index,
                                   'idx_detection_rel':index_rel, 'attack_len': y_test_atk.shape[0],
                                   'attack_type':test_multi[seq[y_test_atk[0]]], 'tpr': tpr, 'pr': pr, 'rec': rec,
                                   'fp':fp, 'fn':fn, 'tp':tp, 'tn':tn, 'target_fpr':des_fpr, 'detected': detected})
                last = last+len(seq_y)
                i = i+1
                
                #LOG ALL INTERMEDIATE IF VERBOSE 
                #fpr_results.to_csv(os.path.join(results_p,  str(des_fpr) + '.csv'), index=None)
            
            sequences_results.append(fpr_results)        
        return sequences_results

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
        with pd.ExcelWriter(os.path.join(results_p,  'final_results.xlsx'), engine='xlsxwriter') as writer:
            df_fpr_out.to_excel(writer, index=False, sheet_name='fpr_latency_tradeoff')
            df_sdr_out.to_excel(writer, index=False, sheet_name='fpr_sdr_tradeoff')


