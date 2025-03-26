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
from evaluator import Evaluator

warnings.filterwarnings('ignore')


class SotaEvaluator(Evaluator):
    
    def __init__(self, results_p: PathManager):
        #self.avg_results = None
        #self.tradeoff_summary = None
        super().__init__(results_p)
     
    def evaluate(self, test_y, test_multi, preds_proba, desired_fprs=DESIRED_FPRS, results_p=None, verbose=False):
        results_p = self.check_if_out_path_is_given(results_p)
        for desired_fpr in desired_fprs:
            metrics_list = []
            attacks = np.unique(test_multi)
            atk_index_list = [np.where(test_multi == value)[0].tolist() for value in attacks]
            preds = preds_proba[:,1]
            print(preds.shape)
            fpr, tpr, thresholds = metrics.roc_curve(test_y, preds)
            precision, recall, thr = metrics.precision_recall_curve(test_y,  preds)
            
            index = np.argmax(fpr > desired_fpr)
            fpr_threshold = thresholds[index]
            
            bin_preds = (preds > fpr_threshold).astype(int)
            
            for indexes, i in zip(atk_index_list, range(0, len(atk_index_list))):
                
                acc = sklearn.metrics.accuracy_score(test_y[indexes], bin_preds[indexes])
                pr = sklearn.metrics.precision_score(test_y[indexes], bin_preds[indexes])
                rec = sklearn.metrics.recall_score(test_y[indexes], bin_preds[indexes])
                f1 = sklearn.metrics.f1_score(test_y[indexes], bin_preds[indexes])
            
                # Append metrics to the list as a dictionary
                metrics_list.append({
                        "File": str(desired_fpr),
                        "Attack": attacks[i],
                        "Accuracy": acc,
                        "Precision": pr,
                        "Recall": rec,
                        "F1 Score": f1})
                print(metrics)
            
                # Convert the list of dictionaries into a DataFrame
            df = pd.DataFrame(metrics_list)
            df.to_csv(os.path.join(results_p, str(desired_fpr) + '.csv'))
        
        




