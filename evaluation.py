"""
This module evaluates an Intrusion Detection model by computing standard classification metrics and the time an attack remains undetected. The evaluation includes the following key metrics:

- **False Positive Rate (FPR):** The proportion of normal instances misclassified as attacks.
- **Attack Latency (Δl):** The time taken to detect each attack sequence.
- **Sequence Detection Rate (SDR):** The number of detected attack sequences over the total attack sequences.

### Evaluation Approach
To measure the average latency (ΔL) and SDR at different FPR levels, the evaluation keeps track of:

1. **Initial Data Point of Each Attack Sequence:** Marks the start of each attack.
2. **Positions of Data Points Labeled as Anomalous:** Tracks where the model detects anomalies.
3. **First Correctly Classified Anomalous Data Point in Each Sequence:** Determines when an attack is first detected.

These metrics are only meaningful if the dataset consists of sequences containing normal and anomalous operations.
"""

import os
import numpy as np
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd 
import loader as ld
from evaluator_latency import LatencyEvaluator
from evaluator_sota import SotaEvaluator
import warnings
from config import *
import plotter as pl

warnings.filterwarnings('ignore')

# ## Imports

#TODO add arguments from command line
#TODO take dataset path from command line 
#TODO take model from command line
#TODO implement a verbose output 
#TODO if does not exist create folder result
#TODO delete content of given results folder before beginning
# Load Dataset and Model for evaluation 
pm = ld.PathManager('dos_mqtt_iot', 'extra')
dl = ld.DataLoader(pm)

#Initialize evalautors
latency = LatencyEvaluator(pm)
sota = SotaEvaluator(pm)

# ## Evaluate overall performance:  (metrics balanced by model)
sota_results_fprs = sota.evaluate(dl.test_y, dl.test_multi, dl.preds_proba)
sota.evaluate_bin_preds(dl.test_y, dl.preds) 
# Plot curves
sota.plot_curves(dl.test_y, dl.preds_proba)
# Evaluate FPR / Latency trade-off
avg_results, tradeoff_summary = latency.evaluate(dl.test_y, dl.test_multi, dl.test_timestamp, dl.test_seq, dl.preds_proba)

print(avg_results)
print(PRINT_SEPARATOR)
print(tradeoff_summary[0])
print(PRINT_SEPARATOR)
print(tradeoff_summary[1])

pl = pl.Plotter(pm, 'try')
pl.plot()



    



