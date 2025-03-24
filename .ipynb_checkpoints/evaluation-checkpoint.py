#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd 
import loader as ld
from evaluator import Evaluator


# ## Imports

#TODO add arguments from command line
#TODO take dataset path from command line 
#TODO take model from command line
#TODO implement a verbose output 

# Load Dataset and Model for evaluation 
pm = ld.PathManager('dos_mqtt_iot', 'xgb')
dl = ld.DataLoader(pm)
ev = Evaluator(pm)

# ## Evaluate overall performance:  (metrics balanced by model)
#overall = ev.eval_overall(dl.test_y, dl.preds)

# Plot curves
#ev.plot_curves(dl.test_y, dl.preds_proba)

# Evaluate FPR / Latency trade-off
sequences_results = ev.eval_fpr_latency(dl.test_y, dl.test_multi, dl.test_timestamp, dl.test_seq, dl.preds_proba)
avg_results = ev.avg_fpr_latency(sequences_results)

print(avg_results)


    



