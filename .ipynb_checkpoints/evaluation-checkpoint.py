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
from evaluator_latency import LatencyEvaluator
from evaluator_sota import SotaEvaluator
import warnings
from config import *

warnings.filterwarnings('ignore')

# ## Imports

#TODO add arguments from command line
#TODO take dataset path from command line 
#TODO take model from command line
#TODO implement a verbose output 

# Load Dataset and Model for evaluation 
pm = ld.PathManager('dos_mqtt_iot', 'xgb')
dl = ld.DataLoader(pm)
latency = LatencyEvaluator(pm)
sota = SotaEvaluator(pm)

# ## Evaluate overall performance:  (metrics balanced by model)
sota.evaluate(dl.test_y, dl.test_multi, dl.preds_proba)

# Plot curves
#ev.plot_curves(dl.test_y, dl.preds_proba)

# Evaluate FPR / Latency trade-off
avg_results, tradeoff_summary = latency.evaluate(dl.test_y, dl.test_multi, dl.test_timestamp, dl.test_seq, dl.preds_proba)

print(avg_results)
print(PRINT_SEPARATOR)
print(tradeoff_summary[0])
print(PRINT_SEPARATOR)
print(tradeoff_summary[1])


    



