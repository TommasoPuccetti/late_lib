import numpy as np
import pandas as pd 
import os
from config import *


#TODO a function that takes your data_frame and produces dataset files needed for the evaluation method.

class PathManager:
    """Initialize the paths to files needed for evaluation.
     Attributes:
         root: data root folder.
         dataset: name of the dataset.
         models: name of the model.
         dataset_root: points to dataset folder e.g. (./data/dos_mqtt_iot).
         files: points to the "dataset" folder of dataset_root (contains the dataset and label folder)
         models: points to the folder where folders dedicated to models are stored.
         target_model: points to a specific dataset folder (contains model and prediction folder)
         test_y_p: points to test binary label.
         test_multi_p: points to test multi-class label.
         timestamp_test_p: points to test timestamp.
         test_seq_p: points to test sequences indices.
         preds_proba_p: points to model prediction probabilities.
         preds_p: points to model binary predictions.
    """
    def __init__(self, dataset, model, root=ROOT, verbose=False):
        self.root = root
        self.dataset = dataset
        self.model = model
        self.dataset_root = os.path.join(self.root, self.dataset)
        self.files = os.path.join(self.dataset_root, "dataset")
        self.models = os.path.join(self.dataset_root, "models")
        self.target_model = os.path.join(self.models, self.model)
        if verbose:
            print_paths()
    
    @property
    def test_y_p(self):
        return os.path.join(self.files, "labels/test_y.npy")
    @property
    def test_multi_p(self):
        return os.path.join(self.files, "labels/test_multi_label.npy")
    @property
    def test_timestamp_p(self):
        return os.path.join(self.files, "labels/test_timestamp.npy")
    @property
    def test_seq_p(self):
        return os.path.join(self.files, "labels/test_sequences.pkl")
    @property
    def preds_proba_p(self):
        return os.path.join(self.target_model, "preds/preds_proba.npy")
    @property
    def preds_p(self):
        return os.path.join(self.target_model, "preds/preds.npy")
    @property
    def results_p(self):
        return os.path.join(self.target_model, "results")
    @property
    def ntest_csv_p(self):
        return os.path.join(self.files, "raw/converted/normal/merged_test_normal.csv")
    @property
    def ntrain_csv_p(self):
        return os.path.join(self.files, "raw/converted/normal/merged_test_normal.csv")
    @property
    def atest_csvs_p(self):
        return os.path.join(self.files, "raw/converted/attacks/test")
    @property
    def atrain_csvs_p(self):
        return os.path.join(self.files, "raw/converted/attacks/train")
    
    def print_paths():
        print("Dataset:  {}, \nModel:  {}".format(preds, self.model))
        print("Dataset path:  {}, \nModel path:  {}".format(self.files, self.target_model))
        print("test_y path:  {}, \ntest_seq path:  {}, \ntest_multi path {}".format(test_y_p, test_seq_p, test_multi_p))
        print("preds_proba path:  {}, \npreds path:  {}".format(preds_proba_p, preds))
        print("Converted normal test path:  {}, \ntrain path:  {}".format(ntrain_csv_p, ntest_csv_p))
        print("Converted attacks test path:  {}, \ntrain path:  {}".format(atrain_csv_p, atest_csv_p))
        print(PRINT_SEPARATOR)


class DataLoader:
    """Load files needed for evaluation.
     Attributes: 
         path_manager: an instance of the PathManager class that points to data to load.
         test_y: .npy binary label.
         test_multi: .npy multi-class label.
         timestamp_test: .npy test timestamp.
         test_seq: .npy test sequences indices.
         preds_proba: .npy prediction probabilities of the model.
         preds: .npy binary predictions of the model.
    """
    def __init__(self, paths: PathManager):
        self.paths = paths
    
    @property
    def test_y(self):
        return np.load(self.paths.test_y_p, allow_pickle=True)
    @property
    def test_multi(self):
        return np.load(self.paths.test_multi_p, allow_pickle=True)
    @property
    def test_timestamp(self):
        return np.load(self.paths.test_timestamp_p, allow_pickle=True)
    @property
    def test_seq(self):
        return np.load(self.paths.test_seq_p, allow_pickle=True)
    @property
    def preds_proba(self):
        return np.load(self.paths.preds_proba_p, allow_pickle=True)
    @property
    def preds(self):
        return np.load(self.paths.preds_p, allow_pickle=True)

