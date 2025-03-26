from abc import ABC, abstractmethod
from loader import PathManager
import sklearn
import results_handler as rh


class Evaluator(ABC):
    
    def __init__(self, results_p: PathManager):
        self.overall = {}
        self.results_p = results_p.results_p

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """Method to be implemented by subclasses"""
        pass

    def check_if_out_path_is_given(self, results_p):
        #if the path is not provided by argument take the one in object param.
        if results_p == None:
            results_p = self.results_p
        return results_p

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
        - `precision (float)`: Proportion of predicted positives that are positive.
        - `f1_score (float)`: Harmonic mean of precision and recall.
        - `fpr (float)`: False positive rate (FP / (FP + TN)).
        - `tn, fp, fn, tp (int)`: Confusion matrix values.
        """ 
        
        acc = sklearn.metrics.accuracy_score(test_y, preds)
        rec = sklearn.metrics.recall_score(test_y, preds)
        prec = sklearn.metrics.precision_score(test_y, preds)
        f1 = sklearn.metrics.f1_score(test_y, preds)
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(test_y, preds).ravel()
        fpr = fp / fp + tn
        
        sota_results = rh.store_sota_results(acc, rec, prec, f1, fpr, tn, fp, fn, tp)
        self.overall = sota_results 
        
        return sota_results

        