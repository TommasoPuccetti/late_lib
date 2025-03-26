from abc import ABC, abstractmethod
from loader import PathManager


class Evaluator(ABC):
    
    def __init__(self, paths: PathManager):
        self.overall = {}
        self.results_p = paths.results_p

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """Method to be implemented by subclasses"""
        pass

    def check_if_out_path_is_given(self, results_p):
        #if the path is not provided by argument take the one in object param.
        if results_p == None:
            results_p = self.results_p
        return results_p       

        