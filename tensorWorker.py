import tensorflow as tf
import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh

from hpbandster.core.worker import Worker

import logging 
logging.basicConfig(level=logging.DEBUG)

import os

class TensorWorker(Worker):
    def __init__(self, run_id = "0", **kwargs):
        super().__init__(run_id=run_id, **kwargs)

    def compute(self, config, budget, Work_directory = '.', *args, **kwargs):
        from sss import runing
        train_accuracy, test_accuracy = runing(config, budget)
        return ({
            "loss": 1-test_accuracy,
            "info": {
                'train accuracy' : train_accuracy,
                'dev accuracy': None,
                'test accuarcy': test_accuracy,
            }
        })

    @staticmethod
    def get_configspace():
        CS = cs.ConfigurationSpace()

        Lr = csh.UniformFloatHyperparameter('lr', 
                                lower=1e-6, upper=1e-3, default_value = '1e-4',
                                log = True)

        network = csh.CategoricalHyperparameter("network", ['onelayer', 'twolayers', 'conv'])
        CS.add_hyperparameters([Lr, network])

        return CS
        
if __name__ == "__main__":
    worker = TensorWorker()
    config = {"lr": 0.0005141272317290901, "network": "twolayers"}
    # config = {
    #     "lr": 0.001, 
    #     "network" : "onelayer"
    # }
    # for _ in [1, 2]:
    res = worker.compute(config=config, budget = 20, Work_directory = ".")
    print(res["info"])