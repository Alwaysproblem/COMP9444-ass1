import logging
logging.basicConfig(level=logging.DEBUG)

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB
from tensorWorker import TensorWorker as worker

min_budget = 10
max_budget = 20
n_iteration = 20

result_logger = hpres.json_result_logger(directory=".", overwrite=False)

Ns = hpns.NameServer(run_id = "0", host="127.0.0.1", port=None)
Ns.start()

w = worker(run_id = "0", host="127.0.0.1")
w.run(background=True)

bohb = BOHB(
    configspace=w.get_configspace(),
    run_id = "0",
    nameserver = '127.0.0.1',
    min_budget = min_budget,
    max_budget = max_budget,
    result_logger = result_logger
)

res = bohb.run(n_iterations = n_iteration)
bohb.shutdown(shutdown_workers=True)

Ns.shutdown()

id2config = res.get_id2config_mapping()
optimal_id = res.get_incumbent_id()

inc_runs = res.get_runs_by_id(optimal_id)

print('Best found configuration:', id2config[optimal_id]['config'])
print('the evaluation parameter when using the best configuration:',inc_runs[-1].info)
print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
print('A total of %i runs where executed.' % len(res.get_all_runs()))

from sss import runing
config = {
    "network" : "conv",
    "lr" : 0.001
    # "lr": 0.0004152099702206564
}
print(runing(config, 20))