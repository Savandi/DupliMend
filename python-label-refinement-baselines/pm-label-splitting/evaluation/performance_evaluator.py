import json
import string
from typing import TextIO

from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator  
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator           
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator  


class PerformanceEvaluator:
    """
    Input: Event Log
    Output: Performance results
    Check quality of a given model, regarding precision, fitness, generalization, simplicity
    """

    def __init__(self, net, im, fm, log, outfile: TextIO, skip_fitness=False):
        self.net = net
        self.im = im
        self.fm = fm
        self.log = log
        self.outfile = outfile
        self.precision = 0
        self.fitness = 0
        self.generalization = 0
        self.simplicity = 0
        self.skip_fitness = skip_fitness
    def _write(self, log_entry: string) -> None:
        self.outfile.write(f'{log_entry}\n')

    def evaluate_performance(self) -> None:
        self.get_fitness()
        self.get_precision()
        self.get_simplicity()
        self.get_generalization()

    def get_fitness(self):
        token_fitness = replay_fitness_evaluator.apply(self.log, self.net, self.im, self.fm,
                                                       variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
        self._write('token_fitness')
        self._write(json.dumps(token_fitness))
        self.fitness = token_fitness['average_trace_fitness']

        return self.fitness

    def get_precision(self) -> float:
        precision = precision_evaluator.apply(self.log, self.net, self.im, self.fm,
                                              variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
        self.precision = precision
        self._write('precision')
        self._write(json.dumps(precision))
        return precision

    def get_generalization(self) -> float:
        generalization = generalization_evaluator.apply(self.log, self.net, self.im, self.fm)
        self._write('generalization')
        self._write(json.dumps(generalization))
        self.generalization = generalization
        return generalization

    def get_simplicity(self):
        simplicity = simplicity_evaluator.apply(self.net)
        self.simplicity = simplicity
        self._write(json.dumps(simplicity))
        return simplicity
