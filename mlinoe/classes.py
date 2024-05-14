from typing import Callable
import multiprocessing as mp
import yaml

class Model:
    def __init__(self, name: str, parameters: list[str], observables: list[str], fun: Callable[[dict[str, float|complex]], dict[str, float]]):
        self.name = name
        self.parameters = set(parameters)
        self.observables = set(observables)
        self.fun = fun

    def __call__(self, parameters: dict[str, float|complex]) -> dict[str, float]:
        if len(self.parameters - set(parameters.keys())) != 0:
            raise KeyError("Missing parameters")
        if len(set(parameters.keys()) - self.parameters) != 0:
            raise KeyError("Unknown parameters")
        res = self.fun(parameters)
        if len(self.observables - set(res.keys())) != 0:
            raise KeyError("Missing observables")
        if len(set(res.keys()) - self.observables) != 0:
            raise KeyError("Unknown observables")
        return res

    def batch(self, pars: list[dict[str, float|complex]], cores: int=1) -> list[dict]:
        if cores == 1:
            return [{'model': self.name, 'pars': p, 'obs': o} for p, o in zip(pars, map(self.__call__, pars))]
        else:
            with mp.Pool(cores) as pool:
                res = pool.map(self.__call__, pars)
            return [{'model': self.name, 'pars': p, 'obs': o} for p, o in zip(pars, res)]

    def batch_yaml(self, pars: list[dict[str, float|complex]], stream, cores: int=1):
        yaml.safe_dump(self.batch(pars, cores), stream)