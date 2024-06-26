from typing import Callable
import multiprocessing as mp
import yaml
import xgboost
import sklearn
import pyarrow as pa
from pyarrow import parquet, dataset
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import warnings
from random import random

class Model:
    def __init__(self, name: str, parameters: dict[str, Callable], observables: list[str], fun: Callable[[dict[str, float|complex]], dict[str, float]]):
        self.name = name
        self.parameters = parameters
        self.observables = observables
        self.fun = fun

    def __call__(self, parameters: dict[str, float|complex]) -> dict[str, float]:
        if len(set(self.parameters.keys()) - set(parameters.keys())) != 0:
            raise KeyError("Missing parameters")
        if len(set(parameters.keys()) - set(self.parameters.keys())) != 0:
            raise KeyError("Unknown parameters")
        res = self.fun(parameters)
        if len(set(self.observables) - set(res.keys())) != 0:
            raise KeyError("Missing observables")
        if len(set(res.keys()) - set(self.observables)) != 0:
            raise KeyError("Unknown observables")
        return res

    def batch(self, pars: list[dict[str, float|complex]], cores: int=1) -> list[dict]:
        if cores == 1:
            return [{'model': self.name} | o | p for p, o in zip(pars, map(self.__call__, pars))]
        else:
            with mp.Pool(cores) as pool:
                res = pool.map(self.__call__, pars)
            return [{'model': self.name} | o | p for p, o in zip(pars, res)]

    def batch_save(self, pars: list[dict[str, float|complex]], path, cores: int=1):
        h = ['model',] + [o for o in self.observables] + [p for p in self.parameters.keys()]
        headers = [pa.field("model", pa.string())] + [pa.field(o, pa.float32(), metadata={'type': 'observable'}) for o in self.observables] + [pa.field(p, pa.float32(), metadata={'type': 'parameter'}) for p in self.parameters.keys()]
        sanitized = self.name.replace("\\", "").replace(r'{', '').replace(r'}', '')
        schema = pa.schema(headers)
        df = pd.DataFrame(self.batch(pars, cores))
        table = pa.Table.from_pandas(df[h], preserve_index=False).cast(schema)
        parquet.write_table(table, f'{path}/{sanitized}.parquet')
        model_file = pathlib.Path(path)/ "_models.yaml"
        if model_file.is_file():
            with open(model_file, 'rt') as f:
                model_dict = dict(yaml.safe_load(f))
        else:
            model_dict = {}
        model_dict |= {self.name: {k: str(v) for k, v in self.parameters.items()}}
        with open(model_file, 'wt') as f:
                yaml.safe_dump(model_dict, f)
        obs_file = pathlib.Path(path)/ "_observables.yaml"
        if obs_file.is_file():
            with open(obs_file, 'rt') as f:
                obs_list = list(yaml.safe_load(f))
            for o in self.observables:
                if o not in obs_list:
                    obs_list.append(o)
                    warnings.warn(f'The observable {o} is not calculated by any other model of the current dataset at {path}', RuntimeWarning)
            for o in set(obs_list)-set(self.observables):
                warnings.warn(f'The observable {o} is missing!')
        else:
            obs_list = self.observables
        with open(obs_file, 'wt') as f:
            yaml.safe_dump(obs_list, f)

    def random_point(self, *args):
        return {p: pf(random()) for p, pf in self.parameters.items()}

    def random_dataset(self, num, path, cores: int=1):
        if cores == 1:
            data = [self.random_point() for _ in range(num)]
        else:
            with mp.Pool(cores) as pool:
                data = pool.map(self.random_point, range(num))
        self.batch_save(data, path, cores)


class Classifier:
    def __init__(self, data: pd.DataFrame, y_keyword: str ='model', model_pars={}):
        self.obsdict = {data.columns[i]: f'obs_{i}' for i in range(len(data.columns)) if data.columns[i] != y_keyword}
        self.data = data.rename(columns=self.obsdict)
        self.y_keyword = y_keyword
        self.trained = False
        self.features = [c for c in self.data.columns if c != self.y_keyword]
        self.enc = sklearn.preprocessing.LabelEncoder()
        self.data['encoded'] = self.enc.fit_transform(self.data[self.y_keyword])

        self.labels = self.enc.inverse_transform(range(len(set(self.data[self.y_keyword]) )))
        self.model = None
        self.train_df = None
        self.valid_df = None

    @classmethod
    def from_dataset(cls, path: str, model_pars={}):
        with open(pathlib.Path(path)/"_observables.yaml", 'rt') as f:
            obs_list = list(yaml.safe_load(f))
        return cls(dataset.dataset(path).to_table(columns = ['model'] + obs_list).to_pandas(), model_pars=model_pars)

    def split_datasets(self, test_split: float=0.3):
        self.train_df, self.valid_df = sklearn.model_selection.train_test_split(self.data, test_size=int(test_split*len(self.data)))
        self.scaler = sklearn.preprocessing.StandardScaler().fit(self.train_df[self.features])

    def train(self, test_split: float=0.3, **fit_pars):
        if self.train_df is None:
            self.split_datasets(test_split)
        self.model.fit(self.scaler.transform(self.train_df[self.features]), self.train_df['encoded'], **fit_pars)
        self.trained = True
        self.y_pred = self.model.predict(self.scaler.transform(self.valid_df[self.features]))

    def report(self):
        if self.trained:
            print(sklearn.metrics.classification_report(self.valid_df['encoded'], self.y_pred, target_names=self.labels))

    def confusion(self):
        if self.trained:
            plt.matshow(sklearn.metrics.confusion_matrix(self.valid_df['encoded'], self.y_pred))
            plt.xticks(range(len(self.labels)), [f'${l}$' for l in self.labels], fontsize=16)
            plt.yticks(range(len(self.labels)), [f'${l}$' for l in self.labels], fontsize=16)
            plt.colorbar()

    def predict_prob_dict(self, point: dict[str, float]):
        p = [point[k] for k in self.obsdict.keys()]
        if self.trained:
            return self.model.predict_proba(self.scaler.transform([p,]))[0]

class XClassifier(Classifier):
    def __init__(self, data: pd.DataFrame, y_keyword: str='model', model_pars={}):
        super().__init__(data, y_keyword, model_pars)
        if len(self.labels) == 2:
            self.model = xgboost.XGBRFClassifier(objective='binary:logistic', **model_pars)
        else:
            self.model = xgboost.XGBRFClassifier(objective='multi:softprob', **model_pars)


class LikelihoodRegressor:
    def __init__(self, data: pd.DataFrame, y_keyword: str ='model', model_pars={}):
        self.obsdict = {data.columns[i]: f'obs_{i}' for i in range(len(data.columns)) if data.columns[i] != y_keyword}
        self.data = data.rename(columns=self.obsdict)
        self.y_keyword = y_keyword
        self.trained = False
        self.features = [c for c in self.data.columns if c != self.y_keyword]
        self.data = pd.get_dummies(self.data, columns=[self.y_keyword], dtype=float)
        self.dummies = []
        for c in self.data.columns:
            if c not in self.features:
                self.dummies.append(c)
        self.labels = [d[len(self.y_keyword)+1:] for d in self.dummies]
        self.model = None
        self.train_df = None
        self.valid_df = None

    @classmethod
    def from_dataset(cls, path: str, model_pars={}):
        with open(pathlib.Path(path)/"_observables.yaml", 'rt') as f:
            obs_list = list(yaml.safe_load(f))
        return cls(dataset.dataset(path).to_table(columns = ['model'] + obs_list).to_pandas(), model_pars=model_pars)

    def split_datasets(self, test_split: float=0.3):
        self.train_df, self.valid_df = sklearn.model_selection.train_test_split(self.data, test_size=int(test_split*len(self.data)))
        self.scaler = sklearn.preprocessing.StandardScaler().fit(self.train_df[self.features])

    def train(self, test_split: float=0.3, **fit_pars):
        if self.train_df is None:
            self.split_datasets(test_split)
        self.model.fit(self.scaler.transform(self.train_df[self.features]), self.train_df[self.dummies], **fit_pars)
        self.trained = True
        self.y_pred = self.model.predict(self.scaler.transform(self.valid_df[self.features]))

    def report(self):
        if self.trained:
            y_class = [[float(y0 == max(y)) for y0 in y] for y in self.y_pred]
            print(sklearn.metrics.classification_report(pd.from_dummies(self.valid_df[self.dummies]), pd.from_dummies(pd.DataFrame(y_class, columns=self.dummies)), labels=self.dummies, target_names=self.labels))

    def confusion(self):
        if self.trained:
            y_class = [[float(y0 == max(y)) for y0 in y] for y in self.y_pred]
            plt.matshow(sklearn.metrics.confusion_matrix(pd.from_dummies(self.valid_df[self.dummies]), pd.from_dummies(pd.DataFrame(y_class, columns=self.dummies))))
            plt.xticks(range(len(self.labels)), [f'${l}$' for l in self.labels], fontsize=16)
            plt.yticks(range(len(self.labels)), [f'${l}$' for l in self.labels], fontsize=16)
            plt.colorbar()

    def predict(self, point: dict[str, float]):
        p = [point[k] for k in self.obsdict.keys()]
        if self.trained:
            return self.model.predict(self.scaler.transform([p,]))[0]

    def predict_proba(self, point: dict[str, float]):
        p = [point[k] for k in self.obsdict.keys()]
        if self.trained:
            return self.model.predict(self.scaler.transform([p,]))[0]/sum(self.model.predict([p,])[0])

class XLikelihoodRegressor(LikelihoodRegressor):
    def __init__(self, data: pd.DataFrame, y_keyword: str='model', model_pars={}):
        super().__init__(data, y_keyword, model_pars)
        self.model = xgboost.XGBRegressor(objective='reg:logistic', **model_pars)

