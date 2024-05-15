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

    @classmethod
    def from_files(cls, inputs: list[str], model_pars={}):
        data = []
        for inp in inputs:
            with open(inp, 'rt') as f:
                m1 = yaml.safe_load(f)
            data += [dict(v['obs']) | {'model': v['model']} for v in m1]
        return cls(pd.DataFrame(data), model_pars=model_pars)

    def train(self, test_split: float=0.3):
        self.train_df, self.valid_df = sklearn.model_selection.train_test_split(self.data, test_size=int(test_split*len(self.data)))
        self.model.fit(self.train_df[self.features], self.train_df['encoded'])
        self.trained = True
        self.y_pred = self.model.predict(self.valid_df[self.features])

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
            return self.model.predict_proba([p,])[0]

class XClassifier(Classifier):
    def __init__(self, data: pd.DataFrame, y_keyword: str='model', model_pars={}):
        super().__init__(data, y_keyword, model_pars)
        self.model = xgboost.XGBRFClassifier(objective='multi:softprob', **model_pars)