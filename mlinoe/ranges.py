import math

class ParameterRange:
    def __init__(self, min, max):
        self.min = min
        self.max = max
    def __call__(self, x):
        if x < 0.5:
            return min
        else:
            return max
    def __repr__(self):
        return f'ParameterRange({self.min}, {self.max})'

class LinearParameterRange(ParameterRange):
    def __call__(self, x):
        return x*self.max + (1-x)*self.min
    def __repr__(self):
        return f'LinearParameterRange({self.min}, {self.max})'

class LogParameterRange(ParameterRange):
    def __call__(self, x):
        return math.exp(x*math.log(self.max) + (1-x)*math.log(self.min))
    def __repr__(self):
        return f'LogParameterRange({self.min}, {self.max})'
        
class SignedLogParameterRange(ParameterRange):
    def __call__(self, x):
        x0 = abs(0.5-x)*2
        return math.exp(x0*math.log(self.max) + (1-x0)*math.log(self.min))*(1-2*int(x>0.5))
    def __repr__(self):
        return f'SignedLogParameterRange({self.min}, {self.max})'