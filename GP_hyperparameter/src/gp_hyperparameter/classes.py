import numpy as np
from .funcs import forrester
class DataPoints:
    def __init__(self, 
                 x=None, 
                 y=None
                 ):
        if x is None:
            self.x = np.random.uniform(0, 1)
        else:
            self.x = x
        if y is None:
            self.y = forrester(self.x)
        else:
            self.y = y
