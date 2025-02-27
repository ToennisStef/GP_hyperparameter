import numpy as np

class DataPoints:
    def __init__(self, 
                 x=None, 
                 y=None
                 ):
        if x is None:
            self.x = np.random.uniform(-2.5, 2.5)
        else:
            self.x = x
        if y is None:
            self.y = np.random.uniform(-2.5, 2.5)
        else:
            self.y = y
