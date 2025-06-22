import numpy as np
import random

class NodeObject:
    
    def __init__(self, area, Type):
        self.x_pos = random.random() * area
        self.y_pos = random.random() * area
        #phi variables
        self.x_vel = random.gauss(0, 1)
        self.y_vel = random.gauss(0, 1)

        self.NodeType = Type
    