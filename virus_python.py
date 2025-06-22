import numpy as np
import random
from NodeObjectClass import NodeObject 

if __name__ == "__main__":
    print("starting program...")
    #variables
    num_nodes = 20
    area = 10

    bad_guys = [4, 6, 7, 13, 17, 18]
    node_list = []
    #initiate all the nodes
    for i in range(num_nodes):
        if(i in bad_guys):
            this_node = NodeObject(area, "Infected")
        else:
            this_node = NodeObject(area, "Susceptible")
        node_list.append(this_node)
    print(len(node_list))
