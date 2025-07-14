from my_classes import PopulationNode
from my_classes import ResourceNode
from my_classes import Animation
import random
import numpy as np 
import matplotlib.pyplot as plt
import sys
def get_B_k_w(n, nodes, k):
    m = len(nodes) - n
    B_k = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            this_node = nodes[j+ n]
            if(k == 1):
                val = this_node.r_to_n_infection_rate_1 
            elif(k == 2):
                val = this_node.r_to_n_infection_rate_2 
            else:
                print("such a k virus doesnt exist")
                sys.exit()

            B_k[i, j] = val

    return B_k
                
def get_B_k(A, n, nodes, k):
    B_k = np.zeros((n,n))
    for i in range(n):
        this_node = nodes[i]
        for j in range(n):
            if(k == 1):
                val = this_node.beta_1 * A[i, j]
            elif(k == 2):
                val = this_node.beta_2 * A[i, j]
            else:
                print("such a k virus doesnt exist")
                sys.exit()

            B_k[i, j] = val

    return B_k
                


def get_adj_matrix(nodes, n, r=10):
    adj_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i):  # Only compute lower triangle (i > j)
            pos_i = np.array(nodes[i].position)
            pos_j = np.array(nodes[j].position)
            d = np.linalg.norm(pos_i - pos_j)

            if d < r:
                weight = np.exp(-d**2)
                adj_matrix[i, j] = weight
                adj_matrix[j, i] = weight  # Symmetric

    return adj_matrix

def get_adj_matrix_w(nodes, n, r=10):
    m = len(nodes) - n
    adj_matrix = np.zeros((m, m))
    for i in range(m):
        for j in range(i):  # Only compute lower triangle (i > j)
            pos_i = np.array(nodes[n + i].position)
            pos_j = np.array(nodes[n + j].position)
            d = np.linalg.norm(pos_i - pos_j)

            if d < r:
                weight = np.exp(-d**2)
                adj_matrix[i, j] = weight
                adj_matrix[j, i] = weight  # Symmetric

    return adj_matrix

def get_w_vec(nodes, k, m):
    total = len(nodes)
    n = total - m
    col_matrix = np.zeros((m, 1))
    for i in range(m):
        this_node = nodes[n + i] #this should be only getting the resource nodes
        if(k == 1): # layer 1 of graph
            node_x_val = this_node.v_1
        elif(k == 2):  #layer 2 of graph
            node_x_val = this_node.v_2
        col_matrix[i, 0] = node_x_val
    return col_matrix

def get_x_vec(nodes, k, n):
    col_matrix = np.zeros((n, 1))
    for i in range(n):
        this_node = nodes[i]
        if(k == 1): # layer 1 of graph
            node_x_val = this_node.v_1
        elif(k == 2):  #layer 2 of graph
            node_x_val = this_node.v_2
        col_matrix[i, 0] = node_x_val
    return col_matrix


if __name__ == "__main__":
    n = 5 # number of population nodes
    m = 2 # number of resource nodes 
    beta_1 = 0.1 # infection rate virus 1
    delta_1 = 0.1 # recovery rate 1
    beta_2 = 0.1 # infection rate virus 2
    delta_2 = 0.1 # recovery rate 2 
    v_1_init = 0.5 
    v_2_init = 0.5
    r_to_n_infection_rate_1 = 0.2
    r_to_n_infection_rate_2 = 0.3
    scaling = 5
    nodes = []
    for i in range(n):
        x_pos = random.random() * scaling
        y_pos = random.random() * scaling
        nodes.append(PopulationNode(x_pos, y_pos, beta_1, delta_1, beta_2, delta_2, v_1_init, v_2_init))

    for i in range(m):
        x_pos = random.random() * scaling
        y_pos = random.random() * scaling
        nodes.append(ResourceNode(x_pos, y_pos, r_to_n_infection_rate_1, r_to_n_infection_rate_2, v_1_init, v_2_init))

    # Create and run animation
    anim = Animation()
    t_start = 0
    sim_time = 30
    delta_t = 0.05
    length = 5
    center = 2.5
        #for each time step:
    while t_start < sim_time:
        #for each timestep solve the differential equation
        A = get_adj_matrix(nodes, n, r=10)
        # k = 1
        x_1_vec = get_x_vec(nodes, 1, n)
        w_1_vec = get_w_vec(nodes, 1, m)
        y_t_1 = np.block([
            [x_1_vec],
            [w_1_vec]
        ])
        X_y_1 = np.block([
            [np.diag(x_1_vec.flatten()), np.zeros_like(x_1_vec)],
            [np.zeros((1, x_1_vec.shape[0] + 1))]
        ])
        B_1 = get_B_k(A, n, nodes, 1)
        B_1_w = get_B_k_w(n, nodes, 1)
        C_1_w = B_1_w.T
        

        #update the position of all the nodes, and then plot them in a animation
        for i in range(len(nodes)):
            this_node = nodes[i]
            dx, dy = delta_t * this_node.x_vel, delta_t * this_node.y_vel
            this_node.update_position(dx, dy)
            #if node comes in contact with bound, phi -> -phi
            this_node.check_boundary_cross(center, length)


        #update the time 
        t_start = t_start + delta_t
        anim.update(nodes)
        plt.pause(0.01)  # Pause to see each update
