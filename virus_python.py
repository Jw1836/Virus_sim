import numpy as np
import random
from AnimationClass import Node
from AnimationClass import Animation
import matplotlib.pyplot as plt
import sys
########################

def get_adj_matrix(node_list, r=10):
    n = len(node_list)
    adj_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i):  # Only compute lower triangle (i > j)
            pos_i = np.array(node_list[i].position)
            pos_j = np.array(node_list[j].position)
            d = np.linalg.norm(pos_i - pos_j)

            if d < r:
                weight = np.exp(-d**2)
                adj_matrix[i, j] = weight
                adj_matrix[j, i] = weight  # Symmetric

    return adj_matrix


def get_X_matrix(node_list):
    num_nodes = len(node_list)
    X_t = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if(i == j):
                this_node = node_list[i]
                X_t[i, j] = this_node.concentration
    return X_t

def get_b_c_matrix(node_list, delta_w, r=10):
    n = len(node_list) - 1  # Exclude shared resource
    col_matrix = np.zeros((n, 1))

    for i in range(n):
        pos = np.array(node_list[i].position)
        d = np.linalg.norm(pos)

        if d < r:
            col_matrix[i, 0] = delta_w * np.exp(-d**2)

    return col_matrix

def get_x_vec(node_list):
    n = len(node_list[:-1])
    col_matrix = np.zeros((n, 1))
    for i in range(n):
        this_node = node_list[i]
        node_x_val = this_node.concentration
        col_matrix[i, 0] = node_x_val
    return col_matrix

def find_avg(node_list):
    n = len(node_list[:-1])
    sum = 0
    for i in range(n):
        this_node = node_list[i]
        sum = sum + this_node.concentration
    return sum / n


####################################


if __name__ == "__main__":
    print("starting program...")
    #animation object
    anim = Animation()
    #
    center = 2.5
    scaling = 5 # keeps the values to start inside the box
    #variables
    num_nodes = 10
    length = 5
    sim_time = 50
    bad_guys = [0, 2, 4, 6, 8]

    node_list = []
    #experiment variables
    beta = 1
    delta_i = 3
    delta_w = 0.25
    z_init = 0.5
    #initiate all the nodes
    for i in range(num_nodes):
        x_pos = random.random() * scaling
        y_pos = random.random() * scaling
        if(i in bad_guys):
            this_node = Node(x_pos, y_pos, "Infected", delta_i, z_init)
        else:
            this_node = Node(x_pos, y_pos, "Susceptible", delta_i, z_init)
        node_list.append(this_node)

    #make shared resource
    w_1 = Node(0, 0, "Shared Resource", delta_w, z_init)
    node_list.append(w_1)
    #Run simulation, at each time step t the nodes should be moving around
    t_start = 0
    delta_t = 0.05
    avg_val_list = []
    eig_val = []
    sr_vals = []
    t_val = []
    #for each time step:
    while t_start < sim_time:
        #for each time step find the following matrices
        D_t = np.eye(num_nodes) * delta_i
        # find the adjacency matrix
        A_t = get_adj_matrix(node_list[:-1])
        B_t = beta *  A_t
        X_t = get_X_matrix(node_list[:-1])
        b_t = get_b_c_matrix(node_list, delta_w)
        c_t = b_t
        x_vec = get_x_vec(node_list)
        z = np.array([[node_list[-1].concentration]])
        y_t = np.block([
            [x_vec],
            [z]
        ])
        zero_vec_0 = np.zeros((X_t.shape[0], 1))
        zero_vec_1 = np.zeros((1, X_t.shape[0] + 1))
        X_y_t = np.block([
            [X_t, zero_vec_0],
            [zero_vec_1]
        ])
        B_w = np.block([
            [B_t, b_t],
            [c_t.T, 0]
        ])
        D_w = np.block([
            [D_t, zero_vec_0],
            [zero_vec_0.T, np.array([[delta_w]])]
        ])
        #print(B_w)
        I = np.eye(X_y_t.shape[0])
        y_dot = (-1 * D_w + (I - X_y_t) @ B_w) @ y_t
        y_t_1 = delta_t * y_dot + y_t
        eigvals, _ = np.linalg.eig(B_w - D_w)
        eig_val.append(np.max(eigvals))
        t_val.append(t_start)

        avg_val = find_avg(node_list)
        avg_val_list.append(avg_val)
        sr_vals.append(w_1.concentration)

        #update the position of all the nodes, and then plot them in a animation
        for i in range(len(node_list)):
            this_node = node_list[i]
            dx, dy = delta_t * this_node.x_vel, delta_t * this_node.y_vel
            this_node.update_position(dx, dy)
            #if node comes in contact with bound, phi -> -phi
            this_node.check_boundary_cross(center, length)
            this_node.concentration = y_t_1[i, 0]


        #update the time 
        t_start = t_start + delta_t
        anim.update(node_list)
        plt.pause(0.01)  # Pause to see each update

plt.show()  # Display the final animation

plt.plot(t_val, eig_val)
plt.title("Max eig value")
plt.xlabel('t')
plt.ylabel('eig')
plt.show()

plt.plot(t_val, avg_val_list, color = "blue", label = "node")
plt.plot(t_val, sr_vals, color = "red", label = "shared resource")
plt.title("Average Infection over Time")
plt.xlabel('t')
plt.ylabel('infection')
plt.legend()
plt.show()

