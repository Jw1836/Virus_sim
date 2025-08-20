from my_classes import PopulationNode
from my_classes import ResourceNode
from my_classes import Animation
import random
import numpy as np 
import matplotlib.pyplot as plt
import sys

#helper functions start here
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

def find_avg(node_list, n, k):
    sum = 0
    for i in range(n):
        this_node = node_list[i]
        if(k == 1):
            concentration = this_node.v_1
        else:
            concentration = this_node.v_2

        sum = sum + concentration
    return sum / n

def diff_equation(k, nodes, alpha):
        A = get_adj_matrix(nodes, n, r=10)
        # k = 1
        x_k_vec = get_x_vec(nodes, k, n)
        x_1_vec = get_x_vec(nodes, 1, n)
        w_k_vec = get_w_vec(nodes, k, m)

        x_2_vec = get_x_vec(nodes, 2, n)

        B_k = get_B_k(A, n, nodes, k)
        B_k_w = get_B_k_w(n, nodes, k)
        C_k_w = B_k_w.T #for homogenous case
        delta_k = nodes[0].delta_1 if k == 1 else nodes[0].delta_2
        delta_k_w = nodes[-1].delta_1_w if k == 1 else nodes[-1].delta_2_w
        D_k = np.eye(n) * delta_k
        D_k_w = np.eye(m) * delta_k_w
        A_k_w =  [[alpha for _ in range(m)] for _ in range(m)] - alpha * m * np.eye(m)
        A_k_w_diag = np.diag(np.diag(A_k_w))
        #start building the full block matrices for k= 1
        y_k_1 = np.block([
            [x_k_vec],
            [w_k_vec]
        ])
        X_y_1 = np.block([
            [np.diag(x_1_vec.flatten()), np.zeros((n, m))],
            [np.zeros((m, m + n))]
        ])
        X_y_2 = np.block([
            [np.diag(x_2_vec.flatten()), np.zeros((n, m))],
            [np.zeros((m, m + n))]
        ])
        B_k_f = np.block([
            [B_k, B_k_w],
            [C_k_w, A_k_w - A_k_w_diag]
        ])
        D_k_f= np.block([
            [D_k, np.zeros((n, m))],
            [np.zeros((m, n)), D_k_w - A_k_w_diag]
        ])

        y_k_2 = ((-1 * D_k_f + (np.eye(n + m) - (X_y_1 + X_y_2)) @ B_k_f) @ y_k_1) * delta_t + y_k_1
        return y_k_2, B_k_f, D_k_f

if __name__ == "__main__":
    n = 7 # number of population nodes
    m = 2 # number of resource nodes 
    alpha = 0.1 # rate of infection from rnode to rnode
    beta_1 = 1 # infection rate virus 1
    delta_1 = 0.5 # recovery rate 1
    beta_2 = 1 # infection rate virus 2
    delta_2 = 0.5 # recovery rate 2 
    # v_1_init = 0.5 
    # v_2_init = 0.5
    r_to_n_infection_rate_1 = alpha
    r_to_n_infection_rate_2 = alpha
    scaling = 5
    bad_guys_1 = [0, 2, 6]
    bad_guys_2 = [1, 3, 5]
    nodes = []
    for i in range(n + m):
        v_1_init = 0
        v_2_init = 0
        if(i in bad_guys_1):
            v_1_init = 0.1
        elif(i in bad_guys_2):
            v_2_init = 0.1
        else:
            pass 

        x_pos = random.random() * scaling
        y_pos = random.random() * scaling    

        if(i < n):
            nodes.append(PopulationNode(x_pos, y_pos, beta_1, delta_1, beta_2, delta_2, v_1_init, v_2_init))
        else:
            nodes.append(ResourceNode(x_pos, y_pos, r_to_n_infection_rate_1, r_to_n_infection_rate_2, v_1_init, v_2_init, delta_1, delta_2))


    # Create and run animation
    anim = Animation()
    t_start = 0
    sim_time = 30
    delta_t = 0.05
    length = 5
    center = 2.5
    avg_val_list_v1 = []
    eig_val_v1 = []
    sr_vals_v1 = []
    avg_val_list_v2 = []
    eig_val_v2 = []
    sr_vals_v2 = []
    t_val = []
        #for each time step:
    while t_start < sim_time:
        #beta's are time varying: (infection rate)
        beta_1_new = 1 + np.sin(t_start / 100)
        beta_2_new =.8 + np.sin(t_start / 100)
        for l in range(n):
            this_node = nodes[l]
            this_node.beta_1 = beta_1_new
            this_node.beta_2 = beta_2_new
        #for each timestep solve the differential equation
        y_2_v1, B_1_f, D_1_f = diff_equation(1, nodes, alpha)
        y_2_v2, B_2_f, D_2_f = diff_equation(2, nodes, alpha)
        
        eigvals_1, _ = np.linalg.eig(B_1_f - D_1_f)
        eig_val_v1.append(np.max(eigvals_1))
        t_val.append(t_start)

        avg_val = find_avg(nodes, n, 1)
        avg_val_list_v1.append(avg_val)
        sr_vals_v1.append(find_avg(nodes[:n], m, 1))

        eigvals_2, _ = np.linalg.eig(B_2_f - D_2_f)
        eig_val_v2.append(np.max(eigvals_2))

        avg_val = find_avg(nodes, n, 2)
        avg_val_list_v2.append(avg_val)
        sr_vals_v2.append(find_avg(nodes[:n], m, 2))

        #update the position of all the nodes, and then plot them in a animation
        for i in range(len(nodes)):
            this_node = nodes[i]
            dx, dy = delta_t * this_node.x_vel, delta_t * this_node.y_vel
            this_node.update_position(dx, dy)
            #if node comes in contact with bound, phi -> -phi
            this_node.check_boundary_cross(center, length)
            #update node value
            this_node.v_1 = y_2_v1[i, 0]
            this_node.v_2 = y_2_v2[i, 0]

        #update the time 
        t_start = t_start + delta_t
        anim.update(nodes)
        plt.pause(0.01)  # Pause to see each update



plt.show()  # Display the final animation


# Create a figure with two subplots (vertically stacked)
fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # 2 rows, 1 column

# First plot: Max eig value
axs[0].plot(t_val, eig_val_v1, color="blue", label="v1")
axs[0].plot(t_val, eig_val_v2, color="red", label="v2")
axs[0].set_title("Max eig value")
axs[0].set_xlabel('t')
axs[0].set_ylabel('eig')
axs[0].legend()

# Second plot: Average Infection over Time
axs[1].plot(t_val, avg_val_list_v2, color="blue", label="node v2")
axs[1].plot(t_val, sr_vals_v2, color="red", label="shared resource v2")
axs[1].plot(t_val, avg_val_list_v1, color="green", label="node v1")
axs[1].plot(t_val, sr_vals_v1, color="black", label="shared resource v1")
axs[1].set_title("Average Infection over Time")
axs[1].set_xlabel('t')
axs[1].set_ylabel('infection')
axs[1].legend()

# Adjust layout to avoid overlap
plt.tight_layout()
plt.show()