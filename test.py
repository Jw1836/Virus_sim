import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define A(t)
def A(t):
    return np.array([[0, 1],
                     [-np.cos(t), -0.1]])

# Define the system: dx/dt = A(t) x
def system(t, x):
    return np.array([[0, 0],
                     [np.sin(t), 0]])

# Time span and evaluation points
t_span = (0, 20)
t_eval = np.linspace(*t_span, 1000)

# Initial conditions for phase portrait
initial_conditions = [
    [1, 0], [0, 1], [-1, 0], [0, -1], 
    [1, 1], [-1, -1], [0.5, -0.5], [-0.5, 0.5]
]

# Plot phase portrait
plt.figure(figsize=(8, 6))
for x0 in initial_conditions:
    sol = solve_ivp(system, t_span, x0, t_eval=t_eval)
    plt.plot(sol.y[0], sol.y[1], label=f"x0={x0}")

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Phase Portrait of $\\dot{x} = A(t)x$')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()
