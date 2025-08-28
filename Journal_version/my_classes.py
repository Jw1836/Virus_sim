
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import random
matplotlib.use('tkagg')  # for GUI-based animation


# ---------------------------
# ResourceNode and PopulationNode Definitions
# ---------------------------

class ResourceNode:
    def __init__(self, x, y, v_1_init, v_2_init, delta_1_w, delta_2_w):
        self.x_pos = x
        self.y_pos = y
        self.delta_1_w = delta_1_w
        self.delta_2_w = delta_2_w
        self.v_1 = v_1_init
        self.v_2 = v_2_init
        #self.color = 'green'
        self.position = (x, y)
        self.x_vel = 0
        self.y_vel = 0

    def update_position(self, dx, dy):
        self.x_pos += dx
        self.y_pos += dy
        self.position = (self.x_pos, self.y_pos)

    
    def check_boundary_cross(self, center, length):
        # Boundary limits based on the center and the length of the square
        left = center - length/2
        right = center + length/2
        bottom = center - length/2
        top = center + length/2

        # Check for crossing the x-boundary
        if self.x_vel > 0 and self.x_pos >= right:  # Moving right and crossed the right boundary
            self.x_vel = -self.x_vel  # Reverse x-velocity (bounce off right boundary)
        elif self.x_vel < 0 and self.x_pos <= left:  # Moving left and crossed the left boundary
            self.x_vel = -self.x_vel  # Reverse x-velocity (bounce off left boundary)

        # Check for crossing the y-boundary
        if self.y_vel > 0 and self.y_pos >= top:  # Moving up and crossed the top boundary
            self.y_vel = -self.y_vel  # Reverse y-velocity (bounce off top boundary)
        elif self.y_vel < 0 and self.y_pos <= bottom:  # Moving down and crossed the bottom boundary
            self.y_vel = -self.y_vel  # Reverse y-velocity (bounce off bottom boundary)


class PopulationNode:
    def __init__(self, x, y, recovery_rate_1, recovery_rate_2, v_1_init, v_2_init):
        self.x_pos = x
        self.y_pos = y
        self.beta_1 = None #done in the while loop
        self.delta_1 = recovery_rate_1
        self.beta_2 = None #done in the while loop
        self.delta_2 = recovery_rate_2
        self.v_1 = v_1_init
        self.v_2 = v_2_init
        #self.color = "blue"
        self.position = (x, y)
        self.x_vel = random.gauss(0, 1) * 0.5
        self.y_vel = random.gauss(0, 1) * 0.5
        self.x_vel = 0
        self.y_vel = 0

    def update_position(self, dx, dy):
        self.x_pos += dx
        self.y_pos += dy
        self.position = (self.x_pos, self.y_pos)

    def check_boundary_cross(self, center, length):
        # Boundary limits based on the center and the length of the square
        left = center - length/2
        right = center + length/2
        bottom = center - length/2
        top = center + length/2

        # Check for crossing the x-boundary
        if self.x_vel > 0 and self.x_pos >= right:  # Moving right and crossed the right boundary
            self.x_vel = -self.x_vel  # Reverse x-velocity (bounce off right boundary)
        elif self.x_vel < 0 and self.x_pos <= left:  # Moving left and crossed the left boundary
            self.x_vel = -self.x_vel  # Reverse x-velocity (bounce off left boundary)

        # Check for crossing the y-boundary
        if self.y_vel > 0 and self.y_pos >= top:  # Moving up and crossed the top boundary
            self.y_vel = -self.y_vel  # Reverse y-velocity (bounce off top boundary)
        elif self.y_vel < 0 and self.y_pos <= bottom:  # Moving down and crossed the bottom boundary
            self.y_vel = -self.y_vel  # Reverse y-velocity (bounce off bottom boundary)


# ---------------------------
# Animation Class Definition
# ---------------------------

class Animation:
    def __init__(self):
        self.FlagInit = True
        # Initialize the list of nodes
        self.node_list = []
        self.center = (2.5, 2.5)
        # Create figure and axis
        #self.fig, self.ax = plt.subplots()
        
        # Handle list to store the patches and line objects
        self.handle = []
        self.texts = []
        # Parameters to define the size of the plot
        self.length = 5
        self.width = 5
        
        # Set axis limits
        #plt.axis([-2.0*self.length, 2.0*self.length, -2.0*self.length, 2.0*self.length])
        plt.xlim(-1, 6)
        # Set y-axis limits
        plt.ylim(-1, 6)
        # Add a black square centered at (2.5, 2.5) with side length of 10, just the outline
        square = mpatches.Rectangle(
            (self.center[0] - 2.5, self.center[1] - 2.5),  # bottom-left corner (center - radius)
            5,  # width
            5,  # height
            linewidth=2,  # Thickness of the square's outline
            edgecolor='black',  # Black border
            facecolor='none'  # No fill color, just outline
        )
        
        # Add the square to the plot
        self.ax.add_patch(square)
                # Define legend items for color meanings
        legend_elements = [
            mpatches.Patch(color='blue', label='v₁ < 0.5'),
            mpatches.Patch(color='green', label='v₂ < 0.5'),
            mpatches.Patch(color='red', label='v₁ ≥ 0.5'),
            mpatches.Patch(color='orange', label='v₂ ≥ 0.5')
        ]

        # Add legend to the axis
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=8, frameon=True)


    def update(self, node_list):
        """
        This function updates the animation with a new set of nodes.
        It takes a list of Node objects.
        """
        # Store the new list of nodes
        self.node_list = node_list
        
        # On the first update (when FlagInit is True), create the nodes and path
        if self.FlagInit:
            for node in self.node_list:
                x, y = node.position  # Get the current position of the node
                # Color logic (unchanged)
                if(node.v_1 < 0.5):
                    node.color = "blue"
                elif(node.v_2 < 0.5):
                    node.color = "green"
                elif(node.v_2 >= 0.5):
                    node.color = "orange"
                elif(node.v_1 >= 0.5):
                    node.color = "red"
                else:
                    node.color = "black"
                
                # Draw shape based on node type
                if isinstance(node, ResourceNode):
                    shape = mpatches.Rectangle((x-0.1, y-0.1), 0.2, 0.2, color=node.color, lw=1)
                else:  # PopulationNode
                    shape = mpatches.Circle((x, y), radius=0.1, color=node.color, lw=1)
                self.ax.add_patch(shape)
                self.handle.append(shape)
                txt = self.ax.text(x + 0.3, y + 0.3, f"v₁={node.v_1:.2f}\nv₂={node.v_2:.2f}",
                                fontsize=7, color='black', ha='left')
                self.texts.append(txt)
            self.FlagInit = False
        else:
            for i, node in enumerate(self.node_list):
                x, y = node.position
                # Color logic (unchanged)
                if((node.v_1 < 0.5) and (node.v_2 < 0.5)):
                    node.color = "blue"
                elif(node.v_2 >= 0.5):
                    node.color = "pink"
                elif(node.v_1 >= 0.5):
                    node.color = "red"
                else:
                    node.color = "black"
                # Update shape position and color
                if isinstance(node, ResourceNode):
                    self.handle[i].set_xy((x-0.1, y-0.1))
                else:
                    self.handle[i].center = (x, y)
                self.handle[i].set_color(node.color)
                self.texts[i].set_position((x + 0.3, y + 0.3))
                self.texts[i].set_text(f"v₁={node.v_1:.2f}\nv₂={node.v_2:.2f}")
        # Redraw the plot with the updated nodes
        #plt.draw()




class PhasePlotAnimation:
    def __init__(self, initial_conditions):
        self.initial_conditions = initial_conditions
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.quiver_handles = []
        self.FlagInit = True

    def update(self, A_matrix_1, A_matrix_2, t_start):
        """
        y_init_list: list of initial condition vectors (each is shape (2,))
        A_matrix_1, A_matrix_2: matrices to multiply (each is shape (2,2))
        t: current time (optional, for title)
        """
        # Remove previous quivers
        y_init_list = self.initial_conditions
        for handle in self.quiver_handles:
            handle.remove()
        self.quiver_handles = []

        for y_init in y_init_list:
            y_dot_1 = A_matrix_1 @ y_init
            y_dot_2 = A_matrix_2 @ y_init
            # Plot y_dot_1 (blue arrow)
            q1 = self.ax.quiver(y_init[0], y_init[1], y_dot_1[0], y_dot_1[1], color='blue', angles='xy', scale_units='xy', scale=1, label='y_dot_1')
            # Plot y_dot_2 (red arrow)
            q2 = self.ax.quiver(y_init[0], y_init[1], y_dot_2[0], y_dot_2[1], color='red', angles='xy', scale_units='xy', scale=1, label='y_dot_2')
            self.quiver_handles.extend([q1, q2])


            self.ax.set_title(f"Phase Plot at t = {t_start:.2f}")

        #plt.draw()