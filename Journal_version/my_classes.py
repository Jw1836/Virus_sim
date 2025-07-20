
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import random
matplotlib.use('tkagg')  # for GUI-based animation


# ---------------------------
# ResourceNode and PopulationNode Definitions
# ---------------------------

class ResourceNode:
    def __init__(self, x, y, r_to_n_infection_rate_1, r_to_n_infection_rate_2, v_1_init, v_2_init, delta_1_w, delta_2_w):
        self.x_pos = x
        self.y_pos = y
        self.r_to_n_infection_rate_1 = r_to_n_infection_rate_1
        self.r_to_n_infection_rate_2 = r_to_n_infection_rate_2
        self.delta_1_w = delta_1_w
        self.delta_2_w = delta_2_w
        self.v_1 = v_1_init
        self.v_2 = v_2_init
        self.color = 'green'
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
    def __init__(self, x, y, infection_rate_1, recovery_rate_1, infection_rate_2, recovery_rate_2, v_1_init, v_2_init):
        self.x_pos = x
        self.y_pos = y
        self.beta_1 = infection_rate_1
        self.delta_1 = recovery_rate_1
        self.beta_2 = infection_rate_2
        self.delta_2 = recovery_rate_2
        self.v_1 = v_1_init
        self.v_2 = v_2_init
        self.color = "red" if v_1_init > 0 or v_2_init > 0 else "blue"
        self.position = (x, y)
        self.x_vel = random.gauss(0, 1) * 0.5
        self.y_vel = random.gauss(0, 1) * 0.5

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
        self.fig, self.ax = plt.subplots()
        
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

    def update(self, node_list):
        """
        This function updates the animation with a new set of nodes.
        It takes a list of Node objects.
        """
        # Store the new list of nodes
        self.node_list = node_list
        
        # On the first update (when FlagInit is True), create the nodes and path
        if self.FlagInit:
            # For each node in the list, create a patch (circle for the node)
            for node in self.node_list:
                x, y = node.position  # Get the current position of the node
                circle = mpatches.Circle((x, y), radius=0.1, color=node.color, lw=1)
                self.ax.add_patch(circle)  # Add the circle to the axes
                self.handle.append(circle)  # Store the handle for future updates
                            # Add text label for v_1 and v_2
                txt = self.ax.text(x + 0.3, y + 0.3, f"v₁={node.v_1:.2f}\nv₂={node.v_2:.2f}",
                                   fontsize=7, color='black', ha='left')
                self.texts.append(txt)
            self.FlagInit = False  # Set flag to False so this block doesn't run again
        
        else:
            # Update the position of each node in the list
            for i, node in enumerate(self.node_list):
                x, y = node.position  # Get the updated position
                self.handle[i].center = (x, y)  # Update the position of the circle (node)

                # Update text position and values
                self.texts[i].set_position((x + 0.3, y + 0.3))
                self.texts[i].set_text(f"v₁={node.v_1:.2f}\nv₂={node.v_2:.2f}")
        
        # Redraw the plot with the updated nodes
        plt.draw()
