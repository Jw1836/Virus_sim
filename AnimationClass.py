
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import random
matplotlib.use('tkagg')  # requires TkInter


class Node:
    """
    A Node class representing a point that has a position (x, y).
    """
    def __init__(self, x_pos, y_pos, Type, delta, z_init):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.z = None
        #phi variables  - velocity
        self.x_vel = random.gauss(0, 1) 
        self.y_vel = random.gauss(0, 1) 
        #delta_i is birth + recovery rate, delta_w is decay rate in resource
        self.delta = delta
        self.NodeType = Type
        if(self.NodeType == "Infected"):
            self.color = "red"
            self.concentration = 1

        elif(self.NodeType == "Susceptible"):
            self.color = "blue"
            self.concentration = 0

        elif(self.NodeType == "Shared Resource"):
            self.color = "green"
            self.x_vel = 0
            self.y_vel = 0
            self.concentration = z_init

        self.position = (self.x_pos, self.y_pos)  # Initialize node position (x, y)
        
        
    def update_position(self, dx, dy):
        """
        Update the position of the node by adding the displacements (dx, dy).
        """
        x, y = self.position
        self.position = (x + dx, y + dy)
        self.x_pos = x + dx
        self.y_pos = y + dy


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
        
        # Parameters to define the size of the plot
        self.length = 5
        self.width = 5
        
        # Set axis limits
        plt.axis([-2.0*self.length, 2.0*self.length, -2.0*self.length, 2.0*self.length])
        
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
                circle = mpatches.Circle((x, y), radius=0.25, color=node.color, lw=1)
                self.ax.add_patch(circle)  # Add the circle to the axes
                self.handle.append(circle)  # Store the handle for future updates
            
            self.FlagInit = False  # Set flag to False so this block doesn't run again
        
        else:
            # Update the position of each node in the list
            for i, node in enumerate(self.node_list):
                x, y = node.position  # Get the updated position
                self.handle[i].center = (x, y)  # Update the position of the circle (node)
        
        # Redraw the plot with the updated nodes
        plt.draw()

# # Example Usage:
# # Create an instance of the Animation class
# anim = Animation()

# # Create a list of nodes (initial positions)
# nodes = [Node(1, 1), Node(2, 2), Node(3, 3)]

# # Simulated node movements (for example, updating by moving each node by a certain displacement)
# node_positions = [
#     [(1, 1), (2, 2), (3, 3)],  # Initial positions
#     [(2, 2), (3, 3), (4, 4)],  # First update
#     [(3, 3), (4, 4), (5, 5)],  # Second update
#     [(4, 4), (5, 5), (6, 6)],  # Third update
# ]

# # Loop through the node positions and update the animation
# for positions in node_positions:
#     # Update the node positions
#     for i, pos in enumerate(positions):
#         print("pos: ", pos)
#         dx, dy = pos[0] - nodes[i].position[0], pos[1] - nodes[i].position[1]
#         nodes[i].update_position(dx, dy)
    
#     # Update the plot with the new node positions
#     anim.update(nodes)
#     plt.pause(0.5)  # Pause to see each update

# plt.show()  # Display the final animation
