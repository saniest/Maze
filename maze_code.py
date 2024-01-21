import random
import numpy as np
from collections import deque
import pickle

import tkinter as tk
import tkinter.filedialog as filedialog

# Constants representing the four sides of a cell
TOP = 0
LEFT = 1
BOTTOM = 2
RIGHT = 3

# set a fixed seed for reproducibility
# random.seed(1234)


class MazeGenerator:
    """
    MazeGenerator class uses "recursive division" algorithm to generate mazes.
    The mazes are represented as a three-dimensional numpy array cells,
    where first and second dimensions represent the x and y coordinates of the cell,
    and the 3rd dimension indicates whether there is a wall on each side of the cell.
    """

    def __init__(self, size: int):
        self.size = size
        self.cells = np.zeros([size, size, 4], dtype=np.uint8)

        # initializing outer walls
        self.cells[0, :, TOP] = 1
        self.cells[size - 1, :, BOTTOM] = 1
        self.cells[:, 0, LEFT] = 1
        self.cells[:, size - 1, RIGHT] = 1

    def divide_horizontally(self, x_start: int, x_end: int, y_start: int, y_end: int):
        """Divide the maze horizontally at a random point, leaving one gap open"""

        # Stop dividing if the region is too small
        if y_end - y_start < 1:
            return

        # Choose a random point to divide the region
        divide_at = random.randint(x_start, x_end - 1)
        # Choose a random point to leave a gap
        gap_pos = random.randrange(y_start, y_end)
        # Set walls
        for i in range(y_start, y_end + 1):
            if i == gap_pos:
                continue

            self.cells[divide_at, i, BOTTOM] = 1

        self.divide_vertically(x_start, divide_at, y_start, y_end)
        self.divide_vertically(divide_at + 1, x_end, y_start, y_end)

    def divide_vertically(self, x_start: int, x_end: int, y_start: int, y_end: int):
        """Divide the maze vertically at a random point, leaving one gap open"""

        # Stop dividing if the region is too small
        if x_end - x_start < 1:
            return

        # Choose a random point to divide the region
        divide_at = random.randint(y_start, y_end - 1)
        # Choose a random point to leave a gap
        gap_pos = random.randrange(x_start, x_end)
        # Set walls
        for i in range(x_start, x_end + 1):
            if i == gap_pos:
                continue

            self.cells[i, divide_at, RIGHT] = 1

        self.divide_horizontally(x_start, x_end, y_start, divide_at)
        self.divide_horizontally(x_start, x_end, divide_at + 1, y_end)

    def generate(self):
        """Generate a maze using recursive division algorithm"""
        self.divide_horizontally(0, self.size - 1, 0, self.size - 1)

        return Maze(self.cells)


class Maze:
    """
    Maze class represents a maze and provides methods to solve it.
    It also provides methods to save and load the maze to/from a file.
    """

    def __init__(self, cells: np.ndarray):
        self.cells = cells
        self.size = cells.shape[0]
        self.exit_point = self.random_exit()
        print("Exit point:", self.exit_point)

    def __getitem__(self, index):
        """dunder method to directly access cells by index"""
        return self.cells[index]

    def save_to_file(self, filename: str):
        """Save the maze to a file using pickle"""
        pickle.dump(self, open(filename, "wb"))

    @classmethod
    def load_from_file(cls, filename: str) -> "Maze":
        """Load the maze from a file"""
        return pickle.load(open(filename, "rb"))

    def get_neighbours(self, x, y) -> set[tuple[int, int]]:
        """Return a set of neighbours for a given cell not blocked by walls"""
        neighbours = set()
        if self.cells[x, y, BOTTOM] == 0 and x < self.size - 1:
            neighbours.add((x + 1, y))
        if self.cells[x, y, RIGHT] == 0 and y < self.size - 1:
            neighbours.add((x, y + 1))
        if self.cells[x - 1, y, BOTTOM] == 0 and x > 0:
            neighbours.add((x - 1, y))
        if self.cells[x, y - 1, RIGHT] == 0 and y > 0:
            neighbours.add((x, y - 1))

        return neighbours

    def random_exit(self) -> tuple[int, int]:
        """
        This function randomly selects an exit point for the maze.
        """

        side = random.choice([TOP, LEFT, BOTTOM, RIGHT])

        if side == TOP:
            x_exit = 0
            y_exit = random.choice(range(self.size))
        elif side == RIGHT:
            y_exit = self.size - 1
            x_exit = random.choice(range(self.size))
        elif side == BOTTOM:
            x_exit = self.size - 1
            y_exit = random.choice(range(1, self.size))
        else:  # side == LEFT
            y_exit = 0
            x_exit = random.choice(range(1, self.size))

        # leave a gap in the wall
        self.cells[x_exit, y_exit, side] = 0

        return x_exit, y_exit

    def solve_bfs(self, x_start: int = 0, y_start: int = 0) -> list[tuple[int, int]]:
        """Solve the maze using breadth first search"""

        x_exit, y_exit = self.exit_point

        # queue to store the path
        queue = deque([(x_start, y_start)])
        # visited cells
        visited = set()
        # solution paths
        solution = []

        while queue:
            x, y = queue.popleft()
            solution.append((x, y))

            # if the current cell is the exit cell, then we are done!
            if x == x_exit and y == y_exit:
                print("Reached to the exit!")
                return solution

            # check if the current cell has any unvisited neighbours
            # since both neighbors and visited are sets, we can use subtraction to find the difference
            unvisited_neighbours = list(self.get_neighbours(x, y) - visited)

            # if it has, then add all of them to the queue
            if unvisited_neighbours:
                queue.extend(unvisited_neighbours)
                visited.update(unvisited_neighbours)

        return solution

    def solve_dfs(self, x_start: int = 1, y_start: int = 1) -> list[tuple[int, int]]:
        """Solve the maze using depth first search"""

        x_exit, y_exit = self.exit_point

        # stack to store the path
        stack = [(x_start, y_start)]
        # visited cells
        visited = set()
        # solution paths
        solution = []

        while stack:

            x, y = stack.pop()
            solution.append((x, y))

            # if the current cell is the exit cell, then we are done!
            if x == x_exit and y == y_exit:
                print("Reached to the exit!")
                return solution

            # check if the current cell has any unvisited neighbours
            # since both neighbors and visited are sets, we can use subtraction to find the difference
            unvisited_neighbours = list(self.get_neighbours(x, y) - visited)

            # if it has, then add all of them to the stack
            if unvisited_neighbours:
                stack.extend(unvisited_neighbours)
                visited.update(unvisited_neighbours)

        return solution


class MazeGUI:
    def __init__(self):

        # Create the Tkinter form
        self.root = tk.Tk()

        # Add an entry field to get the size of the maze
        tk.Label(self.root, text="Maze size").grid(row=0, column=0)
        self.size_entry = tk.Entry(self.root)
        self.size_entry.insert(0, "20")  # default input is 5
        self.size_entry.grid(row=0, column=1)

        # Create a button to generate the maze
        self.generate_button = tk.Button(
            self.root, text="Generate", command=self.generate
        )
        self.generate_button.grid(row=0, column=2)

        # Create a button to save the maze
        self.save_button = tk.Button(self.root, text="Save", command=self.save)
        self.save_button.grid(row=0, column=3)

        # Create a button to open the maze
        self.open_button = tk.Button(self.root, text="Open", command=self.open)
        self.open_button.grid(row=0, column=4)

        # Add an entry field to get the starting point
        tk.Label(self.root, text="Start point").grid(row=1, column=0)
        self.xstart_entry = tk.Entry(self.root)
        self.xstart_entry.insert(0, "0")
        self.xstart_entry.grid(row=1, column=1)

        self.ystart_entry = tk.Entry(self.root)
        self.ystart_entry.insert(0, "0")
        self.ystart_entry.grid(row=1, column=2)

        # Create a button to solve the maze
        self.solve_dfs_button = tk.Button(
            self.root, text="Solve DFS", command=self.solve_dfs
        )
        self.solve_dfs_button.grid(row=1, column=3)

        self.solve_bfs_button = tk.Button(
            self.root, text="Solve BFS", command=self.solve_bfs
        )
        self.solve_bfs_button.grid(row=1, column=4)

        # Create a canvas to display the maze
        self.canvas = tk.Canvas(self.root, width=500, height=500)
        self.canvas.grid(row=2, column=0, columnspan=5)

    def generate(self):
        """Generate a new maze and draw it on the canvas"""
        # Get the size of the maze from the entry field
        self.size = int(self.size_entry.get())

        self.xstart = int(self.xstart_entry.get())
        self.ystart = int(self.ystart_entry.get())

        # Generate the maze
        self.maze = MazeGenerator(self.size).generate()

        # Draw the maze on the canvas
        self.draw(self.canvas, self.maze)

    def draw(self, canvas, maze, linewidth=2):
        """Draw the maze on a Tkinter canvas"""
        # Constants for the size of the cells in the maze
        CELL_SIZE = 500 // self.size

        # Draw the white background
        canvas.create_rectangle(0, 0, 500, 500, fill="white")

        # Draw the walls in the maze
        for y in range(0, self.size):
            for x in range(0, self.size):
                # Draw the top wall
                if maze[y][x][TOP]:
                    canvas.create_line(
                        x * CELL_SIZE,
                        y * CELL_SIZE,
                        (x + 1) * CELL_SIZE,
                        y * CELL_SIZE,
                        fill="black",
                        width=linewidth,
                    )

                # Draw the bottom wall
                if maze[y][x][BOTTOM]:
                    canvas.create_line(
                        x * CELL_SIZE,
                        (y + 1) * CELL_SIZE,
                        (x + 1) * CELL_SIZE,
                        (y + 1) * CELL_SIZE,
                        fill="black",
                        width=linewidth,
                    )

                # Draw the right wall
                if maze[y][x][RIGHT]:
                    canvas.create_line(
                        (x + 1) * CELL_SIZE,
                        y * CELL_SIZE,
                        (x + 1) * CELL_SIZE,
                        (y + 1) * CELL_SIZE,
                        fill="black",
                        width=linewidth,
                    )

                # Draw the left wall
                if maze[y][x][LEFT]:
                    canvas.create_line(
                        x * CELL_SIZE,
                        y * CELL_SIZE,
                        x * CELL_SIZE,
                        (y + 1) * CELL_SIZE,
                        fill="black",
                        width=linewidth,
                    )

        # Draw the red circle at the start position
        canvas.create_oval(
            (self.ystart + 0.10) * CELL_SIZE + 1,
            (self.xstart + 0.10) * CELL_SIZE + 1,
            (self.ystart + 0.90) * CELL_SIZE,
            (self.xstart + 0.90) * CELL_SIZE,
            fill="red",
        )

        xexit, yexit = self.maze.exit_point
        # Draw the green circle at the exit point
        canvas.create_oval(
            (yexit + 0.10) * CELL_SIZE + 1,
            (xexit + 0.10) * CELL_SIZE + 1,
            (yexit + 0.90) * CELL_SIZE,
            (xexit + 0.90) * CELL_SIZE,
            fill="green",
        )

    def save(self):
        """Save the maze to a file"""
        # Ask the user for the file to save to
        file_name = filedialog.asksaveasfilename(defaultextension=".maze")
        # Save the maze to the file
        self.maze.save_to_file(file_name)

    def open(self):
        """Open a maze from a file and draw it on the canvas"""
        # Ask the user for the file to open
        file_name = filedialog.askopenfilename()
        # Open the maze from the file
        self.maze = Maze.load_from_file(file_name)
        self.size = self.maze.size
        # Draw the maze on the canvas
        self.draw(self.canvas, self.maze)

    def solve_dfs(self):
        """Solve the maze using depth-first search"""

        # Get the starting point from the entry fields
        xstart = int(self.xstart_entry.get())
        ystart = int(self.ystart_entry.get())

        solution = self.maze.solve_dfs(x_start=xstart, y_start=ystart)
        self.draw_solution(self.canvas, solution)

    def solve_bfs(self):
        """Solve the maze using breadth-first search"""
        # Get the starting point from the entry fields
        xstart = int(self.xstart_entry.get())
        ystart = int(self.ystart_entry.get())

        solution = self.maze.solve_bfs(x_start=xstart, y_start=ystart)
        self.draw_solution(self.canvas, solution)

    def draw_solution(self, canvas, path):
        """Draw the solution on the canvas"""
        # Constants for the size of the cells in the maze
        CELL_SIZE = 500 // self.size

        # Draw the initial state of the maze
        self.draw(canvas, self.maze)

        # Function to draw the next step of the solution in a animation
        def next_step():
            # If there are more steps in the solution
            if path:
                # Get the next step
                x, y = path.pop(0)
                # Draw the step on the canvas
                canvas.create_rectangle(
                    (y + 0.30) * CELL_SIZE + 1,
                    (x + 0.30) * CELL_SIZE + 1,
                    (y + 0.70) * CELL_SIZE,
                    (x + 0.70) * CELL_SIZE,
                    fill="yellow",
                )
                # Call this function again after a short delay
                self.root.after(20, next_step)

        # Start animating the solution
        next_step()


if __name__ == "__main__":
    # Create an instance of the MazeGUI
    gui = MazeGUI()
    # Run the Tkinter event loop
    gui.root.mainloop()
