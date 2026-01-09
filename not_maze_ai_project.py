import numpy as np
import matplotlib.pyplot as plt
import time
import turtle

# ---------------- CONFIG ----------------
tile_size = 30

maze_overview = [
    "XXXXXXXXXXXXXXX",
    "X   X       XGX",
    "X X X XXXXX X X",
    "X X X     X X X",
    "X X XXXXX X X X",
    "X X     X X   X",
    "X XXXXX X XXX X",
    "XS      X     X",
    "XXXXXXXXXXXXXXX"
]

actions = [(-1,0),(1,0),(0,-1),(0,1)]
goal_reward = 100
wall_penalty = -10
step_penalty = -1

# ---------------- MAZE ----------------
class Maze:
    def __init__(self, overview):
        self.maze_height = len(overview)
        self.maze_width = len(overview[0])
        self.maze = np.zeros((self.maze_height, self.maze_width), dtype=int)
        self.start_position = None
        self.goal_position = None

        for r,row in enumerate(overview):
            for c,ch in enumerate(row):
                if ch == "X":
                    self.maze[r,c] = 1
                elif ch == "S":
                    self.start_position = (r,c)
                elif ch == "G":
                    self.goal_position = (r,c)

# ---------------- Q LEARNING ----------------
class QLearningAgent:
    def __init__(self, maze):
        self.q_table = np.zeros((maze.maze_height, maze.maze_width, 4))
        self.lr = 0.1
        self.gamma = 0.9

    def get_action(self, state, eps):
        if np.random.rand() < eps:
            return np.random.randint(4)
        return np.argmax(self.q_table[state])

    def update(self, s,a,ns,r):
        self.q_table[s][a] += self.lr * (
            r + self.gamma * np.max(self.q_table[ns]) - self.q_table[s][a]
        )

# ---------------- TRAIN / TEST ----------------
def run_episode(agent, maze, train=True, eps=0.1):
    state = maze.start_position
    path = [state]

    for _ in range(500):
        action = agent.get_action(state, eps)
        r,c = state
        dr,dc = actions[action]
        ns = (r+dr, c+dc)

        if ns[0] < 0 or ns[0] >= maze.maze_height or ns[1] < 0 or ns[1] >= maze.maze_width or maze.maze[ns] == 1:
            reward = wall_penalty
            ns = state
        elif ns == maze.goal_position:
            reward = goal_reward
            path.append(ns)
            if train:
                agent.update(state, action, ns, reward)
            break
        else:
            reward = step_penalty
            path.append(ns)

        if train:
            agent.update(state, action, ns, reward)
        state = ns

    return path

def train(agent, maze, episodes=200):
    for ep in range(episodes):
        eps = max(0.01, 1 - ep/episodes)
        run_episode(agent, maze, train=True, eps=eps)

# ---------------- TURTLE VISUAL ----------------
screen = turtle.Screen()
screen.setup(800,600)
screen.bgcolor("#973D3D")
screen.tracer(0)

drawer = turtle.Turtle()
drawer.hideturtle()
drawer.penup()
drawer.shape("square")
drawer.shapesize(tile_size/20)

rows = len(maze_overview)
cols = len(maze_overview[0])
origin_x = -cols * tile_size // 2
origin_y = rows * tile_size // 2

for r,row in enumerate(maze_overview):
    for c,ch in enumerate(row):
        x = origin_x + c*tile_size
        y = origin_y - r*tile_size
        if ch == "X":
            drawer.goto(x,y)
            drawer.stamp()

player = turtle.Turtle()
player.shape("turtle")
player.color("orange")
player.penup()

goal = turtle.Turtle()
goal.shape("square")
goal.color("yellow")
goal.penup()

def cell_to_screen(r,c):
    return origin_x + c*tile_size, origin_y - r*tile_size

# ---------------- RUN ----------------
maze = Maze(maze_overview)
agent = QLearningAgent(maze)

train(agent, maze)

path = run_episode(agent, maze, train=False, eps=0)

player.goto(*cell_to_screen(*maze.start_position))
goal.goto(*cell_to_screen(*maze.goal_position))
screen.update()

for r,c in path:
    player.goto(*cell_to_screen(r,c))
    screen.update()
    time.sleep(0.1)

turtle.done()
 