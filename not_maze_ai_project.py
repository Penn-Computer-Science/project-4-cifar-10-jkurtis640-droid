import numpy as np
import matplotlib.pyplot as plt
import time
import turtle
import random

tile_size = 30
##this is a change

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
# sets up the screen for the maze
screen = turtle.Screen()
screen.title("AI Maze")
screen.bgcolor("#973D3D")
screen.setup(width=1000, height=800)
screen.tracer(0)
# Draws the wall
class WallDrawer(turtle.Turtle):
     ##Initialization
     def __init__(self):
        super().__init__()
        self.hideturtle()
        self.penup()
        self.color("black")
        self.shape("square")
        self.shapesize(tile_size / 20, tile_size / 20)
     def draw_wall(self,x,y):
         self.goto(x,y)
         self.stamp()


#from CGPT

class Maze:
    def __init__(self, overview):
        self.overview = overview
        self.maze_height = len(overview)
        self.maze_width = len(overview[0])

        self.maze = np.zeros((self.maze_height, self.maze_width), dtype=int)
        self.start_position = None
        self.goal_position = None

        for r, row in enumerate(overview):
            for c, ch in enumerate(row):
                if ch == "X":
                    self.maze[r, c] = 1
                elif ch == "S":
                    self.start_position = (r, c)
                elif ch == "G":
                    self.goal_position = (r, c)

        if self.start_position is None:
            raise ValueError("No 'S' found in maze_overview")
        if self.goal_position is None:
            raise ValueError("No 'G' found in maze_overview")

    def show_maze(self):
        plt.figure(figsize=(5,5))
        plt.imshow(self.maze, cmap="gray")
        plt.scatter(self.start_position[1], self.start_position[0], marker="s")
        plt.scatter(self.goal_position[1], self.goal_position[0], marker="s")
        plt.gca().invert_yaxis()
        plt.xticks([]); plt.yticks([])
        plt.show()

class Player(turtle.Turtle):
    def __init__(self, start_x, start_y):
        super().__init__()
        self.shape("turtle")
        self.color("orange")
        self.penup()
        self.goto(start_x,start_y)
    
    def move_up(self):
        new_x = self.xcor()
        new_y = self.ycor() + tile_size
        self.goto(new_x, new_y)

    def move_down(self):
        new_x = self.xcor()
        new_y = self.ycor() - tile_size
        self.goto(new_x,new_y)
   
    def move_left(self):
        new_x = self.xcor()
        new_y = self.ycor() - tile_size
        self.goto(new_x, new_y)

    def move_right(self):
        new_x = self.xcor()
        new_y = self.ycor() + tile_size
        self.goto(new_x,new_y)

wall_drawer = WallDrawer()
walls = []
start_pos = None
goal_pos = None

rows = len(maze_overview)
cols = len(maze_overview[0])
origin_x = -cols * tile_size // 2
origin_y = rows * tile_size // 2

for row_index, row in enumerate(maze_overview):
    for col_index, cell in enumerate(row):
        x = origin_x + col_index * tile_size
        y = origin_y - row_index * tile_size

        if cell == "X":
            wall_drawer.draw_wall(x,y)
            walls.append((x,y))
        elif cell == "S":
            start_pos = (x,y)
        elif cell == "G":
            goal_pos = (x,y)
     
if start_pos is None:
    raise ValueError("No start position 'S' can be defined in maze_overview")
player = Player(start_pos[0], start_pos[1])

goal_turtle = turtle.Turtle()
goal_turtle.shape("square")
goal_turtle.color("yellow")
goal_turtle.penup()
goal_turtle.goto(goal_pos[0], goal_pos[1])

def go_up():
    player.move_up()
    screen.update()

def go_down():
    player.move_down()
    screen.update()

def go_left():
    player.move_left()
    screen.update()

def go_right():
    player.move_right()
    screen.update()

screen.listen()
screen.onkey(go_up,"Up")
screen.onkey(go_down,"Down")
screen.onkey(go_left,"Left")
screen.onkey(go_right,"Right")

#turtle.done()

actions = [(-1,0),
           (1,0),
           (0,-1),
           (0,1)]

class QLearningAgent:
    def __init__(self, maze, learning_rate=0.1, discount_factor=0.9, exploration_start=1.0, exploration_end=0.01, num_episodes=100):

        self.q_table = np.zeros((maze.maze_height, maze.maze_width, 4))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_start = exploration_start
        self.exploration_end = exploration_end
        self.num_episodes = num_episodes

    def get_exploration_rate(self, current_episode):
        exploration_rate = self.exploration_start * (self.exploration_end / self.exploration_start) ** (current_episode / self.num_episodes)
        return exploration_rate
    
    def get_action(self, state, current_episode):
        exploration_rate = self.get_exploration_rate(current_episode)
        if np.random.rand() < exploration_rate:
            return np.random.randint(4)
        else:
            return np.argmax(self.q_table[state])
        
    '''def update_q_table(self, state, action, next_state, reward):
        best_next_action = np.argmax(self.q_table[next_state])

        current_q_value = self.q_table[state][action]

        new_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * self.q_table[next_state][best_next_action] - current_q_value)
        return new_q_value'''
    
    #cgpt
    def update_q_table(self, state, action, next_state, reward):
        best_next_action = np.argmax(self.q_table[next_state])
        current_q_value = self.q_table[state][action]
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        self.q_table[state][action] = current_q_value + self.learning_rate * (td_target - current_q_value)


goal_reward = 100
wall_penalty = -10
step_penalty = -1

def finish_episode(agent, maze, current_episode, train=True):
    current_state = maze.start_position
    is_done = False
    episode_reward = 0
    episode_step = 0
    path = [current_state]
    
    while not is_done:

        action = agent.get_action(current_state,current_episode)

        #next_state = (current_state[0] + actions[action][0], current_state[1], actions[action][1])

        next_state = (
            current_state[0] + actions[action][0],  # row
            current_state[1] + actions[action][1],  # col
        )


        if next_state[0] < 0 or next_state[0] >= maze.maze_height or next_state[1] < 0 or next_state[1] >= maze.maze_width or maze.maze[next_state[1]][next_state[0]] == 1:
            reward = wall_penalty
            next_state = current_state

        elif next_state == (maze.goal_position):
             path.append(current_state)
             reward = goal_reward
             is_done = True

        else:
            path.append(current_state)
            reward = step_penalty

        episode_reward += reward
        episode_step += 1

        if train == True:
            agent.update_q_table(current_state, action, next_state, reward)

        current_state = next_state

    return episode_reward, episode_step, path

def test_agent(agent, maze, num_episodes=1):
    
    episode_reward, episode_step, path = finish_episode(agent, maze, num_episodes, train=False)

    print("Learned Path")
    for row, cols in path:
        print(f"({row}, {cols}->", end='')
    print("Goal")

    print("Number of steps:", episode_step)
    print("Total Reward:", episode_reward)

    if plt.gcf().get_axes():
        plt.cla()

    plt.figure(figsize=(5,5))
    plt.imshow(maze.maze, cmap="gray")

    plt.text(maze.start_position[0], maze.start_postion[1], 'S', ha='center', va='center',color='red', fontsize=20)
    plt.text(maze.goal_position[0], maze.goal_postion[1], 'G', ha='center', va='center',color='green', fontsize=20)

    for position in path:
        plt.text(position[0], position[1], "#", va='center', color='blue',fontsize=20)

    plt.xticks([]), plt.yticks([])
    plt.grid(color='black', linewidth=2)
    plt.show()
    agent = QLearningAgent(maze)
    test_agent(agent,maze)
    return episode_step, episode_reward

def train_agent(agent, maze, num_episodes=100):
    episode_rewards = []
    episode_steps = []

    for episode in range(num_episodes):
        episode_reward, episode_step, path = finish_episode(agent, maze, episode, train=True)

        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cummulative Reward')
    plt.title('Reward per Episode')

    average_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"The average reward is: {average_reward}")

    plt.subplot(1,2,2)
    plt.plot(episode_steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps Taken')
    plt.ylim(0,100)
    plt.title('Steps per Episode')

    average_steps = sum(episode_steps) / len(episode_steps)
    print(f"The average steps is: {average_steps}")

    plt.tight_layout()
    plt.show()

maze_layout = np.array([[]])
start_x = 1
start_y = 2
goal_x = 6
goal_y = 8
maze =  Maze(maze_overview)
maze.show_maze()

agent = QLearningAgent(maze)
train_agent(agent,maze)
test_agent(agent,maze)

 