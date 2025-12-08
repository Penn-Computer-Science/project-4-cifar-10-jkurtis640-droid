import turtle

tile_size = 30
##this is a change

maze_overview = [
    "XXXXXXXXXXXXXXX"
    "X   X       XGX"
    "X X X XXXXX X X"
    "X X X     X X X"
    "X X XXXXX X X X"
    "X X     X X   X"
    "X XXXXX X XXX X"
    "XS      X     X"
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

turtle.done()









    