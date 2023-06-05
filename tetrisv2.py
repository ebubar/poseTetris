#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pygame opencv-python mediapipe')


# In[1]:


import pygame
import cv2
import mediapipe as mp
import os
import random
import numpy as np

'''
Thank you "Tech With Tim" and "Murtaza's Workshop" for great free python tutorials on youtube that help me so much with this work
Tech With Tim | Making Tetris Tutorial: https://www.youtube.com/watch?v=uoR4ilCWwKA
Murtaza's Workshop | Hand Tracking Tutorial: https://www.youtube.com/watch?v=p5Z_GGRCI5s
'''

#Getting mediapipe Pose ready
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a) # Start point
    b = np.array(b) # Mid point
    c = np.array(c) # End point
        
    ba = a-b
    bc = c-b
    
    cosine_angle = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)
    
    radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle2d = np.abs(radians*180/np.pi)
    
    if angle2d >180.0:
        angle2d = 360-angle2d
    
    return angle

# Will determine the distance between two points given two vector inputs
def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    
    distance = np.linalg.norm(b-a)
    return distance

#Capture webcam
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#Prepare pygame window position, fonts and background music
os.environ['SDL_VIDEO_WINDOW_POS'] ="0,0"
pygame.init()
pygame.font.init()
pygame.mixer.init()
pygame.mixer.music.load('tetris.mp3')

#Global variables
screen_size = pygame.display.get_desktop_sizes()[0]
s_width = screen_size[0]
s_height = screen_size[1]
play_width = 300  # meaning 300 // 10 = 30 width per block
play_height = 600  # meaning 600 // 20 = 30 height per block
block_size = 30
top_left_x = (s_width - play_width) // 2
top_left_y = (s_height - play_height) // 3

#Initialize the body segments
#Get coordinates of body landmarks and save them as more logical values
#HEAD
head_points = [0,0,0]
            
#UPPER BODY
left_shoulder = [0,0,0]
left_elbow = [0,0,0]
left_wrist = [0,0,0]

right_shoulder = [0,0,0]
right_elbow = [0,0,0]
right_wrist = [0,0,0]
            
#LOWER BODY
left_hip = [0.,0.,0.]
left_knee = [0.,0.,0.]
left_ankle = [0.,0.,0.]
left_heel = [0.,0.,0.]            
left_toe = [0.,0.,0.]
right_hip = [0.,0.,0.]
right_knee = [0.,0.,0.]
right_ankle = [0.,0.,0.]
right_heel = [0.,0.,0.]
right_toe = [0.,0.,0.]

#Initialize relevant angles
right_shoulder_angle = 0.
left_shoulder_angle = 0.
right_elbow_angle = 0.
left_elbow_angle = 0.
right_knee_angle = 0.
left_knee_angle = 0.
right_hip_angle = 0.
left_hip_angle = 0.
right_ankle_angle = 0.
left_ankle_angle = 0.           

#The shapes with all possible rotations
S = [['.....',
      '.....',
      '..00.',
      '.00..',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]

Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]

I = [['..0..',
      '..0..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]

O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]

L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]

T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]

#index 0-6 get you a shape and its corresponding colours
shapes = [S, Z, I, O, J, L, T]
shape_colors = [(0, 230, 115), (255, 51, 51), (0, 204, 255), (255, 255, 128), (0, 102, 255), (255, 140, 26), (204, 51, 255)]

#Class for the Shapes
class Piece(object):  # *
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]
        self.rotation = 0

#create the grid
def create_grid(locked_pos={}):  # *
    grid = [[(0,0,0) for _ in range(10)] for _ in range(20)]

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if (j, i) in locked_pos:
                c = locked_pos[(j,i)]
                grid[i][j] = c
    return grid

#convert the shapes into its positions
def convert_shape_format(shape):
    positions = []
    format = shape.shape[shape.rotation % len(shape.shape)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                positions.append((shape.x + j, shape.y + i))

    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)

    return positions

#test whether or not the falling shape is in a valid space
def valid_space(shape, grid):
    accepted_pos = [[(j, i) for j in range(10) if grid[i][j] == (0,0,0)] for i in range(20)]
    accepted_pos = [j for sub in accepted_pos for j in sub]

    formatted = convert_shape_format(shape)

    for pos in formatted:
        if pos not in accepted_pos:
            if pos[1] > -1:
                return False
    return True

#check whether or not the user have lost
def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y < 1:
            return True

    return False

#get a random shape
def get_shape():
    return Piece(5, 0, random.choice(shapes))

#put a text in the middle of the screen
def draw_text_middle(surface, text, size, color):
    font = pygame.font.SysFont("britannic", size, bold=True)
    label = font.render(text, 1, color)

    surface.blit(label, (top_left_x + play_width /2 - (label.get_width()/2), top_left_y + play_height/2 - label.get_height()/2))

#draw the lines onto the grid
def draw_grid(surface, grid):
    sx = top_left_x
    sy = top_left_y

    for i in range(len(grid)):
        pygame.draw.line(surface, (128,128,128), (sx, sy + i*block_size), (sx+play_width, sy+ i*block_size))
        for j in range(len(grid[i])):
            pygame.draw.line(surface, (128, 128, 128), (sx + j*block_size, sy),(sx + j*block_size, sy + play_height))

#clear a row
def clear_rows(grid, locked):

    inc = 0
    for i in range(len(grid)-1, -1, -1):
        row = grid[i]
        if (0,0,0) not in row:
            inc += 1
            ind = i
            for j in range(len(row)):
                try:
                    del locked[(j,i)]
                except:
                    continue

    if inc > 0:
        for key in sorted(list(locked), key=lambda x: x[1])[::-1]:
            x, y = key
            if y < ind:
                newKey = (x, y + inc)
                locked[newKey] = locked.pop(key)

    return inc

#draw the window that shows the next shape
def draw_next_shape(shape, surface):
    font = pygame.font.SysFont('britannic', 30)
    label = font.render('Next Shape', 1, (255,255,255))

    sx = top_left_x + play_width + 50
    sy = top_left_y + play_height/2 - 100
    format = shape.shape[shape.rotation % len(shape.shape)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                pygame.draw.rect(surface, shape.color, (sx + j*block_size, sy + i*block_size, block_size, block_size), 0)

    surface.blit(label, (sx + 10, sy - 40))

#draw the main window
def draw_window(surface, grid, score=0):
    surface.fill((0, 0, 0))

    pygame.font.init()
    font = pygame.font.SysFont('britannic', 60)
    label = font.render('BALANCE TETRIS', 1, (255, 255, 255))

    surface.blit(label, (top_left_x + play_width / 2 - (label.get_width() / 2), 15))

    #show current score
    font = pygame.font.SysFont('britannic', 30)
    label = font.render('Score: ' + str(score), 1, (255,255,255))

    sx = top_left_x + play_width + 50
    sy = top_left_y + play_height/2 - 100

    surface.blit(label, (sx + 20, sy + 160))

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            pygame.draw.rect(surface, grid[i][j], (top_left_x + j*block_size, top_left_y + i*block_size, block_size, block_size), 0)

    pygame.draw.rect(surface, (215, 215, 215), (top_left_x, top_left_y, play_width, play_height), 5)

    draw_grid(surface, grid)

#add scores that correspond to the amount of rows cleared
def add_score(rows):
    conversion = {
        0: 0,
        1: 40,
        2: 100,
        3: 300,
        4: 1200
    }
    return conversion.get(rows)

#THE MAIN FUNCTION THAT RUNS THE GAME
def main(win):
    locked_positions = {}
    grid = create_grid(locked_positions)

    change_piece = False
    run = True
    current_piece = get_shape()
    next_piece = get_shape()
    clock = pygame.time.Clock()
    fall_time = 0
    fall_speed_real = 2
    fall_speed = fall_speed_real
    level_time = 0
    score = 0

    left_wait = 0
    right_wait = 0
    rotate_wait = 0.3
    down_wait = 0
    fall_speed_down = 0.05

    #THE MAIN WHILE LOOP
    while run:
        grid = create_grid(locked_positions)

        fall_time += clock.get_rawtime()
        level_time += clock.get_rawtime()
        clock.tick()

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            success, frame = cam.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image,1)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                landmarks = results.pose_landmarks.landmark
                #Get coordinates of body landmarks and save them as more logical values
                #HEAD
                head_points = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y, landmarks[mp_pose.PoseLandmark.NOSE.value].z]
                #UPPER BODY
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
                #LOWER BODY
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
                left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].z]            
                left_toe = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
                right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].z]
                right_toe = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z]
                #Calculate relevant angles
                right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
                left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_hip_angle = calculate_angle(right_knee, right_hip, right_shoulder)
                left_hip_angle = calculate_angle(left_knee, left_hip, left_shoulder)
                right_ankle_angle = calculate_angle(right_knee, right_heel, right_toe)
                left_ankle_angle = calculate_angle(left_knee, left_heel, left_toe) 
                
                
                if (left_heel[1] < 0.8):
                    print("LEFT = ", left_heel[1])
                    left_wait += 1
                if (right_heel[1] < 0.8):
                    print("RIGHT = ",right_heel[1])
                    right_wait += 1
                if (right_hip_angle > 140) and (right_hip_angle) < 160:
                    rotate_wait += 1
                if (left_hip_angle > 140) and (left_hip_angle < 160):
                    rotate_wait += 1
                if (left_knee_angle < 90.0) and (right_knee_angle < 90):
                    down_wait += 1

            except:
                pass

        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.namedWindow("BALANCE CAM")
        cv2.moveWindow("BALANCE CAM", 0, 121)
        cv2.imshow("BALANCE CAM", image)
        cv2.waitKey(10)
        if cv2.waitKey(10) == 27: # This puts you out of the loop above if you hit q
            pygame.display.quit()
            cam.release() # Releases the webcam from your memory
            cv2.destroyAllWindows() # Closes the window for the webcam
            quit()

            #if enough time (fall_speed) have passsed, piece moves down 1 block
        if fall_time/1000 > fall_speed:
            fall_time = 0
            current_piece.y += 1
            if not(valid_space(current_piece, grid)) and current_piece.y > 0:
                current_piece.y -= 1
                change_piece = True

        #"if you gesture to the LEFT for at least 4 frames, piece move LEFT"
        if left_wait >= 4:
            current_piece.x -= 1
            if not (valid_space(current_piece, grid)):
                current_piece.x += 1
            left_wait = 0
            right_wait = 0
            rotate_wait = 0
            down_wait = 0

        #"if you gesture to the RIGHT for at least 4 frames, piece move RIGHT"
        if right_wait >= 4:
            current_piece.x += 1
            if not (valid_space(current_piece, grid)):
                current_piece.x -= 1
            left_wait = 0
            right_wait = 0
            rotate_wait = 0
            down_wait = 0

        #"if you gesture to ROTATE  for at least 4 frames, piece ROTATES"
        if rotate_wait >= 3:
            current_piece.rotation += 1
            if not (valid_space(current_piece, grid)):
                current_piece.rotation -= 1
            left_wait = 0
            right_wait = 0
            rotate_wait = 0
            down_wait = 0

        #"if you gesture to go DOWN (no hand on the screen) for at least 5 frames, piece go DOWN (moves very fast)"
        if down_wait >= 4:
            fall_speed = fall_speed_down
            left_wait = 0
            right_wait = 0
            rotate_wait = 0
            down_wait = 0

        shape_pos = convert_shape_format(current_piece)

        #colour the grid where the shape is
        for i in range(len(shape_pos)):
            x, y = shape_pos[i]
            if y > -1:
                grid[y][x] = current_piece.color

        if change_piece:
            for pos in shape_pos:
                p = (pos[0], pos[1])
                locked_positions[p] = current_piece.color
            current_piece = next_piece
            next_piece = get_shape()
            change_piece = False
            score += add_score(clear_rows(grid, locked_positions))
            fall_speed = fall_speed_real
            down_wait = 0

        draw_window(win, grid, score)
        draw_next_shape(next_piece, win)
        pygame.display.update()
        
        if check_lost(locked_positions):
            draw_text_middle(win, "YOU LOST!", 80, (255,255,255))
            pygame.display.update()
            pygame.time.delay(1500)
            run = False

#Menu screen that will lead to the main function
def main_menu(win):
    run = True
    while run:
        win.fill((0,0,0))
        draw_text_middle(win, 'Press Any Key To Start', 60, (255,255,255))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                pygame.mixer.music.play()
                main(win)
    pygame.display.quit()

#win = pygame.display.set_mode((s_width, s_height))
screen_size = pygame.display.get_desktop_sizes()[0]
win = pygame.display.set_mode(screen_size,pygame.FULLSCREEN)
pygame.display.set_caption('BALANCE TETRIS')
try:
    main_menu(win)
except:
    pygame.quit()



# In[ ]:


print(left_knee_angle)


# In[ ]:




