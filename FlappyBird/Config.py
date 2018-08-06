import numpy as np
import random
from pathlib import Path


class Parameters():
    sys_width = 288
    sys_height = 512
    sys_frame_rate = 500
    sys_caption = "Flappy Bird"


    music_volume= 0
    mix_volume = 0

    bird_height = 24
    bird_width = 34
    bird_row_col = (4, 1)
    bird_location = np.array([60, 200])
    bird_frame_speed = int(sys_frame_rate/30)
    bird_moving_speed = np.array([0, 0])
    bird_moving_acc = np.array([0, 1])
    bird_actions = [None, np.array([0, -12])]

    ground_row_col = (1, 1)
    ground_frame_speed = 0
    ground_moving_speed = np.array([-3, 0])
    ground_moving_acc = np.array([0, 0])

    pipe_head = 20
    pipe_y_interval = 160
    pipe_y_range = (0, 400)  # (0,240)
    pipe_x_interval = 210
    pipe_num = 3
    pipe_width = 52
    pipe_height = 320
    pipe1_location = sys_width
    pipe2_location = sys_width + pipe_x_interval
    pipe3_location = sys_width + 2 * pipe_x_interval

    ground1_location = np.array([0, 400])
    ground2_location = np.array([336, 400])
    ground_num = 2

    font_size = 50
    font_location = (144, 30)

    state_x_min = -228
    state_x_max = 52
    state_y_min = -5 - 400 + pipe_head
    state_y_max = 400 - pipe_head - pipe_y_interval

    graph_factor = 30


class File_Path():
    font = 'FlappyBird/font/KGS_shadow.ttf'
    bg = ["FlappyBird/pic/BG1.png", "FlappyBird/pic/BG2.png"]
    bird = ["FlappyBird/pic/BD1.png", "FlappyBird/pic/BD2.png", "FlappyBird/pic/BD3.png"]
    ground = "FlappyBird/pic/GD.png"
    pipe = ["FlappyBird/pic/Pipe1.png", "FlappyBird/pic/Pipe2.png"]
    hit = "FlappyBird/wave/hit.wav"
    point = "FlappyBird/wave/point.wav"
    bgm = [str(p) for p in Path('FlappyBird/music').iterdir() if p.match('*.mp3')]
