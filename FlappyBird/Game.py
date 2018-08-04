import pygame
import pygame.freetype

import numpy as np
from pathlib import Path

from FlappyBird.Config import Parameters
from FlappyBird.Config import File_Path


# 新建了一个sprite 用于存放 游戏中所有的 对象
# 包括几个功能:
# 1.读取图片
# 2.旋转(主要是pipe需要旋转)
# 3.移动速度(pipe ground graph 其实都是水平向左运动 小鸟是上下运动 需要分别设置)
# 4.刷新图片(小鸟的翅膀是需要扇动的 所以需要进行刷新 其他的都不需要)
# 5.移动 根据移动速度进行移动
# 6.验证 (验证小鸟是否出框 验证groud 验证oipe 是否出框 如果出框 就需要要重置)
# 7.更新 
class mySprite(pygame.sprite.Sprite):
    def __init__(self, filename, row_col_nums, location, frame_speed, moving_speed, moving_acc):
        pygame.sprite.Sprite.__init__(self)
        self.images = {}
        self._loadIamge(filename,row_col_nums)
        self.idx = 0
        self.image = self.images[self.idx]
        self.rect = pygame.Rect(location[0], location[1], self.image.get_width(), self.image.get_height())
        self.frame_speed = frame_speed
        self.moving_speed = moving_speed
        self.moving_acc = moving_acc
        self.frame_counter = 0


    def _loadIamge(self,filename,row_col_nums):
        ori_image = pygame.image.load(filename).convert_alpha()
        ori_rect = ori_image.get_rect()
        image_width = int(ori_rect.width/row_col_nums[1])
        image_height = int(ori_rect.height/row_col_nums[0])
        for i in range(row_col_nums[0]):
            for j in range(row_col_nums[1]):
                rect_tmp = (j * image_width,
                            i * image_height,
                            image_width,
                            image_height
                            )
                self.images[i*row_col_nums[1] + j] = ori_image.subsurface(rect_tmp)

    def roll(self):
        self.image = pygame.transform.flip(self.image, False, True)

    def setMoving_speed(self, moving_speed):
        self.moving_speed = moving_speed

    def refreshImage(self):
        if self.frame_speed > 0 :
            self.frame_counter += 1
            if self.frame_counter >= self.frame_speed:
                self.frame_counter = 0
                self.idx += 1
                if self.idx == len(self.images):
                    self.idx = 0
                self.image = self.images[self.idx]

    def move(self):
        self.moving_speed = self.moving_speed + self.moving_acc
        self.rect.x = self.rect.x + self.moving_speed[0]
        self.rect.y = self.rect.y + self.moving_speed[1]

    def validate(self):
        pass

    def update(self):
        self.refreshImage()
        self.move()
        self.validate()

# 继承mysprite，构建小鸟 
# 重写validate 函数(验证是否到达上框) 并添加一个 jump函数(其实就是设置moving_speed)
class Bird(mySprite):
    def __init__(self, filename, row_col_nums, location, frame_speed, moving_speed, moving_acc):
        mySprite.__init__(self, filename, row_col_nums, location, frame_speed, moving_speed, moving_acc)
        # self.secondJump = True

    def validate(self):
        if self.rect[1] <= 0:
            self.rect[1] = 0
            self.setMoving_speed(np.array([0, 0]))

    
    def jump(self, moving_speed):
        if moving_speed is not None:
            self.setMoving_speed(moving_speed)

# 继承mysprite，构建地面
# 重写validate 函数(验证是否出左边框) 
class Ground(mySprite):
    def __init__(self, filename, row_col_nums, location, frame_speed, moving_speed, moving_acc,num):
        mySprite.__init__(self, filename, row_col_nums, location, frame_speed, moving_speed, moving_acc)
        self.num = num

    def validate(self):
        if self.rect.x + self.rect.width < 0:
            self.rect[0] += self.num * self.rect.width

# 构建一个graph类，把Q值的图片加载在游戏背景上 方便debug
# 加载Q值的图 
class Graph(mySprite):
    def __init__(self, filename, row_col_nums, location, frame_speed, moving_speed, moving_acc,container):
        mySprite.__init__(self, filename, row_col_nums, location,frame_speed, moving_speed, moving_acc)
        self.container = container

# 继承mysprite 构建pipe类
class Pipe(mySprite):
    pipe_head = Parameters.pipe_head
    pipe_x_interval = Parameters.pipe_x_interval
    pipe_y_interval = Parameters.pipe_y_interval
    pipe_y_range = Parameters.pipe_y_range
    pipe_height = Parameters.pipe_height
    state_x_min = Parameters.state_x_min
    state_y_min = Parameters.state_y_min
    bird_height = Parameters.bird_height
    bird_width = Parameters.bird_width

    @classmethod
    def resetY(cls):
        cls.yd = int(np.random.uniform(
            cls.pipe_y_range[0] + cls.pipe_head+cls.pipe_y_interval, cls.pipe_y_range[1]-cls.pipe_head))
        cls.yu = cls.yd - cls.pipe_y_interval - cls.pipe_height
        cls.counter = True

    @classmethod
    def counts(cls):
        cls.counter = cls.counter ^ True
        if cls.counter is True:
            cls.resetY()

    def __init__(self, filename, row_col_nums, x_location, frame_speed, moving_speed, moving_acc, up_down,num, graph_file=None,graph_row_col=None,graph_container = None):
        self.num = num
        self.up_down = up_down

        location = (x_location,self.yu if up_down else self.yd)
        mySprite.__init__(self, filename, row_col_nums, location,
                                  frame_speed, moving_speed, moving_acc)
        self.counts()
        #如果管子是down 则会同时创建一个图
        if self.up_down is True:
            self.roll()
            self.graph = None
        else:
            if graph_file is None or graph_container is None :
                # print("alert:需要一个Q值图和图的容器!!!!!")
                self.graph = None
                self.container = None
            else:
                # 最原始的Q值矩阵中 上下限 分别为-228,52, -360,400-40-self._pipe_y_interval
                # 0,0 点 表示 鸟的左上角 和 管子左上角 重叠
                # 经过平移  图的左上角 为  管子左上角.x +228 管子左上角.y + 360
                # 为了图好看 又把 图往右下角移动了半个鸟的身位置
                graph_location = (self.rect[0] + self.state_x_min + self.bird_width/2,
                                  self.rect[1] + self.state_y_min + self.bird_height)
                self.graph = Graph(graph_file, graph_row_col, graph_location,
                                   self.frame_speed, self.moving_speed, self.moving_acc, graph_container)

    # 每次管子重置位置的时候，会连同图的位置一起重置
    # 同时重置的时候，会把图的位置移动到图层的最后一层
    def _resetGraph(self):
        graph_location = (self.rect[0] + self.state_x_min + self.bird_width/2,
                          self.rect[1] + self.state_y_min + self.bird_height)
        self.graph.rect[0] = graph_location[0]
        self.graph.rect[1] = graph_location[1]
        self.graph.container.move_to_back(self.graph)

    def validate(self):
        if self.rect.x + self.rect.width + Parameters.graph_factor < 0:
            self.rect[0] += self.pipe_x_interval * self.num
            self.rect[1] = self.yu if self.up_down else self.yd
            self.counts()
            if self.graph is not None:
                self._resetGraph()

# 构建一个 积分牌 类
class ScoreBord():
    def __init__(self,font_path,font_size,font_location):
        self.score = 0
        self.font = pygame.freetype.Font(font_path)
        self.font_size = font_size
        self.font_location = font_location
        self.image = None
        self.prev = False

    def incre(self, add_one_point):
        add = False
        if self.prev ^ add_one_point and add_one_point:
            self.score += 1
            add = True
        self.prev = add_one_point
        return add

    def update(self):
        self.image, self.location = self.font.render(
            "%d" % self.score, size=self.font_size)

    def draw(self, screen):
        screen.blit(
            self.image, (self.font_location[0] - self.location.width / 2, self.font_location[1]))



# 构建游戏
class FlappyBird():
    def __init__(self, is_speed_in_state = False,vec = True,graph = False,interval = 4):
        pygame.init()
        pygame.freetype.init()
        pygame.display.set_caption(Parameters.sys_caption)
        pygame.mixer.music.set_volume(Parameters.music_volume)

        self.play_index = 0
        self.musicFlag = True
        self.is_speed_in_state = is_speed_in_state
        self.vec = vec
        self.graph = graph
        self.interval = interval

        # 先建管子 再建图 
        # 更新的时候 先更新图，在更新管子，如果管子重置，图也跟着重置

    def _build(self):
        self._screen = pygame.display.set_mode((Parameters.sys_width, Parameters.sys_height))
        self._bg = pygame.image.load(np.random.choice(File_Path.bg))

        self._groud_group = pygame.sprite.LayeredUpdates()
        if self.graph:
            self._graph_group = pygame.sprite.LayeredUpdates()
        self._flappy = pygame.sprite.Group()

        self._ground1 = Ground(File_Path.ground, Parameters.ground_row_col, Parameters.ground1_location,Parameters.ground_frame_speed,Parameters.ground_moving_speed,Parameters.ground_moving_acc,Parameters.ground_num)
        self._ground2 = Ground(File_Path.ground, Parameters.ground_row_col, Parameters.ground2_location,Parameters.ground_frame_speed, Parameters.ground_moving_speed, Parameters.ground_moving_acc, Parameters.ground_num)

        self._graphs = [str(p) for p in Path('FlappyBird/graph').iterdir() if p.match('*.png')]

        pipeFile = np.random.choice(File_Path.pipe)
        Pipe.resetY()
        self._pipe1u = Pipe(pipeFile,Parameters.ground_row_col,Parameters.pipe1_location,Parameters.ground_frame_speed,Parameters.ground_moving_speed,Parameters.ground_moving_acc,True,Parameters.pipe_num)
        if self.graph:
            self._pipe1d = Pipe(pipeFile,Parameters.ground_row_col,Parameters.pipe1_location,Parameters.ground_frame_speed,Parameters.ground_moving_speed,Parameters.ground_moving_acc,False,Parameters.pipe_num,self._graphs[0],Parameters.ground_row_col,self._graph_group)
        else:
            self._pipe1d = Pipe(pipeFile,Parameters.ground_row_col,Parameters.pipe1_location,Parameters.ground_frame_speed,Parameters.ground_moving_speed,Parameters.ground_moving_acc,False,Parameters.pipe_num)
        self._pipe2u = Pipe(pipeFile,Parameters.ground_row_col,Parameters.pipe2_location,Parameters.ground_frame_speed,Parameters.ground_moving_speed,Parameters.ground_moving_acc,True,Parameters.pipe_num)
        if self.graph:
            self._pipe2d = Pipe(pipeFile,Parameters.ground_row_col,Parameters.pipe2_location,Parameters.ground_frame_speed,Parameters.ground_moving_speed,Parameters.ground_moving_acc,False,Parameters.pipe_num,self._graphs[0],Parameters.ground_row_col,self._graph_group)
        else:
            self._pipe2d = Pipe(pipeFile,Parameters.ground_row_col,Parameters.pipe2_location,Parameters.ground_frame_speed,Parameters.ground_moving_speed,Parameters.ground_moving_acc,False,Parameters.pipe_num)
        self._pipe3u = Pipe(pipeFile,Parameters.ground_row_col,Parameters.pipe3_location,Parameters.ground_frame_speed,Parameters.ground_moving_speed,Parameters.ground_moving_acc,True,Parameters.pipe_num)
        if self.graph:
            self._pipe3d = Pipe(pipeFile,Parameters.ground_row_col,Parameters.pipe3_location,Parameters.ground_frame_speed,Parameters.ground_moving_speed,Parameters.ground_moving_acc,False,Parameters.pipe_num,self._graphs[0],Parameters.ground_row_col,self._graph_group)
        else:
            self._pipe3d = Pipe(pipeFile,Parameters.ground_row_col,Parameters.pipe3_location,Parameters.ground_frame_speed,Parameters.ground_moving_speed,Parameters.ground_moving_acc,False,Parameters.pipe_num)
        
        self._groud_group.add(self._ground1, layer=1)
        self._groud_group.add(self._ground2, layer=1)
        self._groud_group.add(self._pipe1u,layer=0)
        self._groud_group.add(self._pipe1d,layer=0)
        self._groud_group.add(self._pipe2u,layer=0)
        self._groud_group.add(self._pipe2d,layer=0)
        self._groud_group.add(self._pipe3u,layer=0)
        self._groud_group.add(self._pipe3d,layer=0)
        
        if self.graph:
            self._graph_group.add(self._pipe1d.graph,layer = 3)
            self._graph_group.add(self._pipe2d.graph,layer = 2)
            self._graph_group.add(self._pipe3d.graph,layer = 1)

        self._bird = Bird(np.random.choice(File_Path.bird), Parameters.bird_row_col,Parameters.bird_location,
                          Parameters.bird_frame_speed, Parameters.bird_moving_speed, Parameters.bird_moving_acc)
        self._flappy.add(self._bird)

        self._score_bord = ScoreBord(File_Path.font, Parameters.font_size, Parameters.font_location)
        
        if self.is_speed_in_state:
            self.now = (0,0,0,0,0)
        else:
            self.now = (0,0,0,0)
        self.add_one_point = False


    def _playSound(self, filename):
        s = pygame.mixer.Sound(filename)
        s.set_volume(Parameters.mix_volume)
        s.play()

    def _playBGM(self):
        # 播放音乐
        if pygame.mixer.music.get_busy() == False and self.musicFlag == True:
            if self.play_index >= len(File_Path.bgm):
                self.play_index = 0
            pygame.mixer.music.load(File_Path.bgm[self.play_index])
            pygame.mixer.music.play()
            self.musicFlag = False
            self.play_index += 1
        elif pygame.mixer.music.get_busy():
            self.musicFlag = True

    def _updateState(self):
        #计算状态
        locations = np.array([[self._pipe1d.rect[0], self._pipe1d.rect[1]],[self._pipe2d.rect[0], self._pipe2d.rect[1]],[self._pipe3d.rect[0], self._pipe3d.rect[1]]])
        ind = locations[:, 0] > self._bird.rect.x - self._pipe1d.rect.width
        pip_x, pip_y = (locations[ind][np.argmin(locations[ind],axis = 0)[0],:])

        if self.is_speed_in_state:
            self.now = (self._bird.rect.x, self._bird.rect.y, pip_x, pip_y ,self._bird.moving_speed[1])
        else:
            self.now = (self._bird.rect.x, self._bird.rect.y, pip_x, pip_y)


        #判断是否得分
        if pip_x < self._bird.rect.x + self._bird.rect.width:
            self._add_one_point = True
        else:
            self._add_one_point = False
        if self._score_bord.incre(self._add_one_point):
            self._playSound(File_Path.point)

        self.dead = self._score_bord.score

        #碰撞检测
        if pygame.sprite.spritecollideany(self._bird, self._groud_group) or self._bird.rect.y <= 0:
            self._playSound(File_Path.hit)
            self.dead = -10

    def getImage(self):
        return pygame.surfarray.array3d(self._screen)

    def get_screen(self):
        return self._screen

    def run(self, func=None):

        while(True):
            self._build()
            clock = pygame.time.Clock()
            c_ter = 0
            while True:

                c_ter += 1

                #设置时间
                clock.tick(Parameters.sys_frame_rate)

                #播放BGM
                self._playBGM()
                
                # 刷新图像
                # 先更新图，再更新管子
                # 画的时候，先画图 在画管子和小鸟
                if self.graph:
                    self._graph_group.update()
                self._flappy.update()
                self._groud_group.update()
                self._score_bord.update()
                self._screen.blit(self._bg, (0, 0))
                if self.graph:
                    self._graph_group.draw(self._screen)
                self._flappy.draw(self._screen)
                self._groud_group.draw(self._screen)
                self._score_bord.draw(self._screen)
                pygame.display.update()

                # 更新状态
                self._updateState()

                jump = 0

                # 获取跳跃动作
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                    if event.type == pygame.K_SPACE:
                        jump = 1
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        jump = 1

                #判断是否修改动作
                if func is not None and self.vec and (c_ter % self.interval == 0 or self.dead < 0):
                    jump = func(self.now, self.dead, jump)

                if func is not None and not self.vec and (c_ter % self.interval == 0 or self.dead <0):
                    jump = func(self.getImage(),self.dead,jump)

                self._bird.jump(Parameters.bird_actions[jump])

                if self.dead < 0:
                    break
