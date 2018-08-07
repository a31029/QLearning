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
    def __init__(self,filename = np.random.choice(File_Path.bird),
                row_col_nums = Parameters.bird_row_col,
                location = Parameters.bird_location,
                frame_speed = Parameters.bird_frame_speed,
                moving_speed = Parameters.bird_moving_speed,
                moving_acc = Parameters.bird_moving_acc):

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

    def __init__(self, index,filename = File_Path.ground,
                row_col_nums = Parameters.ground_row_col,
                location = Parameters.ground_location,
                frame_speed = Parameters.ground_frame_speed,
                moving_speed = Parameters.ground_moving_speed,
                moving_acc= Parameters.ground_moving_acc,
                total = Parameters.ground_num):

        self.index = index
        self.total = total
        mySprite.__init__(self, filename, row_col_nums, location[self.index], frame_speed, moving_speed, moving_acc)

    def validate(self):
        if self.rect.x + self.rect.width < 0:
            self.rect[0] += self.total * self.rect.width



# 加载Q值的图 
class Graph(mySprite):
    def __init__(self,father,container,filename,
                row_col_nums = Parameters.ground_row_col,
                frame_speed = Parameters.ground_frame_speed,
                moving_speed = Parameters.ground_moving_speed,
                moving_acc = Parameters.ground_moving_acc):

        self.father = father
        self.container = container
        location = (self.father.rect.x,self.father.rect.y)
        mySprite.__init__(self, filename, row_col_nums, location,frame_speed, moving_speed, moving_acc)

    def move(self):
        pass

    def validate(self):
        if self.rect.x - Parameters.state_x_min - Parameters.bird_width/2 < self.father.rect.x:
            self.container.move_to_back(self)
        self.rect.x = self.father.rect.x + Parameters.state_x_min + Parameters.bird_width/2
        self.rect.y = self.father.rect.y + Parameters.state_y_min + Parameters.bird_height



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

    loc = {}
    flg = {}

    @classmethod
    def build(cls,total_num):
        for i in range(total_num):
            cls.loc[i] = {}
            cls.loc[i]['d'] ,cls.loc[i]['u'] = cls.randY()
            cls.flg[i] = True

    @classmethod
    def refresh(cls,n):
        if cls.flg[n]:
            cls.loc[n]['d'], cls.loc[n]['u'] = cls.randY()
        cls.flg[n] = cls.flg[n] ^ True

    @classmethod
    def randY(cls):
        d = int(np.random.uniform(cls.pipe_y_range[0] + cls.pipe_head+cls.pipe_y_interval, cls.pipe_y_range[1]-cls.pipe_head))
        # d = np.random.choice([cls.pipe_y_range[0] + cls.pipe_head+cls.pipe_y_interval, cls.pipe_y_range[1]-cls.pipe_head - 10])
        u = d - cls.pipe_y_interval - cls.pipe_height
        return d,u

    def __init__(self,index,up_down,
                filename = File_Path.pipe[0],
                row_col_nums = Parameters.ground_row_col,
                x_location = Parameters.pipe_location,
                frame_speed = Parameters.ground_frame_speed,
                moving_speed = Parameters.ground_moving_speed,
                moving_acc = Parameters.ground_moving_acc,
                total = Parameters.pipe_num):
            
        self.index = index
        self.up_down = up_down
        self.total = total

        location = (x_location[self.index],self.loc[self.index][self.up_down])
        mySprite.__init__(self, filename, row_col_nums, location,frame_speed, moving_speed, moving_acc)
        
        if self.up_down == 'u':
            self.roll()

    def validate(self):
        if self.rect.x + self.rect.width + Parameters.graph_factor < 0:
            self.refresh(self.index)
            self.rect[0] += self.pipe_x_interval * self.total
            self.rect[1] = self.loc[self.index][self.up_down]


# 构建一个 积分牌 类
class ScoreBord():
    def __init__(self,
                font_path = File_Path.font,
                font_size = Parameters.font_size,
                font_location = Parameters.font_location):
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
    def __init__(self, is_speed_in_state = False,vec = True,graph = False):
        pygame.init()
        pygame.freetype.init()
        pygame.display.set_caption(Parameters.sys_caption)
        pygame.mixer.music.set_volume(Parameters.music_volume)

        self.play_index = 0
        self.musicFlag = True
        self.is_speed_in_state = is_speed_in_state
        self.vec = vec
        self.graph = graph

        # 先建管子 再建图 
        # 更新的时候 先更新图，在更新管子，如果管子重置，图也跟着重置

    def _build(self):

        self._graphs_files = [str(p) for p in Path('FlappyBird/graph').iterdir() if p.match('*.png')]

        self._screen = pygame.display.set_mode((Parameters.sys_width, Parameters.sys_height))
        self._bg = pygame.image.load(np.random.choice(File_Path.bg))

        self._ground_group = pygame.sprite.LayeredUpdates()
        self._grounds = [Ground(i) for i in range(Parameters.ground_num)]
        Pipe.build(Parameters.pipe_num)
        self._pipes = [Pipe(i,j) for i in range(Parameters.pipe_num) for j in ['u','d']]
        
        self._flappy = pygame.sprite.Group()
        self._bird = Bird()

        self._score_bord = ScoreBord(File_Path.font, Parameters.font_size, Parameters.font_location)

        for obj in self._grounds:
            self._ground_group.add(obj,layer = 1) 
        for obj in self._pipes:
            self._ground_group.add(obj,layer = 0)
        self._flappy.add(self._bird)
        
        if self.graph:
            self._graph_group = pygame.sprite.LayeredUpdates()
            self._graphs = [Graph(obj,self._graph_group,self._graphs_files[0]) for obj in self._pipes if obj.up_down == 'd']
            for i,obj in enumerate(self._graphs):
                self._graph_group.add(obj,layer = Parameters.pipe_num - i)

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
        locations = np.array([[obj.rect.x,obj.rect.y] for obj in self._pipes if obj.up_down == 'd'])
        ind = locations[:, 0] > self._bird.rect.x - Parameters.pipe_width
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
        if pygame.sprite.spritecollideany(self._bird, self._ground_group) or self._bird.rect.y <= 0:
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
            while True:

                #设置时间
                clock.tick(Parameters.sys_frame_rate)

                #播放BGM
                self._playBGM()
                
                # 刷新图像
                self._flappy.update()
                self._ground_group.update()
                self._score_bord.update()

                self._screen.blit(self._bg, (0, 0))
                if self.graph:
                    self._graph_group.update()
                    self._graph_group.draw(self._screen)

                self._flappy.draw(self._screen)
                self._ground_group.draw(self._screen)
                self._score_bord.draw(self._screen)
                pygame.display.update()

                # 更新状态
                self._updateState()

                jump = 0

                # 获取跳跃动作
                for event in pygame.event.get():
                    if event.type == 2 and event.dict['key'] == 32:
                        jump = 1
                        break
                    if event.type == 2 and event.dict['key'] == 27:
                        pygame.quit()
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        jump = 1
                        break

                #判断是否修改动作
                if func is not None and self.vec:
                    jump = func(self.now, self.dead, jump)

                if func is not None and not self.vec :
                    jump = func(self.getImage(),self.dead,jump)

                self._bird.jump(Parameters.bird_actions[jump])

                if self.dead < 0:
                    break
