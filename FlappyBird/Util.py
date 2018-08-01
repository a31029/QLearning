import numpy as np
import random

class Index():
    def __init__(self, n, x_min, x_max, y_min, y_max):
        self._n = n
        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max
        self._shape = (int((self._x_max-self._x_min)/n) + 1,
                       int((self._y_max-self._y_min)/n) + 1)

    def size(self):
        return self._shape

    def length(self):
        return self._shape[0] * self._shape[1]

    def trans2d(self, now):
        if len(now) == 4:
            x1, y1, x2, y2 = now
        elif len(now) == 5:
            x1, y1, x2, y2, speed = now
        x = x1 - x2 - self._x_min
        y = y1 - y2 - self._y_min
        int_x, int_y = int(x/self._n), int(y/self._n)
        if int_y >= self._shape[1]:
            int_y = self._shape[1]-1
            print('y 超过范围了')
            print(y1, y2)
        if int_x < 0 or int_y < 0 or int_x >= self._shape[0]:
            print("错啦")
            print(x1, y1, x2, y2)
            print("错啦")
            1/0
        if len(now) == 4:
            return int_x, int_y
        elif len(now) == 5:
            return int_x, int_y, speed
