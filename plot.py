import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

from FlappyBird.Config import Parameters
from FlappyBird.Util import Index

x = np.load('Algorithm/QLearning/Matrix/Q.npy')

# x = np.array([[10,3],[0,5]])
# y = np.array([[3, 10], [5, 0]])

# x = np.stack((y,x))

# print(x)
diff = x[:,:,0]-x[:,:,1]
diff[diff>0]= 1
diff[diff<0] = -1
diff = diff.transpose(1, 0)

x,y = diff.shape
index = Index(7, Parameters.state_x_min, Parameters.state_x_max,
              Parameters.state_y_min, Parameters.state_y_max)


image = misc.imresize(
    diff, (index._y_max-index._y_min, index._x_max-index._x_min))
plt.imsave('/home/liuxiang/Desktop/1.png',image, cmap=plt.cm.gray)


