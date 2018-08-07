import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

from FlappyBird.Config import Parameters
from FlappyBird.Util import Index
from PIL import Image

x = np.load('Algorithm/QLearning/Matrix/Q.npy')

diff = x[:,:,0] - x[:,:,1]
diff[diff>0]= 1
diff[diff<0] = -1
diff = diff.transpose(1, 0)

x,y = diff.shape
index = Index(4, Parameters.state_x_min, Parameters.state_x_max,
              Parameters.state_y_min, Parameters.state_y_max)

image = misc.imresize(
    diff, (index._y_max-index._y_min, index._x_max-index._x_min))
plt.imsave('FlappyBird/graph/zz.png',image, cmap=plt.cm.gray)


