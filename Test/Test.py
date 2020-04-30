from Model.SSR_net import SSR_net
import matplotlib.image as mpimg
import numpy as np

model = SSR_net(image_size=200, stage_num=[3, 3, 3], lambda_local=0.25, lambda_d=0.25)()

model.load_weights('../output/weights-improvement-44-6.48.h5')

X = mpimg.imread('hjc.png')
print(np.max(X))
X = X.reshape(1, 200, 200, 3)
print(model.predict(X))