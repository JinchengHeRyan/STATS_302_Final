from Model.SSR_net import SSR_net
import matplotlib.image as mpimg

model = SSR_net(image_size=200, stage_num=[3, 3, 3], lambda_local=0.25, lambda_d=0.25)()

model.load_weights('../output/ssrnet_3_3_3_64_1.0_1.0.h5')

X = mpimg.imread('syydd.png')
X = X.reshape(1, 200, 200, 3)
print(model.predict(X))