# judge if the prediction is over 20
from keras import models
import numpy as np

model_wrn_4 = models.load_model("../Model/migration_model/wrn_4.h5")
model_dense_4 = models.load_model("../Model/migration_model/densenet_4.h5")
model_wrn_7 = models.load_model("../Model/migration_model/wrn_7.h5")
model_dense_7 = models.load_model("../Model/migration_model/densenet_7.h5")


def find_mig(prediction, fig):
    if prediction >= 20:
        # Predict the wide resnet for 4 classification
        y_4_1 = model_wrn_4.predict(fig)

        # Predict the dense net for 4 classification
        y_4_2 = model_dense_4.predict(fig)

        y_4_2 = y_4_2 / np.sum(y_4_2)

        # Predict the wide resnet for 7 classification
        y_7_1 = model_wrn_7.predict(fig)

        # Predict the dense net for 7 classification
        y_7_2 = model_dense_7.predict(fig)

        y_7_2 = y_7_2 / np.sum(y_7_2)

        # Calculate the exp value of each model
        y_4_1_exp = y_4_1.dot([12.5, 37.5, 62.5, 87.5])
        y_4_2_exp = y_4_2.dot([12.5, 37.5, 62.5, 87.5])

        y_7_1_exp = y_7_1.dot([2.5, 10, 22.5, 37.5, 52.5, 67.5, 87.5])

        y_7_2_exp = y_7_2.dot([2.5, 10, 22.5, 37.5, 52.5, 67.5, 87.5])

        # calculate the norm distribution
        def norm(mu, sigma):
            x = np.linspace(1, 100, 100)
            return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)

        y_4_1_norm = norm(y_4_1_exp, 12.5)
        y_4_2_norm = norm(y_4_2_exp, 12.5)
        y_7_1_norm = norm(y_7_1_exp, 7.5)
        y_7_2_norm = norm(y_7_2_exp, 7.5)

        if prediction <= 30:
            sum_norm = y_4_1_exp + y_7_1_exp
        else:
            sum_norm = y_4_2_exp + y_7_2_exp

        # sum up the distribution
        # sum_norm = y_4_1_norm + y_4_2_norm + y_7_1_norm + y_7_2_norm
        # sum_norm = y_4_2_norm + y_7_1_norm + y_7_2_norm
        # sum_norm = y_4_1_exp+ y_4_2_exp + y_7_1_exp + y_7_2_exp

        # find the max distribution
        # striction = np.max(sum_norm) * 0.9
        # judge = sum_norm >= striction
        # # calcuate the min for the range
        # min = 0
        # for i in judge:
        #     if i:
        #         break
        #     min += 1
        # # calculate the max for the range
        # max = len(judge)
        # for i in judge:
        #     if i:
        #         break
        #     max -= 1
        # # get the symbol
        # print("min = ", min)
        # print("max = ", max)
        # if prediction <= min + 1:
        #     return '+'
        # if prediction >= max + 1:
        #     return '-'
        # else:
        #     return ''
        if sum_norm / 2 <= prediction - 5:
            return '-'
        elif sum_norm / 2 >= prediction + 5:
            return '+'
        else:
            return ''
    else:
        return ''
