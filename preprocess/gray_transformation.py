import os
import os.path
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance


# The neighborhood pixel of the pixel value >245 is identified as belonging to the background color
# If there are more than 2 pixel values of 4 pixels above and below a pixel
# The pixel belongs to the background color, then the pixel is the target point, otherwise it is noise
def denoising(im):
    pixdata = im.load()
    w, h = im.size
    for j in range(1, h - 1):
        for i in range(1, w - 1):
            count = 0
            if pixdata[i, j - 1] > 245:
                count = count + 1
            if pixdata[i, j + 1] > 245:
                count = count + 1
            if pixdata[i + 1, j] > 245:
                count = count + 1
            if pixdata[i - 1, j] > 245:
                count = count + 1
            if count > 2:
                pixdata[i, j] = 255
    return im


def imgTransfer(f_name):
    im = Image.open(f_name)
    im = im.filter(ImageFilter.MedianFilter(1))  # set up the filter
    im = ImageEnhance.Contrast(im).enhance(1.5)  # enhance the figure
    im = im.convert('L')  # gray transfer
    im = denoising(im)  # denoise the fig
    return im


os.makedirs("gray_data")

# The process of saving the figures and recode the figures
path = "face_age"
age_list = os.listdir(path)
count = 1
for age in age_list:
    # find the path of file names for ages
    fig_list = os.listdir(path + "/" + age)
    # a wrong folder of the figures should be removed
    if age == "face_age":
        continue
    for fig in fig_list:
        path_new = path + "/" + age + "/" + fig
        imgTransfer(path_new).save("gray_data/" + str(count) + ".png")
        count = count + 1

# The process of saving the fliped figures and recode the fliped figures
path = "face_age"
age_list = os.listdir(path)
for age in age_list:
    # find the path of file names for ages
    fig_list = os.listdir(path + "/" + age)
    # a wrong folder of the figures should be removed
    if age == "face_age":
        continue
    for fig in fig_list:
        path_new = path + "/" + age + "/" + fig
        imgTransfer(path_new).transpose(Image.FLIP_LEFT_RIGHT).save("gray_data/" + str(count) + ".png")
        count = count + 1
        print(count)
