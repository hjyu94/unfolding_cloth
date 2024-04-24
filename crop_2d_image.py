import numpy as np
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from PIL import Image
import matplotlib.pyplot as plt

# 예시로 주어진 이미지 배열
input_image_path = "/home/hjeong/code/unfolding_cloth/datasets/temp/image_left.png"
image = load_rgb(input_image_path)

# 주어진 좌표와 크기
# x_center = 1100
# y_center = 900
# offset = 300


# 이미지를 잘라냅니다.
# cropped_image = image[y_center - offset : y_center + offset, x_center - offset : x_center + offset]
cropped_image = image[186:186+565, 1030:1030+194]
pil_image = Image.fromarray(cropped_image)
pil_image.save("cropped_2d_img.png")
plt.imshow(cropped_image)
plt.show()

print("잘라낸 이미지의 shape:", cropped_image.shape)