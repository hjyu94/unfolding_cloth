import numpy as np
import cv2
import torch
import albumentations as albu
import os
from PIL import Image

from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image

from cloths_segmentation.cloths_segmentation.pre_trained_models import create_model
import time

def segmentation(model, input_image_path, output_image_path):
    image = load_rgb(input_image_path)

    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)

    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

    with torch.no_grad():
        prediction = model(x)[0][0]

    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)
    # np.save("datasets/temp/mask.npy", mask)
    # imshow(mask)
    # rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    #
    # for i in range(mask.shape[0]):
    #     for j in range(mask.shape[1]):
    #         if mask[i, j] == 1:
    #             rgb_image[i, j] = [255, 255, 0]
    #         else:
    #             rgb_image[i, j] = [128, 0, 128]

    # rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # rgb_image[mask == 1] = [255, 255, 0]
    # rgb_image[mask != 1] = [128, 0, 128]

    # rgb_image = np.full((mask.shape[0], mask.shape[1], 3), [128, 0, 128], dtype=np.uint8)
    # rgb_image[mask == 1] = [255, 255, 0]
    #
    # pil_image = Image.fromarray(rgb_image)
    # pil_image.save(output_image_path)



if __name__ == '__main__':
    start_time = time.time()
    model = create_model("Unet_2020-10-30")
    model.eval()

    root_dir = "/home/hjeong/code/unfolding_cloth/datasets/temp/"
    input_img_path = root_dir + "image_left.png"
    output_img_path = root_dir + "segmentation_output.png"
    segmentation(model, input_img_path, output_img_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("변환 행렬 생성에 걸린 시간:", elapsed_time, "초")