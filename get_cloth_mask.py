from pylab import imshow
import numpy as np
import cv2
import torch
import albumentations as albu
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from cloths_segmentation.pre_trained_models import create_model
import warnings
warnings.filterwarnings("ignore")

model = create_model("Unet_2020-10-30")
model.eval()
image = load_rgb("./static/cloth_web.jpg")

transform = albu.Compose([albu.Normalize(p=1)], p=1)

padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)

x = transform(image=padded_image)["image"]
x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

with torch.no_grad():
    prediction = model(x)[0][0]

mask = (prediction > 0).cpu().numpy().astype(np.uint8)
mask = unpad(mask, pads)

# Initialize the destination images
img = np.full((1024, 768, 3), 255, dtype=np.uint8)
seg_img = np.full((1024, 768), 0, dtype=np.uint8)

# Read the original image
b = cv2.imread("./static/cloth_web.jpg")
b_img = mask * 255

# Resize if needed
if b.shape[1] <= 600 and b.shape[0] <= 500:
    b = cv2.resize(b, (int(b.shape[1] * 1.2), int(b.shape[0] * 1.2)))
    b_img = cv2.resize(b_img, (int(b_img.shape[1] * 1.2), int(b_img.shape[0] * 1.2)))

# Compute the start and end indices for height and width to avoid rounding issues
height, width, _ = b.shape
start_h = int((1024 - height) // 2)
end_h = start_h + height
start_w = int((768 - width) // 2)
end_w = start_w + width

# Place `b` into `img`
img[start_h:end_h, start_w:end_w] = b

# Place `b_img` into `seg_img`
seg_img[start_h:end_h, start_w:end_w] = b_img

# Save the results
cv2.imwrite("./HR-VITON-main/test/test/cloth/00001_00.jpg", img)
cv2.imwrite("./HR-VITON-main/test/test/cloth-mask/00001_00.jpg", seg_img)
