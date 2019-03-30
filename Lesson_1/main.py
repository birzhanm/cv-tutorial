"""
This file aims to clarify how to
1. Load an image using pillow library
2. Obtain a matrix representation of an image
3. Perform basic image processing operations such as cropping, rotating and so on
"""


"""
Part 1: Load an image using pillow library
"""

# link to a sample image in jpg format
sample_image_url = "https://sample-videos.com/img/Sample-jpg-image-100kb.jpg"

# download the image using wget setting the name of the file as "sample_image.jpg"
import wget
import os
# make sure that the image is donwloaded only once
if not os.path.exists("sample_image.jpg"):
    wget.download(sample_image_url, "sample_image.jpg")

import PIL
from PIL import Image
# load "sample_image.jpg" using Image module
img = Image.open("sample_image.jpg")
# let us show the image
img.show()
# let us determine the size of the image
size = img.size
print(size)

"""
Obtain a matrix representation of an image
"""
import numpy as np
img_rgb = img.convert('RGB')
img_np = np.asarray(img_rgb)
print(img_np)
# img_np can be viewed as a three-dimensional array width*height*channels

"""
Perform basic image processing operations such as cropping, rotating and so on
"""
# let us crop 400x400 image from bottom right of the original image
border = (288,288,688,688) #left, up, right, bottom
img_cropped = img.crop(border)
img_cropped.show()

# let us resize the cropped image into 224x224 size
new_size = (224, 224)
img_resized = img_cropped.resize(new_size, PIL.Image.BICUBIC)
print(img_resized.size)

# let us rotate the resized images
angle = 45
img_rotated = img_resized.rotate(angle)
img_rotated.show()
# let us save the rotated image
# we need to make sure that we save only once
if not os.path.exists("rotated_image.jpg"):
    img_rotated.save("rotated_image.jpg")
