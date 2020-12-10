from PIL import Image
import numpy as np


img = Image.open("uv_weight_mask.png").convert("L")
arr = np.asarray(img)

start = 125
end = 160

mask1 = (arr == 64)
mask2 = np.zeros(arr.shape, dtype=bool)
mask2[start:end,:] = True
mask = (mask1 == True) & (mask2 == True)

newarr = arr.copy()
newarr[mask] = 150


newimg = Image.fromarray(np.uint8(newarr)).convert("L")
newimg.save("new_uv_weight.jpg")