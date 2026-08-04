import cv2
import numpy as np

depth=cv2.imread(
"/home/tenda/HumanEgodata/data/serve_bread/realsense/rs_serve_bread_000/preprocess/all_data/00150/depth.png",
cv2.IMREAD_UNCHANGED
)

mask=cv2.imread(
    "/home/tenda/HumanEgodata/data/serve_bread/realsense/rs_serve_bread_000/preprocess/all_data/00150/mask_obj1.png",
    cv2.IMREAD_GRAYSCALE
)

obj_depth=depth[mask>0]

print("mask pixels:",len(obj_depth))
print("valid depth:",np.count_nonzero(obj_depth))
print("valid ratio:",
      np.count_nonzero(obj_depth)/len(obj_depth))

print("median depth:",
      np.median(obj_depth[obj_depth>0]))
