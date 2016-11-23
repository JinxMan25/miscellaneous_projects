import numpy as np
import cv2
from matplotlib import import pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv2.imread("box.png", 0)
img2 = cv2.imread("box_in_scene.png", 0)


