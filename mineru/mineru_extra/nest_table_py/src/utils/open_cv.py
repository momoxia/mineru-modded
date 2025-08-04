import cv2
import numpy as np
import os
from typing import Dict

class Kernals:
    h_kernel = [15, 1]
    v_kernel = [1, 15]
    d_kernal = [2, 2]
    c_kernal = 2

class Limits:
    size = [50, 50]

class CvLoad:
    def __init__(self):
        self.H: Dict
        self.V: Dict
    # 读取图片所有线信息
    def __call__(self, img) -> dict:
        if img is None:
            raise FileNotFoundError(f"img not found")
        
        lines = {}

        K = Kernals # 核参数
        L = Limits  # 限制参数

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, K.h_kernel)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, K.v_kernel)
        d_kernel = np.ones(K.d_kernal, np.uint8)

        horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
        vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)

        horizontal_dilated = cv2.dilate(horizontal, d_kernel, iterations=1)
        vertical_dilated = cv2.dilate(vertical, d_kernel, iterations=1)

        contours_h, _ = cv2.findContours(
            horizontal_dilated, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        # x: 起始点x y: 起始点y w: 直线宽度 h: 直线长度
        contours_h = sorted(contours_h, key=lambda c: cv2.boundingRect(c)[1])
        lines["H"] = {}
        for i, cnt in enumerate(contours_h):
            x, y, w, h = cv2.boundingRect(cnt)
            lines["H"][i] = {"p": (x, y, w, h), "r": set()}
    
        contours_v, _ = cv2.findContours(
            vertical_dilated, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        contours_v = sorted(contours_v, key=lambda c: cv2.boundingRect(c)[0])
        lines["V"] = {}
        for i, cnt in enumerate(contours_v):
            x, y, w, h = cv2.boundingRect(cnt)
            lines["V"][i] = {"p": (x, y, w, h), "r": set()}

        self.H = lines["H"]
        self.V = lines["V"]