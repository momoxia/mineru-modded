from rapidocr import RapidOCR
import numpy as np
from PIL import Image


ocr_engine = RapidOCR()

def img_ocr(image):
    """
    对图像进行OCR识别
    :param image: 可以是文件路径、PIL.Image对象或numpy数组(cv2图像)
    :return: 包含文本框坐标和对应文本的字典
    """
    # 如果是字符串，假定为文件路径
    if isinstance(image, str):
        ocr_result = ocr_engine(image)
    # 如果是PIL图像，转换为numpy数组
    elif isinstance(image, Image.Image):
        image_array = np.array(image)
        ocr_result = ocr_engine(image_array)
    # 如果是numpy数组(cv2图像)，直接使用
    elif isinstance(image, np.ndarray):
        ocr_result = ocr_engine(image)
    else:
        raise ValueError("Unsupported image type. Expected str, PIL.Image, or numpy.ndarray")
    
    boxes = ocr_result.boxes
    txts = ocr_result.txts
    info = {}
    match_list = []
    if boxes is not None and txts is not None:
        for box, txt in zip(boxes, txts):
            box_bbox = (float(box[0][0]),
                        float(box[0][1]),
                        float(box[2][0]),
                        float(box[2][1]))
            info[box_bbox] = txt
            match_list.append(box_bbox)
        match_list.sort(key=lambda item: (item[1], item[0]))

    return info, match_list

