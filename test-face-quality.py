




import cv2
import os
from modelscope.pipelines import pipeline
import platform
from modelscope.utils.constant import  Tasks
import numpy as np
from utils.logger_settings import api_logger


def get_mask_head(result):
    masks = result['masks']
    scores = result['scores']
    labels = result['labels']
    mask_hair = np.zeros((512, 512))
    mask_face = np.zeros((512, 512))
    mask_human = np.zeros((512, 512))
    for i in range(len(labels)):
        if scores[i] > 0.8:
            if labels[i] == 'Face':
                if np.sum(masks[i]) > np.sum(mask_face):
                    mask_face = masks[i]
            elif labels[i] == 'Human':
                if np.sum(masks[i]) > np.sum(mask_human):
                    mask_human = masks[i]
            elif labels[i] == 'Hair':
                if np.sum(masks[i]) > np.sum(mask_hair):
                    mask_hair = masks[i]
    mask_head = np.clip(mask_hair + mask_face, 0, 1)
    ksize = max(int(np.sqrt(np.sum(mask_face)) / 20), 1)
    kernel = np.ones((ksize, ksize))
    mask_head = cv2.dilate(mask_head, kernel, iterations=1) * mask_human
    _, mask_head = cv2.threshold((mask_head * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mask_head, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
    max_idx = np.argmax(area)
    mask_head = np.zeros((512, 512)).astype(np.uint8)
    cv2.fillPoly(mask_head, [contours[max_idx]], 255)
    mask_head = mask_head.astype(np.float32) / 255
    mask_head = np.clip(mask_head + mask_face, 0, 1)
    mask_head = np.expand_dims(mask_head, 2)
    return mask_head

if platform.system() == "Linux":
    tmp_path="/data/work/facechain/worker_data/qw/training_data/ly261666/cv_portrait_model/a4_labeled/tmp.png"
    imagePath="/data/work/facechain/worker_data/qw/training_data/ly261666/cv_portrait_model/a4/000.jpg"
else:
    tmp_path="out/tmp.png"
    imagePath="out/000.jpg"


api_logger.info("加载模型 segmentation_pipeline")
segmentation_pipeline = pipeline(Tasks.image_segmentation,
                                 'damo/cv_resnet101_image-multiple-human-parsing',
                                   model_revision='v1.0.1')

api_logger.info("加载模型 facial_landmark_confidence_func")
facial_landmark_confidence_func = pipeline(Tasks.face_2d_keypoints,
                                                'damo/cv_manual_facial-landmark-confidence_flcm', model_revision='v2.5')


api_logger.info(f"segmentation_pipeline")
result = segmentation_pipeline(tmp_path)
api_logger.info(f"get_mask_head")
mask_head = get_mask_head(result)
api_logger.info(f"重新读取{tmp_path}")
im = cv2.imread(tmp_path)
im = im * mask_head + 255 * (1 - mask_head)


raw_result = facial_landmark_confidence_func(im)
api_logger.info(f"结束facial_landmark_confidence_func {raw_result}")