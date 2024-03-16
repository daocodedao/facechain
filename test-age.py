


import cv2
import os
from modelscope.pipelines import pipeline
import platform
from modelscope.utils.constant import  Tasks
import numpy as np
from utils.logger_settings import api_logger


if platform.system() == "Linux":
    tmp_path="/data/work/facechain/worker_data/qw/training_data/ly261666/cv_portrait_model/a9_labeled/tmp.png"
    imagePath="/data/work/facechain/worker_data/qw/training_data/ly261666/cv_portrait_model/a9/000.jpg"
else:
    tmp_path="out/tmp.png"
    imagePath="out/000.jpg"

# # https://www.modelscope.cn/models/iic/cv_resnet34_face-attribute-recognition_fairface/summary
# api_logger.info("加载模型 人脸属性识别模型 fair_face_attribute_func")
# api_logger.info("推理：输入图片，如存在人脸则返回人的性别以及年龄区间。")
# fair_face_attribute_func = pipeline(Tasks.face_attribute_recognition,
#                                             'damo/cv_resnet34_face-attribute-recognition_fairface', model_revision='v2.0.2')

# api_logger.info("检测年龄和性别")
# attribute_result = fair_face_attribute_func(tmp_path)
# # if cnt == 0:
# score_gender = np.array(attribute_result['scores'][0])
# score_age = np.array(attribute_result['scores'][1])

# gender = np.argmax(score_gender)
# age = np.argmax(score_age)

# print(f"gender = {gender} age = {age}")


# #pip install thop

# model_id = 'damo/cv_resnet50_pedestrian-attribute-recognition_image'
# pedestrian_attribute_recognition = pipeline(Tasks.pedestrian_attribute_recognition, model=model_id)
# # output = pedestrian_attribute_recognition(tmp_path)
# output = pedestrian_attribute_recognition(tmp_path)


# print("done")

# # the output contains boxes and labels
# print(output)
# else:
#     score_gender += np.array(attribute_result['scores'][0])
#     score_age += np.array(attribute_result['scores'][1])




# from deepface import DeepFace
# import cv2
# import matplotlib.pyplot as plt

# img1 = cv2.imread(tmp_path)
# # result = DeepFace.analyze(img1, actions=['age', 'gender'])
# result = DeepFace.analyze(img_path=img1, actions=['gender', 'race'], detector_backend = "yolov8", enforce_detection=False)
# print(result)

os.environ['HTTP_PROXY'] = '192.168.0.77:18808'
os.environ['HTTPS_PROXY'] = '192.168.0.77:18808'


from facelib import FaceDetector,AgeGenderEstimator

detector = FaceDetector()
faces, boxes, scores, landmarks = detector.detect_align(cv2.imread(tmp_path))

age_gender_detector = AgeGenderEstimator()
genders, ages = age_gender_detector.detect(faces)

print(genders)
print(ages)
print("done")

