


import cv2
import os
from modelscope.pipelines import pipeline

face_detection = 'face-detection'

# tmp_path="/data/work/facechain/worker_data/qw/training_data/ly261666/cv_portrait_model/a4_labeled/tmp.png"
# imagePath="/data/work/facechain/worker_data/qw/training_data/ly261666/cv_portrait_model/a4/000.jpg"

tmp_path="out/tmp.png"
imagePath="out/000.jpg"

face_detection = pipeline(task=face_detection, model='damo/cv_ddsar_face-detection_iclr23-damofd', model_revision='v1.1')

# img_path = os.path.join(imdir, imname)
im = cv2.imread(imagePath)
h, w, _ = im.shape
max_size = max(w, h)
ratio = 1024 / max_size
new_w = round(w * ratio)
new_h = round(h * ratio)

imt = cv2.resize(im, (new_w, new_h))
print(f"图片保存到{tmp_path}")

cv2.imwrite(tmp_path, imt)
result_det = face_detection(tmp_path)
bboxes = result_det['boxes']
print(f"检测人脸数量{len(bboxes)}")