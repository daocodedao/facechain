


from modelscope.pipelines import pipeline
from modelscope.utils.constant import  Tasks
from modelscope.outputs import OutputKeys
import platform


if platform.system() == "Linux":
    tmp_path="/data/work/facechain/worker_data/qw/training_data/ly261666/cv_portrait_model/a4_labeled/tmp.png"
    imagePath="/data/work/facechain/worker_data/qw/training_data/ly261666/cv_portrait_model/a4/000.jpg"
else:
    tmp_path="out/tmp.png"
    imagePath="out/000.jpg"

face_quality_func = pipeline(Tasks.face_quality_assessment, 'damo/cv_manual_face-quality-assessment_fqa', model_revision='v2.0')


face_quality_score = face_quality_func(imagePath)[OutputKeys.SCORES]

print(face_quality_score)
print("done")