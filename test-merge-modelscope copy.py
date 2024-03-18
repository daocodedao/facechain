
# https://www.modelscope.cn/models/iic/cv_unet_face_fusion_torch/summary

import cv2
# from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
import platform

image_face_fusion = pipeline('face_fusion_torch',
                            model='damo/cv_unet_face_fusion_torch', model_revision='v1.0.3')

if platform.system() == "linux":
    template_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/facefusion_template.jpg'
    user_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/facefusion_user.jpg'
else:
    template_path = './images/liudehua.jpg'
    user_path = './images/zhoujielun.jpg'

print(template_path)
result = image_face_fusion(dict(template=template_path, user=user_path))

cv2.imwrite('result.png', result['output_img'])
print('finished!')