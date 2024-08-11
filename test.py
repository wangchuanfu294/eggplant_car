import torch
from torchvision import models, transforms
import cv2
import numpy as np

# 1. 加载预训练的 DeepLabV3 模型
model = models.segmentation.deeplabv3_resnet101(weights='DeepLabV3_ResNet101_Weights.DEFAULT')
model.eval()  # 切换到评估模式

# 2. 定义图像预处理和后处理的变换
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(520),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. 读取图像
image = cv2.imread('resource/2.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 4. 图像预处理
input_tensor = preprocess(image_rgb)
input_batch = input_tensor.unsqueeze(0)  # 增加批次维度

# 5. 进行前向推理
with torch.no_grad():
    output = model(input_batch)['out'][0]

# 6. 获取预测结果并进行后处理
output_predictions = output.argmax(0)  # 获取每个像素的最大类别
output_predictions = output_predictions.byte().cpu().numpy()

# 7. 生成分割颜色图
# 需要将颜色映射的大小调整为模型可能输出的所有类别数
num_classes = output_predictions.max() + 1
color_map = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)  # 随机生成颜色映射

# 应用颜色映射
segmentation_map = color_map[output_predictions]

# 显示结果
cv2.imshow('Segmentation Map', segmentation_map)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果
cv2.imwrite('segmentation_map.jpg', segmentation_map)
