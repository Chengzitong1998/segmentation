import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
# 加载预训练的 Mask R-CNN 模型
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def transform_image(image):
    image_tensor = F.to_tensor(image)
    return image_tensor.unsqueeze(0)

def perform_instance_segmentation(image_path):
    image = load_image(image_path)
    image_tensor = transform_image(image)

    with torch.no_grad():
        predictions = model(image_tensor)

    return image, predictions

def visualize_results(image, predictions, threshold=0.5):
    # 获取分割掩码和类别
    masks = predictions[0]['masks']
    scores = predictions[0]['scores']
    labels = predictions[0]['labels']

    # 创建一个空的图像用于显示结果
    output_image = image.copy()

    for i in range(len(masks)):
        if scores[i] > threshold:  # 根据阈值过滤
            mask = masks[i, 0].mul(255).byte().cpu().numpy()  # 获取掩码
            color = np.random.randint(0, 255, size=(3,)).tolist()  # 随机颜色

            # 将掩码应用到图像
            output_image[mask > 0] = output_image[mask > 0] * 0.5 + np.array(color) * 0.5

            # 绘制边界框
            bbox = predictions[0]['boxes'][i].cpu().numpy().astype(int)
            cv2.rectangle(output_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    plt.imshow(output_image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    image_path = '/root/autodl-tmp/segmentation/1.png'
    image, predictions = perform_instance_segmentation(image_path)
    visualize_results(image, predictions)
