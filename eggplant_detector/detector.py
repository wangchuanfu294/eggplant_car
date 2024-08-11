from ultralytics import YOLO
import numpy as np
import cv2

from semantic_segmentation import segmentate
#====全局参数====
resource_path = "resource/2.jpg"
video = False
#==============

def main():
  model = YOLO("eggplant_detector/best.pt")  # 加载YOLO模型

  frame = cv2.imread(resource_path)
  frame = cv2.resize(frame, (480, 640))
  
  segmentation_map = segmentate(frame)

  results = model.track(frame, agnostic_nms=True)

  for result in results:

    # 检查是否有检测到的框
    if result.boxes is not None and len(result.boxes) > 0 :

      x1, y1, x2, y2 = result.boxes.xyxy[0]
    
      # 假设你已经得到了 x1, y1, x2, y2 这四个坐标
      roi = frame[int(y1):int(y2), int(x1):int(x2)]
      segmentation_roi = segmentation_map[int(y1)-50:int(y2)+50, int(x1)-50:int(x2)+50]

      cv2.imshow("segmentation_roi", segmentation_roi)
      cv2.waitKey(1)
      
      # 转换为灰度图像
      gray_image = cv2.cvtColor(segmentation_roi, cv2.COLOR_BGR2GRAY)
      cv2.imshow('gray Image', gray_image)
      cv2.waitKey(1)

      # 二值化处理
      _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      # 执行开运算以去除噪声
      kernel = np.ones((5, 5), np.uint8)  # 选择合适的核大小
      binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

      # 计算白色像素和黑色像素的数量
      white_pixels = np.sum(binary_image == 255)
      black_pixels = np.sum(binary_image == 0)
      print(white_pixels,black_pixels)
      # 判断前景和背景，如果前景是黑色，则反转图像
      if black_pixels < white_pixels:
        binary_image = cv2.bitwise_not(binary_image)

      cv2.imshow('Binary Image', binary_image)
      cv2.waitKey(1)

      # 查找轮廓
      contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      # 复制原始图像，以便在其上绘制矩形
      output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

      # 初始化变量来存储最大面积的矩形
      max_area = 0
      best_rect = None
      best_box = None

      # 遍历所有轮廓
      for contour in contours:
        # 计算最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)  # 将角点坐标转为整数

        # 计算矩形面积
        width = rect[1][0]
        height = rect[1][1]
        area = width * height

        # 判断是否为最大面积的矩形
        if area > max_area:
          max_area = area
          best_rect = rect
          best_box = box

      # 仅绘制面积最大的矩形
      if best_box is not None:
        # 复制原始图像，以便在其上绘制矩形
        output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(output_image, [best_box], 0, (255,0,0), 2)  # 绘制为白色，线宽 2

        # 输出矩形的斜向角度
        angle = best_rect[2]
        print(f"最大矩形的斜向角度: {angle}°")

        # 显示结果
        cv2.imshow('Largest Rotated Rectangle', output_image)
        cv2.waitKey(1)

      cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 绿色框，线条宽度为2
      
  cv2.imshow('Camera Frame', frame)
  cv2.waitKey(5000)

if __name__ == "__main__":
  main()
