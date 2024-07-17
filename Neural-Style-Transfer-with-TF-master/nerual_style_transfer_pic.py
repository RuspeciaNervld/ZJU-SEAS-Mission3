import argparse
import cv2
import numpy as np
import os

# 解析命令行参数，这里只需要模型文件夹和风格模型的路径
ap = argparse.ArgumentParser()
ap.add_argument("--modelsFolder", default="models",
                help="path to folder of neural style transfer models")
ap.add_argument("--styleModel", type=str, required=True,
                help="path to the style transfer model")
ap.add_argument("--inputImage", type=str, required=True,
                help="path to the input image")

modelsFolder =  "./models"
styleModelPaths = ['./models/the_wave.t7', './models/la_muse.t7', './models/composition_vii.t7', './models/the_scream.t7', './models/starry_night.t7', './models/candy.t7', './models/udnie.t7', './models/feathers.t7', './models/mosaic.t7']
styleModelNames = ['海滩海浪', '拉姆兹', '第七组合', '尖叫', '星空', '糖果', '乌德尼', '羽毛', '拼花']
inputImagePath = "./images/boat.jpg"

for i in range(len(styleModelPaths)):
    styleModelPath = styleModelPaths[i]

    # 加载风格迁移模型
    net = cv2.dnn.readNetFromTorch(styleModelPath)

    # 加载输入图像
    inputImage = cv2.imread(inputImagePath)
    (h, w) = inputImage.shape[:2]

    # 调整图像大小以匹配模型的输入要求
    blob = cv2.dnn.blobFromImage(inputImage, 1.0, (w, h),
                                (103.939, 116.779, 123.680), swapRB=False, crop=False)
    net.setInput(blob)
    output = net.forward()

    # 处理输出结果
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    output /= 255.0
    output = output.transpose(1, 2, 0)

    # 应用归一化
    output_n = cv2.normalize(src=output, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 显示结果
    # cv2.imshow('Style Transfer', output_n.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果
    outputFileName = f"style_transfer_output_{i}.jpg"
    cv2.imwrite(outputFileName, output_n.astype(np.uint8))
    print(f"Style transfer output saved to {outputFileName}")