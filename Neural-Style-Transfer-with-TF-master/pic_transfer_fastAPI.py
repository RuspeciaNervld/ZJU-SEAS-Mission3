from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
from io import BytesIO

app = FastAPI()

def style_transfer(input_image_np, style_model_path):
    # 加载风格迁移模型
    net = cv2.dnn.readNetFromTorch(style_model_path)
    
    # 调整图像大小以匹配模型的输入要求
    # 这里假设 input_image_np 已经是预处理过的图像，因此不需要再次调整大小
    # 如果需要调整大小，可以使用 cv2.resize 或其他方法
    
    # 使用 dnn 模块进行风格迁移
    blob = cv2.dnn.blobFromImage(input_image_np, 1.0, (input_image_np.shape[1], input_image_np.shape[0]),
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
    
    # 将归一化后的图像转换为8位无符号整数
    output_n = output_n.astype(np.uint8)

    # 返回风格迁移后的图像
    return output_n

# 将base64编码的图像转换为NumPy数组
def base64_to_cv2(b64str):
    img_data = base64.b64decode(b64str)
    image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    return image

# 将NumPy数组转换为base64编码
def cv2_to_base64(image_np):
    ret, buf = cv2.imencode('.jpg', image_np)
    jpg_as_text = base64.b64encode(buf).decode('utf-8')
    return jpg_as_text

from pydantic import BaseModel
class StyleTransferRequest(BaseModel):
    style_index: int
    img_base64: str

global styleModelPaths
global modelsFolder
# FastAPI路由
@app.post("/style_transfer")
async def style_transfer_api(request_data: StyleTransferRequest):
    print(f"Received request for style index {request_data.style_index}")
    print(f"Received image with length {len(request_data.img_base64)}")
    styleModelPath = styleModelPaths[request_data.style_index]

    # 使用定义的函数进行风格迁移
    input_image_np = base64_to_cv2(request_data.img_base64)
    output_image_np = style_transfer(input_image_np, styleModelPath)

    # 将风格迁移后的图像保存在本地
    outputFileName = f"style_transfer_output_{request_data.style_index}.jpg"
    cv2.imwrite(outputFileName, output_image_np)
    print(f"Style transfer output saved to {outputFileName}")

    output_img_base64 = cv2_to_base64(output_image_np)

    return {"output_image_base64": output_img_base64}

# 运行Uvicorn服务器
if __name__ == "__main__":
    import uvicorn
    modelsFolder = "./models"
    styleModelPaths = ['./models/the_wave.t7', './models/la_muse.t7', './models/composition_vii.t7', './models/the_scream.t7', './models/starry_night.t7', './models/candy.t7', './models/udnie.t7', './models/feathers.t7', './models/mosaic.t7']
    uvicorn.run(app, host="0.0.0.0", port=8001)