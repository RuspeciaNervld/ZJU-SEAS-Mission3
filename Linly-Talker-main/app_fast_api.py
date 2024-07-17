# gradio_api.py
import base64
from fastapi import FastAPI, Request, HTTPException
import uvicorn
import threading
from app import Talker_response,init  # 确保从正确的位置导入 main 函数
from moviepy.editor import VideoFileClip, AudioFileClip

app = FastAPI()

@app.post("/gradio_api")
async def gradio_api(request: Request):
    # 这里实现 API 逻辑，调用 Gradio 应用并返回结果
    data = await request.json()
    text = data.get("text", "")+"。以上是我的提问，请你尽可能一下子完整解答我的疑惑。"  # 假设我们期望从请求中获取文本输入

    # 使用 Gradio 的 queue 方法处理请求
    # 注意：这里假设您的 Gradio 应用使用 queue() 方法启动
    voice = "zh-CN-XiaoxiaoNeural"
    rate = 0
    volume = 100
    pitch = 0
    batch_size = 2
    response = Talker_response(text, voice, rate, volume, pitch, batch_size)

    if response is None:
        raise HTTPException(status_code=503, detail="Gradio 应用未响应")

    # 将保存在response[0]路径中的视频文件转换为 base64 编码的字符串
    # 获取视频
    video = VideoFileClip(response[0])
    audio = AudioFileClip(response[1].replace('vtt','wav'))
    # 合并视频和音频为
    video = video.set_audio(audio)
    # 保存合并后的视频
    video.write_videofile("output.mp4")

    # 将视频转换为 base64 编码的字符串
    with open("output.mp4", "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode("utf-8")
    # 返回响应
    response_body = {"video": video_base64}

    return response_body

if __name__ == "__main__":
    init()
    uvicorn.run(app, host="0.0.0.0", port=8000)