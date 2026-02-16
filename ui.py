import gradio as gr
import subprocess
import os
import time
import base64

# === 配置 ===
# C++ 编译好的可执行文件路径
EXE_PATH = "./build/v5lite_trt"  # 请根据实际编译输出路径修改
CONFIG_PATH = "./config.yaml"   # 配置文件路径
MODE_FLAG = "webui"              # 触发 C++ 进入循环模式的标志

def encode_image(image_path):
    if not os.path.exists(image_path):
        print(f"⚠️ 警告: 图片未找到 - {image_path}")
        return ""
    with open(image_path, "rb") as f:
        # 读取并编码
        encoded_string = base64.b64encode(f.read()).decode("utf-8")
        # 根据后缀判断类型 (jpg/png)
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = "image/png" if "png" in ext else "image/jpeg"
        return f"data:{mime_type};base64,{encoded_string}"

def get_bg_css(image_path):
    """读取图片并生成通过 Base64 嵌入的 CSS"""
    if not os.path.exists(image_path):
        print(f"⚠️ 背景图片未找到: {image_path}，将不显示背景。")
        return ""
    
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower()
        mime = "image/png" if ".png" in ext else "image/jpeg"
        
        # === 核心修改逻辑 ===
        # 使用 linear-gradient(color, color) 创建一个纯色层
        # rgba(255, 255, 255, 0.5) 代表：红色255, 绿色255, 蓝色255 (纯白), 透明度0.5 (50%)
        return f"""
        .gradio-container {{
            /* 语法：background-image: 顶层遮罩, 底层图片 */
            background-image: linear-gradient(rgba(255, 255, 255, 0.6), rgba(255, 255, 255, 0.6)), url('data:{mime};base64,{data}') !important;
            
            background-size: cover !important;        /* 铺满 */
            background-repeat: no-repeat !important;  /* 不重复 */
            background-position: center !important;   /* 居中 */
            background-attachment: fixed !important;  /* 固定不动 */
        }}
        """

class CPPInferenceService:
    def __init__(self):
        self.process = None
        self.start_service()

    def start_service(self):
        """启动 C++ 子进程"""
        print("正在启动 C++ 推理后端...")
        try:
            # 相当于执行: ./v5lite_trt ../config.yaml webui
            self.process = subprocess.Popen(
                [EXE_PATH, CONFIG_PATH, MODE_FLAG],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,  # 以文本模式通信
                bufsize=1   # 行缓冲
            )
            
            # 等待 C++ 输出 "READY"
            while True:
                line = self.process.stdout.readline()
                if "READY" in line:
                    print("C++ 后端已就绪！")
                    break
                if line == "" and self.process.poll() is not None:
                    print("C++ 后端启动失败，请检查路径或日志。")
                    print(self.process.stderr.read())
                    break
        except Exception as e:
            print(f"启动失败: {e}")

    def infer(self, image_path):
        """发送图片路径给 C++ 并获取结果"""
        if self.process is None or self.process.poll() is not None:
            print("后端未运行，尝试重启...")
            self.start_service()
            if self.process is None:
                return None, "", 0, 0, 0

        # 1. 发送路径 (加上换行符)
        try:
            self.process.stdin.write(os.path.abspath(image_path) + "\n")
            self.process.stdin.flush()
            
            # 2. 读取详细输出信息
            output_lines = []
            result_path = ""
            prep_time = 0.0
            inf_time = 0.0
            post_time = 0.0
            
            # 读取所有输出行，直到获取结果路径
            while True:
                line = self.process.stdout.readline()
                if not line:
                    break
                
                line = line.strip()
                output_lines.append(line)
                
                # 提取推理时间信息
                if "prepare image take:" in line:
                    prep_time = float(line.split(":")[1].strip().split(" ")[0])
                elif "Inference take:" in line:
                    inf_time = float(line.split(":")[1].strip().split(" ")[0])
                elif "Post process take:" in line:
                    post_time = float(line.split(":")[1].strip().split(" ")[0])
                # 检查是否为结果路径
                elif os.path.exists(line):
                    result_path = line
                    break
            
            if "ERROR" in result_path or not os.path.exists(result_path):
                print(f"推理错误: {result_path}")
                return None, "\n".join(output_lines), prep_time, inf_time, post_time
                
            return result_path, "\n".join(output_lines), prep_time, inf_time, post_time
        except Exception as e:
            print(f"通信错误: {e}")
            return None, str(e), 0, 0, 0

    def close(self):
        if self.process:
            self.process.terminate()

# 初始化服务
service = CPPInferenceService()

def run_inference(file):
    if file is None:
        return None, None, "", 0, 0, 0
    
    # # Gradio 传入的是 numpy array，我们需要先存为临时文件供 C++ 读取
    # temp_input = "temp_query.jpg"
    # 注意：使用 OpenCV 保存，确保格式正确
    import cv2
    # image 是 RGB (Gradio 默认)，OpenCV 需要 BGR
    # cv2.imwrite(temp_input, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    file_path = file.name
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # 复制到临时文件
    temp_input = f"temp_query{file_ext}"
    import shutil
    shutil.copy(file_path, temp_input)
    
    # 调用 C++
    output_path, output_info, prep_time, inf_time, post_time = service.infer(temp_input)
    
    if output_path:
        # 读取结果并转回 RGB 供 Gradio 显示
        #res_img = cv2.imread(output_path)
        total_time = prep_time + inf_time + post_time
        
        # 构建详细信息
        details = f"预处理时间: {prep_time:.2f} ms\n"
        details += f"推理时间: {inf_time:.2f} ms\n"
        details += f"后处理时间: {post_time:.2f} ms\n"
        details += f"总时间: {total_time:.2f} ms\n\n"
        details += f"C++ 输出信息:\n{output_info}"
        
        # 计算 FPS
        fps = 1000 / total_time if total_time > 0 else 0
        
        
        # 根据文件类型返回不同结果
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            # 读取结果并转回 RGB 供 Gradio 显示
            res_img = cv2.imread(output_path)
            return cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), None, details, prep_time, inf_time, fps
        elif file_ext in ['.mp4', '.avi', '.mkv', '.mov']:
            # 对于视频，返回视频路径
            return None, output_path, details, prep_time, inf_time, fps
        else:
            return None, None, f"不支持的文件类型: {file_ext}", 0, 0, 0
    else:
        return None, None, f"推理失败\n\n{output_info}", 0, 0, 0

# 定义界面
# 生成 CSS 字符串
my_css = get_bg_css("/media/F/hbf/YOLOv5-Lite-master/cpp_demo/tensorrt/tree/background.png")
with gr.Blocks(css=my_css, title="C++ Backend Inference") as demo:
    # 添加CSS样式，设置背景图
    

    logo_path = "/media/F/hbf/YOLOv5-Lite-master/cpp_demo/tensorrt/samples/xidian.jpg"
    logo_src = encode_image(logo_path)

    # 添加logo
    with gr.Row():
        # === 替换开始 ===
        
        gr.HTML(f"""
        <div style="display: flex; align-items: center; gap: 30px; padding: 10px 0;">
            <!-- 左侧 Logo -->
            <div style="width: 100px; height: 100px; flex-shrink: 0; display: flex; align-items: center; justify-content: center;">
                <img src="{logo_src}" style="width: 100%; height: 100%; object-fit: contain; display: block;">
            </div>

            <!-- 右侧文字 -->
            <div style="display: flex; flex-direction: column; justify-content: center;">
            <h1 style="
                margin: 0; 
                font-size: 24px; 
                line-height: 1.5; 
                font-weight: bold;
                /* === 新增样式开始 === */
                background-color: #00008B;  /* 橙色背景 (DarkOrange) */
                color: yellow;               /* 白色文字 */
                padding: 15px 25px;         /* 内边距：上下15px，左右25px */
                border-radius: 12px;        /* 圆角边框 */
                box-shadow: 0 4px 6px rgba(0,0,0,0.1); /* 轻微阴影，增加立体感 */
                display: inline-block;      /* 让背景框根据文字内容自适应宽度 */
                /* === 新增样式结束 === */
            ">
                🚀 基于边缘端推理优化的农林病虫害无人机实时检测系统
            </h1>
        </div>
        </div>
        """)
        # === 替换结束 ===

    
    with gr.Row():
        inp = gr.File(label="上传图片或视频", file_types=["image", "video"], height=500)
        with gr.Column():
            img_out = gr.Image(label="推理结果", height=240)
            vid_out = gr.Video(label="推理结果", height=240)
    btn = gr.Button("开始推理", variant="primary")
    with gr.Row():
        details = gr.Textbox(label="推理详细信息", lines=10, interactive=False)
        
    with gr.Row():
        prep_time = gr.Number(label="预处理时间 (ms)", interactive=False)
        inf_time = gr.Number(label="推理时间 (ms)", interactive=False)
        fps = gr.Number(label="FPS", interactive=False)
        
    
    btn.click(run_inference, inputs=inp, outputs=[img_out, vid_out, details, prep_time, inf_time, fps])

if __name__ == "__main__":
    try:
        demo.launch(server_name="0.0.0.0", server_port=7860)
    finally:
        service.close()
