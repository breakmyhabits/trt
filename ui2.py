import gradio as gr
import yaml
import os
import subprocess
import shutil
import time
import base64

# --- 全局配置与路径 ---

# 假设 export.py 在当前目录下 (YOLOv5 导出脚本)
EXPORT_SCRIPT = "export.py" 
# 假设 trtexec 已在系统环境变量中，否则请写绝对路径 (例如: /usr/src/tensorrt/bin/trtexec)
TRTEXEC_CMD = "/usr/src/tensorrt/bin/trtexec"  

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
    tab_css = """
    /* 针对所有具有 Tab 角色的按钮外框 */
    button[role="tab"] {
        background-color: #ADD8E6 !important; /* 浅蓝色背景 (LightBlue) */
        border: 1px solid #87CEFA !important; /* 边框颜色 */
        
        /* === 尺寸与形状调整 === */
        border-radius: 16px 16px 0 0 !important; /* 增大顶部圆角，使其更圆润 */
        margin-right: 8px !important;            /* 增大标签之间的间距，避免拥挤 */
        padding: 16px 40px !important;           /* 【关键】大幅增加内边距：上下16px，左右40px，撑大方框 */
        min-height: 120px !important;             /* 【关键】设置最小高度，确保方框足够高 */
        
        opacity: 1 !important;                
        display: flex !important;                /* 启用 flex 布局 */
        align-items: center !important;          /* 确保内部文字垂直居中 */
        justify-content: center !important;      /* 确保内部文字水平居中 */
    }
    
    /* 强制穿透修改按钮内的文字颜色和粗细 (覆盖 Gradio 内部的 span 样式) */
    button[role="tab"], button[role="tab"] * {
        color: black !important;              /* 强制纯黑字体 */
        font-weight: 500 !important;          /* 强制最粗体 */
        font-size: 20px !important;           /* 【关键】将字体调大到 20px，与大方框更匹配 */
        letter-spacing: 1.4px !important;       /* 增加一点字间距，显得更大气 */
    }
    
    /* 选中状态下的 Tab 按钮样式 (使用 aria-selected 属性更精准) */
    button[role="tab"][aria-selected="true"],
    button[role="tab"].selected {
        background-color: #87CEEB !important; /* 天蓝色背景 (SkyBlue)，比未选中深一点 */
        border-bottom: none !important;       /* 去除底边框 */
        box-shadow: none !important;          /* 去除阴影 */
    }
    
    /* 隐藏 Gradio 默认的那条橙色/蓝色的选中下划线 */
    .tab-nav::before, .tab-nav::after, 
    button[role="tab"]::before, button[role="tab"]::after {
        display: none !important;
        background: transparent !important;
    }
    
    /* 确保标签栏的整体容器高度不受限制 */
    .tab-nav {
        min-height: 150px !important;
        border-bottom: 2px solid #87CEEB !important; /* 在标签栏底部加一条统一颜色的线，增强整体感 */
    }
    """
    if not os.path.exists(image_path):
        print(f"⚠️ 背景图片未找到: {image_path}，将不显示背景。")
        return tab_css
    
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower()
        mime = "image/png" if ".png" in ext else "image/jpeg"
        
        # === 核心修改逻辑 ===
        # 使用 linear-gradient(color, color) 创建一个纯色层
        # rgba(255, 255, 255, 0.5) 代表：红色255, 绿色255, 蓝色255 (纯白), 透明度0.5 (50%)
        bag_css = f"""
        .gradio-container {{
            /* 语法：background-image: 顶层遮罩, 底层图片 */
            background-image: linear-gradient(rgba(255, 255, 255, 0.6), rgba(255, 255, 255, 0.6)), url('data:{mime};base64,{data}') !important;
            
            background-size: cover !important;        /* 铺满 */
            background-repeat: no-repeat !important;  /* 不重复 */
            background-position: center !important;   /* 居中 */
            background-attachment: fixed !important;  /* 固定不动 */
        }}
        
        """
        return bag_css + tab_css

# =============================================================================
# 1. [...](asc_slot://start-slot-7)辅助功能函数 (配置管理 & 模型转换)
# =============================================================================
def load_config():
    """读取 config.yaml 文件内容"""
    if not os.path.exists(CONFIG_PATH):
        return "# Config file not found."
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return f.read()

def save_config(new_content):
    """保存内容到 config.yaml"""
    try:
        # [...](asc_slot://start-slot-13)校验 YAML 格式是否合法
        yaml.safe_load(new_content)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(new_content)
        return "配置已成功保存 (Configuration saved successfully)."
    except yaml.YAMLError as e:
        return f"YAML 格式错误 (Invalid YAML format): {e}"
    except Exception as e:
        return f"保存失败 (Error saving config): {e}"
def upload_config_file(file):
    """上传文件覆盖 config.yaml"""
    if file is None:
        return load_config(), "未上传文件"
    try:
        shutil.copy(file.name, CONFIG_PATH)
        return load_config(), "配置已通过文件覆盖更新。"
    except Exception as e:
        return load_config(), f"文件覆盖失败: {e}"
def convert_model_pipeline(pt_file, input_size=640, batch_size=1, precision="fp16"):
    """
    一键转换流水线: PyTorch (.pt) -> ONNX -> TensorRT (.engine)
    后端自动调用 export.py 和 trtexec
    """
    if pt_file is None:
        return "请先上传 .pt 模型文件。"

    logs = []
    
    # [...](asc_slot://start-slot-19)获取文件路径
    pt_path = pt_file.name
    # [...](asc_slot://start-slot-21)提取文件名 (不带后缀)
    base_name = os.path.splitext(os.path.basename(pt_path))[0]
    # [...](asc_slot://start-slot-23)定义输出路径 (默认保存到当前运行目录，方便后续调用)
    output_dir = os.getcwd()
    onnx_path = os.path.join(output_dir, f"{base_name}.onnx")
    engine_path = os.path.join(output_dir, f"{base_name}.engine")

    logs.append(f"====== 开始转换流程: {base_name} ======")
    
    # [...](asc_slot://start-slot-27)--- 步骤 1: PyTorch -> ONNX ---
    logs.append(f"[Step 1] 正在导出 ONNX: {pt_path} -> {onnx_path} ...")
    
    # [...](asc_slot://start-slot-29)构造 export.py 命令 (参考 YOLOv5 标准导出参数)
    # [...](asc_slot://start-slot-31)注意：你需要确保目录下有 export.py，或者修改此处路径指向 YOLOv5 的 export.py
    cmd_export = [
        "python", EXPORT_SCRIPT,
        "--weights", pt_path,
        "--img-size", str(input_size),
        "--batch-size", str(batch_size),
        "--device", "0",
    ]
    
    try:
        # 执行导出命令
        process = subprocess.run(cmd_export, capture_output=True, text=True, check=True)
        logs.append("[Export Log]:\n" + process.stdout)
    except subprocess.CalledProcessError as e:
        return "\n".join(logs) + f"\n[Error] ONNX 导出失败:\n{e.stderr}"
    except FileNotFoundError:
        return "\n".join(logs) + f"\n[Error] 找不到 {EXPORT_SCRIPT}，请确保该脚本在根目录下。"

    # [...](asc_slot://start-slot-33)检查 ONNX 是否生成 (通常 export.py 会在 pt 文件同级生成，或我们需要将其移动)
    # [...](asc_slot://start-slot-35)这里做一个简单的查找逻辑
    generated_onnx_temp = pt_path.replace(".pt", ".onnx")
    if os.path.exists(generated_onnx_temp):
        # [...](asc_slot://start-slot-37)如果生成在临时目录，移动到当前工作目录
        if generated_onnx_temp != onnx_path:
            shutil.move(generated_onnx_temp, onnx_path)
    
    if not os.path.exists(onnx_path):
         # [...](asc_slot://start-slot-39)尝试直接在当前目录找
         if not os.path.exists(f"{base_name}.onnx"):
            return "\n".join(logs) + "\n[Error] 未检测到生成的 ONNX 文件。"
         else:
            onnx_path = os.path.join(output_dir, f"{base_name}.onnx")

    logs.append(f"[Success] ONNX 文件已就绪: {onnx_path}")

    # [...](asc_slot://start-slot-45)--- 步骤 2: ONNX -> TensorRT ---
    logs.append(f"[Step 2] 正在构建 TensorRT 引擎 (使用 trtexec) ...")
    
    # [...](asc_slot://start-slot-47)构造 trtexec 命令
    cmd_trtexec = [
        TRTEXEC_CMD,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        
        "--verbose"
    ]
    if precision == "fp16":
        cmd_trtexec.append("--fp16") 
    elif precision == "int8":
        cmd_trtexec.append("--int8")

    try:
        process = subprocess.run(cmd_trtexec, capture_output=True, text=True, check=True)
        # trtexec 输出通常很长，这里只截取最后一部分或显示成功信息
        logs.append("[trtexec Log]: (Output truncated for brevity...)\n" + process.stdout[-1000:]) 
        logs.append(f"\n====== 转换成功! ======")
        logs.append(f"Engine saved to: {engine_path}")
    except subprocess.CalledProcessError as e:
        logs.append(f"\n[Error] trtexec 转换失败:\n{e.stderr}")
        return "\n".join(logs)
    except FileNotFoundError:
        return "\n".join(logs) + f"\n[Error] 找不到命令 '{TRTEXEC_CMD}'。请确保 TensorRT 已安装并添加到 PATH 环境变量。"

    return "\n".join(logs)

# =============================================================================
# 2. [...](asc_slot://start-slot-51)现有功能包装 (CPPInference)
# =============================================================================
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

# =============================================================================
# 3. 前端页面布局 (Gradio)
# =============================================================================

# [...](asc_slot://start-slot-65)自定义 CSS 样式
my_css = get_bg_css("/media/F/hbf/YOLOv5-Lite-master/cpp_demo/tensorrt/tree/background.png")
with gr.Blocks(css=my_css, title="TensorRT Inference Platform") as demo:
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
    
    with gr.Tabs():
        
        # [...](asc_slot://start-slot-67)--- 选项卡 1: 模型加载 (新增) ---
        with gr.Tab("模型加载"):
            gr.Markdown("### PyTorch (.pt) 模型上传与转换")
            gr.Markdown("自动流程: `PyTorch` -> `ONNX` -> `TensorRT Engine`")
            
            with gr.Row():
                with gr.Column(scale=1):
                    pt_file = gr.File(label="上传 .pt 模型文件", file_types=[".pt"])
                    with gr.Row():
                        input_size_num = gr.Number(value=640, label="Input Size (imgsz)", precision=0)
                        batch_size_num = gr.Number(value=1, label="Batch Size", precision=0)
                    with gr.Row():
                        precision_dropdown = gr.Dropdown(
                            choices=["fp32", "fp16", "int8"],
                            value="fp16",
                            label="量化精度选择"
                        )
                    convert_btn = gr.Button("一键转换", variant="primary")
                
                with gr.Column(scale=2):
                    log_output = gr.Textbox(label="转换日志 (Conversion Logs)", lines=15, autoscroll=True)

            convert_btn.click(
                fn=convert_model_pipeline,
                inputs=[pt_file, input_size_num, batch_size_num, precision_dropdown],
                outputs=[log_output]
            )

        # [...](asc_slot://start-slot-73)--- 选项卡 2: 参数调整 (新增) ---
        with gr.Tab("参数调整"):
            
            with gr.Row():
                # [...](asc_slot://start-slot-75)左侧：在线编辑器
                with gr.Column(scale=2):
                    gr.Markdown("#### 修改config参数文件")
                    config_editor = gr.Code(label="Current Config Content", value=load_config, language="yaml", lines=20)
                    with gr.Row():
                        refresh_btn = gr.Button("刷新")
                        save_conf_btn = gr.Button("保存修改", variant="primary")
                        status_msg = gr.Textbox(label="状态", show_label=False, lines=1)
                
                # [...](asc_slot://start-slot-77)右侧：文件上传覆盖
                with gr.Column(scale=1):
                    gr.Markdown("#### 上传配置文件覆盖")
                    upload_conf_file = gr.File(label="上传 .yaml 文件", file_types=[".yaml", ".yml"])
                    overwrite_btn = gr.Button("覆盖当前配置 (Overwrite)")
            
            # 绑定事件
            refresh_btn.click(fn=load_config, inputs=[], outputs=[config_editor])
            save_conf_btn.click(fn=save_config, inputs=[config_editor], outputs=[status_msg])
            overwrite_btn.click(fn=upload_config_file, inputs=[upload_conf_file], outputs=[config_editor, status_msg])

        # [...](asc_slot://start-slot-79)--- 选项卡 3: 模型推理 (包含现有功能) ---
        with gr.Tab("模型推理"):
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
    demo.launch(server_name="0.0.0.0", share=False)
