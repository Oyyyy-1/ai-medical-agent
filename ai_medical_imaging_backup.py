# import os
# import streamlit as st
# from PIL import Image as PILImage
# import io
# from huggingface_hub import InferenceClient
# import tempfile
# import base64
#
# # ==================== 配置区 ====================
# VISION_MODEL = "llava-hf/llava-1.5-7b-hf"  # 多模态视觉模型
#
#
# # ==================== 初始化 HF 客户端 ====================
# @st.cache_resource
# def get_hf_client(token):
#     if not token or len(token) < 20:  # 简单校验 token 长度
#         return None
#     try:
#         client = InferenceClient(token=token)
#         # 简单测试一下连接是否有效
#         client.text_generation("Hello", max_new_tokens=1)
#         return client
#     except Exception as e:
#         st.error(f"Hugging Face token 无效或网络问题: {str(e)}")
#         return None
#
#
# # ==================== Streamlit 界面 ====================
# st.title("🏥 农业/医疗图像分析 Agent（Hugging Face 版）")
# st.write("上传图像进行 AI 分析（基于 LLaVA 模型）")
#
# # 初始化 session_state
# if "hf_token" not in st.session_state:
#     st.session_state.hf_token = None
# if "hf_client" not in st.session_state:
#     st.session_state.hf_client = None
#
# # 侧边栏配置
# with st.sidebar:
#     st.title("ℹ️ 配置")
#
#     token_input = st.text_input(
#         "输入你的 Hugging Face Token:",
#         type="password",
#         value=st.session_state.hf_token if st.session_state.hf_token else "",
#         help="从 https://huggingface.co/settings/tokens 获取（需要 read 或 inference 权限）"
#     )
#
#     if st.button("保存 Token 并验证"):
#         if token_input and len(token_input) > 20:
#             st.session_state.hf_token = token_input
#             st.session_state.hf_client = get_hf_client(token_input)
#             if st.session_state.hf_client:
#                 st.success("Token 有效！已连接 Hugging Face Inference API")
#             else:
#                 st.error("Token 无效，请检查是否正确复制或权限不足")
#         else:
#             st.warning("请输入有效的 Hugging Face token")
#
#     if st.session_state.hf_token:
#         if st.button("重置 Token"):
#             st.session_state.hf_token = None
#             st.session_state.hf_client = None
#             st.rerun()
#
#     st.info(f"当前模型：{VISION_MODEL}")
#     st.caption("免费限额：约 1000 次请求/天（视模型负载而定）")
#
#     st.warning(
#         "⚠️ 声明：本工具仅供学习和参考，**不作为医疗诊断或农业决策依据**。\n"
#         "所有分析结果请由专业人员审核。"
#     )
#
# # ==================== 主界面 ====================
# uploaded_file = st.file_uploader(
#     "上传图像（支持 JPG, JPEG, PNG）",
#     type=["jpg", "jpeg", "png"],
#     help="推荐清晰的果实/叶片/医疗图像"
# )
#
# if uploaded_file is not None:
#     # 显示上传的图像
#     image = PILImage.open(uploaded_file)
#     width, height = image.size
#     aspect_ratio = width / height
#     new_width = 500
#     new_height = int(new_width / aspect_ratio)
#     resized_image = image.resize((new_width, new_height))
#
#     st.image(
#         resized_image,
#         caption="已上传图像",
#         use_container_width=True
#     )
#
#     analyze_button = st.button(
#         "🔍 开始分析",
#         type="primary",
#         use_container_width=True
#     )
#
#     if analyze_button:
#         if not st.session_state.hf_client:
#             st.error("请先在侧边栏输入并保存有效的 Hugging Face Token")
#         else:
#             with st.spinner("正在使用 LLaVA 模型分析图像...（可能需要 5-30 秒）"):
#                 try:
#                     # 保存临时文件
#                     with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
#                         image.save(tmp_file.name)
#                         temp_path = tmp_file.name
#
#                     # 构建详细的分析 prompt
#                     detailed_prompt = """
# 你是一位专业的农业病害与植物图像分析专家，同时具备医学影像分析能力。
# 请对上传的图像进行详细分析，并按以下结构组织你的回答：
#
# ### 1. 图像基本信息
# - 图像类型（植物叶片/果实/医疗X光/MRI等）
# - 拍摄对象（草莓/苹果/人类器官等）
# - 图像质量（清晰/模糊/光照/角度等）
#
# ### 2. 主要发现
# - 观察到的主要特征（颜色、纹理、斑点、腐烂、畸形等）
# - 异常描述（位置、大小、形状、严重程度：无/轻微/中度/严重）
#
# ### 3. 可能诊断
# - 最可能的病害/问题（例如：灰霉病、白粉病、叶斑病、骨折等）
# - 置信度（高/中/低）
# - 2-3 个可能的其他诊断（鉴别诊断）
#
# ### 4. 通俗解释
# - 用简单易懂的语言向农民/患者解释问题
# - 给出直观的比喻（如果合适）
#
# ### 5. 建议
# - 可能的处理方法或下一步检查建议
#
# 请使用 markdown 格式，清晰分段，语言专业但易懂。
#                     """
#
#                     # 调用 Hugging Face Inference API
#                     response = st.session_state.hf_client.chat_completion(
#                         model=VISION_MODEL,
#                         messages=[
#                             {
#                                 "role": "user",
#                                 "content": [
#                                     {"type": "image_url",
#                                      "image_url": {"url": f"data:image/png;base64,{image_to_base64(image)}"}},
#                                     {"type": "text", "text": detailed_prompt}
#                                 ]
#                             }
#                         ],
#                         max_tokens=800,
#                         temperature=0.7
#                     )
#
#                     st.markdown("### 📋 分析结果")
#                     st.markdown("---")
#                     st.markdown(response.choices[0].message.content)
#                     st.markdown("---")
#                     st.caption("由 Hugging Face Inference API (LLaVA-1.5-7b) 生成，仅供参考")
#
#                 except Exception as e:
#                     st.error(f"分析失败：{str(e)}")
#                     st.info(
#                         "可能原因：Token 无效、模型限额已用完、网络问题、图像格式不支持等。请检查 token 是否有 inference 权限。")
#                 finally:
#                     # 清理临时文件
#                     if 'temp_path' in locals() and os.path.exists(temp_path):
#                         try:
#                             os.unlink(temp_path)
#                         except:
#                             pass
#
# else:
#     st.info("请上传一张图像开始分析")
#
#
# # 辅助函数：将 PIL Image 转为 base64
# def image_to_base64(pil_image):
#     buffered = io.BytesIO()
#     pil_image.save(buffered, format="PNG")
#     return base64.b64encode(buffered.getvalue()).decode("utf-8")




#  测试token用  hugging face 免费版模型存在诸多限制如没有提供推理服务/没有对应接口等等....
# import os
# import streamlit as st
# from PIL import Image as PILImage
# import io
# from huggingface_hub import InferenceClient
# import tempfile
# import base64
#
# # ==================== 配置区 ====================
# # VISION_MODEL = "llava-hf/llava-1.5-7b-hf"  # 多模态视觉模型
# # VISION_MODEL = "Qwen/Qwen-VL-Chat"
# VISION_MODEL = "microsoft/Florence-2-large"
# # ==================== 初始化 HF 客户端 ====================
# @st.cache_resource
# def get_hf_client(token):
#     if not token or len(token) < 20:
#         return None
#     try:
#         client = InferenceClient(token=token)
#         # 轻量测试（避免路由问题）
#         try:
#             client.chat_completion(
#                 model="gpt2",
#                 messages=[{"role": "user", "content": "Hi"}],
#                 max_tokens=1
#             )
#         except Exception as ping_err:
#             st.warning(f"连接测试轻微警告（可忽略）：{str(ping_err)}")
#         return client
#     except Exception as e:
#         st.error(f"Hugging Face 初始化失败: {str(e)}")
#         return None
#
#
# # ==================== Streamlit 界面 ====================
# st.title("🏥 农业/医疗图像分析 Agent（Hugging Face 版）")
# st.write("上传图像进行 AI 分析（基于 LLaVA 模型）")
#
# # 初始化 session_state
# if "hf_token" not in st.session_state:
#     st.session_state.hf_token = None
# if "hf_client" not in st.session_state:
#     st.session_state.hf_client = None
#
# # 侧边栏配置
# with st.sidebar:
#     st.title("ℹ️ 配置")
#
#     token_input = st.text_input(
#         "输入你的 Hugging Face Token:",
#         type="password",
#         value=st.session_state.hf_token if st.session_state.hf_token else "",
#         help="从 https://huggingface.co/settings/tokens 获取（必须勾选 Inference Providers 权限）"
#     )
#
#     if st.button("保存 Token 并验证"):
#         if token_input and len(token_input.strip()) > 20:
#             st.session_state.hf_token = token_input.strip()
#             st.session_state.hf_client = get_hf_client(token_input.strip())
#             if st.session_state.hf_client:
#                 st.success("Token 已连接！（测试通过或轻微警告可忽略）")
#             else:
#                 st.error("Token 初始化失败，请检查权限或网络")
#         else:
#             st.warning("请输入有效的 Hugging Face token（hf_ 开头，长度>20）")
#
#     if st.session_state.hf_token:
#         if st.button("重置 Token"):
#             st.session_state.hf_token = None
#             st.session_state.hf_client = None
#             st.rerun()
#
#     st.info(f"当前模型：{VISION_MODEL}")
#     st.caption("免费限额：约 1000 次请求/天（视模型负载而定）")
#
#     st.warning(
#         "⚠️ 声明：本工具仅供学习和参考，**不作为医疗诊断或农业决策依据**。\n"
#         "所有分析结果请由专业人员审核。"
#     )
#
# # ==================== 主界面 ====================
# uploaded_file = st.file_uploader(
#     "上传图像（支持 JPG, JPEG, PNG）",
#     type=["jpg", "jpeg", "png"],
#     help="推荐清晰的果实/叶片/医疗图像"
# )
#
#
# def image_to_base64(pil_image):
#     """将 PIL Image 转换为 base64 字符串"""
#     buffered = io.BytesIO()
#     pil_image.save(buffered, format="PNG")
#     return base64.b64encode(buffered.getvalue()).decode("utf-8")
#
#
# if uploaded_file is not None:
#     # 显示上传的图像
#     image = PILImage.open(uploaded_file)
#     width, height = image.size
#     aspect_ratio = width / height
#     new_width = 500
#     new_height = int(new_width / aspect_ratio)
#     resized_image = image.resize((new_width, new_height))
#
#     st.image(
#         resized_image,
#         caption="已上传图像",
#         use_container_width=True
#     )
#
#     analyze_button = st.button(
#         "🔍 开始分析",
#         type="primary",
#         use_container_width=True
#     )
#
#     if analyze_button:
#         if not st.session_state.hf_client:
#             st.error("请先在侧边栏输入并保存有效的 Hugging Face Token")
#         else:
#             with st.spinner("正在使用 LLaVA 模型分析图像...（可能需要 5-30 秒）"):
#                 try:
#                     # 转换为 base64（避免临时文件权限问题）
#                     base64_image = image_to_base64(image)
#
#                     # 构建详细的分析 prompt
#                     detailed_prompt = """
# 你是一位专业的农业病害与植物图像分析专家，同时具备医学影像分析能力。
# 请对上传的图像进行详细分析，并按以下结构组织你的回答：
#
# ### 1. 图像基本信息
# - 图像类型（植物叶片/果实/医疗X光/MRI等）
# - 拍摄对象（草莓/苹果/人类器官等）
# - 图像质量（清晰/模糊/光照/角度等）
#
# ### 2. 主要发现
# - 观察到的主要特征（颜色、纹理、斑点、腐烂、畸形等）
# - 异常描述（位置、大小、形状、严重程度：无/轻微/中度/严重）
#
# ### 3. 可能诊断
# - 最可能的病害/问题（例如：灰霉病、白粉病、叶斑病、骨折等）
# - 置信度（高/中/低）
# - 2-3 个可能的其他诊断（鉴别诊断）
#
# ### 4. 通俗解释
# - 用简单易懂的语言向农民/患者解释问题
# - 给出直观的比喻（如果合适）
#
# ### 5. 建议
# - 可能的处理方法或下一步检查建议
#
# 请使用 markdown 格式，清晰分段，语言专业但易懂。
#                     """
#
#                     # 调用 Hugging Face Inference API
#                     response = st.session_state.hf_client.chat_completion(
#                         model=VISION_MODEL,
#                         messages=[
#                             {
#                                 "role": "user",
#                                 "content": [
#                                     {"type": "image_url",
#                                      "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
#                                     {"type": "text", "text": detailed_prompt}
#                                 ]
#                             }
#                         ],
#                         max_tokens=800,
#                         temperature=0.7
#                     )
#
#                     st.markdown("### 📋 分析结果")
#                     st.markdown("---")
#                     st.markdown(response.choices[0].message.content)
#                     st.markdown("---")
#                     st.caption("由 Hugging Face Inference API (LLaVA-1.5-7b) 生成，仅供参考")
#
#                 except Exception as e:
#                     st.error(f"分析失败：{str(e)}")
#                     st.info(
#                         "常见原因及解决：1. Token 缺少 Inference Providers 权限 → 重新生成 token 并勾选；2. 模型负载高/限额 → 等几分钟或换模型；3. 网络问题 → 检查代理或重试")
#                 # 无需临时文件，无需 finally 清理
#
# else:
#     st.info("请上传一张图像开始分析")


# ollama拉取模型到本地版本
import os
import streamlit as st
from PIL import Image as PILImage
import io
import ollama
import tempfile
import base64

# ==================== 配置区 ====================
VISION_MODEL = "llava:7b"  # 本地模型

st.title("🏥 农业/医疗图像分析 Agent（Ollama 本地版）")
st.write("上传图像进行 AI 分析（基于本地 LLaVA 模型）")

# 侧边栏配置
with st.sidebar:
    st.title("ℹ️ 配置")

    st.info(f"当前模型：{VISION_MODEL}（本地 Ollama）")
    st.caption("确保已在另一个终端运行 'ollama serve'")

    # 添加手动重试按钮
    if st.button("🔄 重新检测 Ollama 服务"):
        st.rerun()

    st.warning(
        "⚠️ 声明：本工具仅供学习和参考，**不作为医疗诊断或农业决策依据**。\n"
        "所有分析结果请由专业人员审核。"
    )

# ==================== 主界面 ====================
uploaded_file = st.file_uploader(
    "上传图像（支持 JPG, JPEG, PNG）",
    type=["jpg", "jpeg", "png"],
    help="推荐清晰的果实/叶片/医疗图像"
)


def image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


if uploaded_file is not None:
    image = PILImage.open(uploaded_file)
    width, height = image.size
    aspect_ratio = width / height
    new_width = 500
    new_height = int(new_width / aspect_ratio)
    resized_image = image.resize((new_width, new_height))

    st.image(resized_image, caption="已上传图像", use_container_width=True)

    analyze_button = st.button("🔍 开始分析", type="primary", use_container_width=True)

    if analyze_button:
        with st.spinner("正在连接本地 Ollama 并分析图像...（首次可能 10-60 秒）"):
            try:
                # 先简单 ping 服务（避免启动时卡死）
                ollama.list()  # 如果这里失败，会进 except

                # 转换为 base64（Ollama 支持 base64 图像）
                base64_image = image_to_base64(image)

                detailed_prompt = """
你是一位专业的农业病害与植物图像分析专家，同时具备医学影像分析能力。
请对上传的图像进行详细分析，并按以下结构组织你的回答：

### 1. 图像基本信息
- 图像类型（植物叶片/果实/医疗X光/MRI等）
- 拍摄对象（草莓/苹果/人类器官等）
- 图像质量（清晰/模糊/光照/角度等）

### 2. 主要发现
- 观察到的主要特征（颜色、纹理、斑点、腐烂、畸形等）
- 异常描述（位置、大小、形状、严重程度：无/轻微/中度/严重）

### 3. 可能诊断
- 最可能的病害/问题（例如：灰霉病、白粉病、叶斑病、骨折等）
- 置信度（高/中/低）
- 2-3 个可能的其他诊断（鉴别诊断）

### 4. 通俗解释
- 用简单易懂的语言向农民/患者解释问题
- 给出直观的比喻（如果合适）

### 5. 建议
- 可能的处理方法或下一步检查建议

请使用 markdown 格式，清晰分段，语言专业但易懂。
                """

                response = ollama.chat(
                    model=VISION_MODEL,
                    messages=[
                        {
                            'role': 'user',
                            'content': detailed_prompt,
                            'images': [base64_image]
                        }
                    ]
                )

                st.markdown("### 📋 分析结果")
                st.markdown("---")
                st.markdown(response['message']['content'])
                st.markdown("---")
                st.caption("由本地 Ollama LLaVA-7b 生成，仅供参考")

            except Exception as e:
                st.error(f"分析失败：{str(e)}")
                st.info("""
常见原因及解决方法：
1. Ollama 服务未运行 → 在另一个终端运行 'ollama serve' 并保持窗口打开
2. 模型未加载 → 运行 'ollama run llava:7b "测试"' 预加载一次
3. 连接超时 → 检查防火墙是否拦截 127.0.0.1:11434，或重启电脑
4. ollama-python 库问题 → 运行 'pip install --upgrade ollama' 更新库
5. 内存不足 → 关闭其他程序，或用量化版 'ollama pull llava:7b-q4_0'
                """)
                st.info("点击侧边栏 '重新检测 Ollama 服务' 按钮重试")

else:
    st.info("请上传一张图像开始分析")