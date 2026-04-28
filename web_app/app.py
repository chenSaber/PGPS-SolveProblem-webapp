# http://127.0.0.1:8000/docs
import sys
import os
import logging
import uvicorn
import traceback
import base64
import io
from PIL import Image
import torchvision.transforms as transforms
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import json

from algebra_engine import build_visualization_steps

# 1. 强制路径挂载
CURRENT_DIR = os.getcwd()
sys.path.insert(0, CURRENT_DIR)

# 2. 导入配置、网络与词典
from config.config_default import get_parser
from core.network import Network
from datasets.preprossing import SrcLang, TgtLang, SN
from datasets.utils import get_combined_text, get_text_index, get_var_arg, get_token, split_text, get_point_angleID_tag, \
    get_num_arg_tag, remove_sem_dup
from datasets.operators import result_compute, normalize_exp
from utils import sequence_mask

app = FastAPI(title="PGPS 几何推理可视化后端", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
global_model = None
global_args = None
global_src_lang = None
global_tgt_lang = None


class QuestionRequest(BaseModel):
    prob_id: str
    text: str
    stru_seqs: list
    sem_seqs: list
    image_base64: str = None  # 接收前端传来的图像 Base64 编码


class StepResponse(BaseModel):
    step_num: int
    action: str
    equation: str
    result: float = None


@app.on_event("startup")
def load_ai_model():
    global global_model, global_args, global_src_lang, global_tgt_lang
    print("正在初始化 FastAPI 后端服务...")

    old_argv = sys.argv
    sys.argv = ['app.py']
    try:
        global_args = get_parser()
    finally:
        sys.argv = old_argv

    global_args.use_MLM_pretrain = True
    logging.basicConfig(level=logging.INFO)
    global_args.logger = logging.getLogger("FastAPI_Backend")

    print("正在加载源语言与目标语言词典...")
    src_path = os.path.join(CURRENT_DIR, "vocab", "vocab_src.txt")
    tgt_path = os.path.join(CURRENT_DIR, "vocab", "vocab_tgt.txt")

    global_src_lang = SrcLang(src_path)
    global_tgt_lang = TgtLang(tgt_path)

    print("正在构建多模态神经网络...")
    global_model = Network(global_args, global_src_lang, global_tgt_lang)

    # 请确保此处的权重路径是你本地有效的路径！
    model_path = os.path.join(CURRENT_DIR, "log", "2026-03-02-21-31-47", "best_model.pth")
    print(f"正在加载原始权重: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        old_state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in old_state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v

        global_model.load_state_dict(new_state_dict, strict=True)
        global_model.eval()
        if torch.cuda.is_available():
            global_model = global_model.cuda()
        print("模型加载完毕，核心引擎待命！")
    except Exception as e:
        print(f"模型加载失败，请检查路径是否正确: {e}")


def to_tensor(data_list):
    t = torch.LongTensor(data_list).unsqueeze(0)
    return t.cuda() if torch.cuda.is_available() else t


# 新增：根据题号获取 test.json 中的题目详情
@app.get("/get_test_problem/{prob_id}")
async def get_test_problem(prob_id: str):
    test_json_path = os.path.join(CURRENT_DIR, "datasets/PGPS9K_all/PGPS9K/train.json")
    diagram_dir = os.path.join(CURRENT_DIR, "datasets/PGPS9K_all/Diagram_Visual/")

    if not os.path.exists(test_json_path):
        raise HTTPException(status_code=404, detail="找不到 test.json 文件")

    try:
        with open(test_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 兼容处理：有时候题号是 "6522"，有时候是 "prob_6522"
        real_prob_id = prob_id
        if real_prob_id not in data and real_prob_id.replace("prob_", "") in data:
            real_prob_id = real_prob_id.replace("prob_", "")

        if real_prob_id not in data:
            raise HTTPException(status_code=404, detail=f"题号 {prob_id} 不存在")

        prob = data[real_prob_id]

        # 处理图片转码
        img_base64 = None
        img_filename = prob.get("diagram", "")
        if img_filename:
            img_path = os.path.join(diagram_dir, img_filename)
            if os.path.exists(img_path):
                with open(img_path, "rb") as f_img:
                    encoded = base64.b64encode(f_img.read()).decode('utf-8')
                    img_base64 = f"data:image/png;base64,{encoded}"

        # 🚨 极度安全的字段提取：防 null 崩溃机制
        text_content = prob.get("text") or ""
        stru_list = prob.get("parsing_stru_seqs") or []
        sem_list = prob.get("parsing_sem_seqs") or []

        return {
            "text": str(text_content),
            "stru_seqs": "\n".join(stru_list),
            "sem_seqs": "\n".join(sem_list),
            "image_base64": img_base64
        }
    except Exception as e:
        # 打印详细的红色崩溃日记，方便定位真凶
        print(f"读取题目 {prob_id} 失败！")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/solve")
async def solve_geometry_problem(req: QuestionRequest):
    if global_model is None:
        raise HTTPException(status_code=500, detail="模型未就绪")

    print(f"\n==========================================")
    print(f"开始实时推断：{req.prob_id}")

    try:
        # ==========================================
        # 步骤 1：处理多模态视觉流 (Vision Stream)
        # ==========================================
        if req.image_base64:
            try:
                # 提取 base64 数据并解码
                img_str = req.image_base64.split(',')[1] if ',' in req.image_base64 else req.image_base64
                img_data = base64.b64decode(img_str)
                image = Image.open(io.BytesIO(img_data)).convert('RGB')

                # 缩放为 128x128，并进行 ImageNet 标准归一化 (极其关键！)
                transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                diagram = transform(image).unsqueeze(0)
                print("视觉中枢已激活：成功加载并转换真实图像张量！")
            except Exception as e:
                print(f"图片解析失败，降级为空白张量: {e}")
                diagram = torch.zeros(1, 3, 128, 128)
        else:
            diagram = torch.zeros(1, 3, 128, 128)
            print("未检测到图片上传，视觉特征使用默认 Zero Tensor。")

        if torch.cuda.is_available():
            diagram = diagram.cuda()

        # ==========================================
        # 步骤 2：数据结构化清洗与查字典 (Tokenization)
        # ==========================================
        text_data, stru_data, sem_data = SN(), SN(), SN()

        text_data.token = get_token(req.text)
        stru_data.token = [get_token(item) + [','] for item in req.stru_seqs if item.strip()]
        sem_data.token = [get_token(item) + [','] for item in req.sem_seqs if item.strip()]

        text_data.sect_tag = []
        stru_data.sect_tag = [['[STRU]'] * len(item) for item in stru_data.token]
        sem_data.sect_tag = [['[COND]'] * len(item) for item in sem_data.token]
        split_text(text_data)

        text_data.class_tag = ['[GEN]'] * len(text_data.token)
        stru_data.class_tag = [['[GEN]'] * len(item) for item in stru_data.token]
        sem_data.class_tag = [['[GEN]'] * len(item) for item in sem_data.token]

        get_point_angleID_tag(text_data, stru_data, sem_data)
        get_num_arg_tag(text_data, sem_data)

        dummy_exp = []
        remove_sem_dup(text_data, sem_data, dummy_exp)

        combine_text = SN()
        get_combined_text(text_data, stru_data, sem_data, combine_text, global_args)

        text_token_raw, text_sect_tag_raw, text_class_tag_raw = get_text_index(combine_text, global_src_lang)
        var_arg_positions_raw, var_values, arg_values = get_var_arg(combine_text, global_args)

        text_token = to_tensor(text_token_raw)
        text_sect_tag = to_tensor(text_sect_tag_raw)
        text_class_tag = to_tensor(text_class_tag_raw)

        if len(var_arg_positions_raw) == 0:
            var_arg_positions_raw = [0]

        var_arg_positions = to_tensor(var_arg_positions_raw)

        text_len = [len(text_class_tag_raw)]
        text_len_tensor = torch.LongTensor(text_len)
        if torch.cuda.is_available(): text_len_tensor = text_len_tensor.cuda()

        text_dict = {
            'token': text_token,
            'sect_tag': text_sect_tag,
            'class_tag': text_class_tag,
            'len': text_len_tensor
        }

        len_var = [max(len(var_arg_positions_raw), 1)]
        len_tensor = torch.LongTensor(len_var)
        if torch.cuda.is_available(): len_tensor = len_tensor.cuda()

        var_dict = {
            'pos': var_arg_positions,
            'len': len_tensor,
            'var_value': [var_values],
            'arg_value': [arg_values]
        }

        # ==========================================
        # 步骤 3：送入大模型前向传播 (Forward Pass)
        # ==========================================
        with torch.no_grad():
            test_preds = global_model(diagram, text_dict, var_dict, {'exp': None})

        best_pred_idx = test_preds[0][0] if isinstance(test_preds[0][0], list) else test_preds[0]

        # ==========================================
        # 步骤 4：解码器 (Decoder) —— 把 ID 转回公式
        # ==========================================
        var2arg_dict = {'N' + str(i + len(var_values)): item for i, item in enumerate(arg_values)}
        pred_action_str = global_tgt_lang.sentence_from_indexes(best_pred_idx, var2arg_dict)
        pred_action_list = normalize_exp(pred_action_str)

        num_str_list = [str(x) for x in var_values]
        final_ans = float(result_compute(num_all_list=num_str_list, exp_tokens=pred_action_list))

        # ==========================================
        # 步骤 5：调用外部代数引擎，生成可视化 JSON
        # ==========================================
        steps, clean_var_map, raw_expression = build_visualization_steps(
            sem_seqs=req.sem_seqs,
            var_values=var_values,
            pred_action_list=pred_action_list,
            final_ans=final_ans
        )

        return {
            "status": "success",
            "prob_id": req.prob_id,
            "steps": steps,
            "final_answer": round(final_ans, 3),
            "var_map": clean_var_map,
            "raw_expression": raw_expression
        }

    except Exception as e:
        print(f"推断过程崩溃：{str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)