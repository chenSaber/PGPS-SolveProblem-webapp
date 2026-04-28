import requests
import base64
import json
import os

# ==========================================
# 1. 核心配置区 (以后只需在这里修改题号！)
# ==========================================
PROB_ID = "prob_66"  # 👈 随意修改为 test.json 里的任意题号

TEST_JSON_PATH = "./datasets/PGPS9K_all/PGPS9K/train.json"
DIAGRAM_DIR = "./datasets/PGPS9K_all/Diagram_Visual/"
API_URL = "http://127.0.0.1:8000/solve"


# ==========================================


def load_problem_data(prob_id):
    """从 test.json 中动态加载题目数据"""
    if not os.path.exists(TEST_JSON_PATH):
        print(f" 找不到测试集文件: {TEST_JSON_PATH}")
        return None

    try:
        with open(TEST_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if prob_id not in data:
            print(f" 错误: 在测试集中找不到题目 {prob_id}")
            return None

        return data[prob_id]
    except Exception as e:
        print(f" 读取 test.json 失败: {e}")
        return None


def image_to_base64(img_path):
    """将图片文件转化为 Base64 编码"""
    if not img_path or not os.path.exists(img_path):
        print(f" 图片不存在，将进行无图推理: {img_path}")
        return None
    try:
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/png;base64,{encoded_string}"
    except Exception as e:
        print(f" 读取图片失败 ({e})")
        return None


def test_model():
    # 1. 动态加载题目
    prob_data = load_problem_data(PROB_ID)
    if not prob_data:
        return

    # 2. 提取并组装字段
    img_filename = prob_data.get("diagram", "")
    img_path = os.path.join(DIAGRAM_DIR, img_filename) if img_filename else None

    text = prob_data.get("text", "")
    stru_seqs = prob_data.get("parsing_stru_seqs", [])
    sem_seqs = prob_data.get("parsing_sem_seqs", [])
    true_answer = prob_data.get("answer", "未知")
    true_expression = prob_data.get("expression", "未知")

    # 3. 构造请求体
    payload = {
        "prob_id": PROB_ID,
        "text": text,
        "stru_seqs": stru_seqs,
        "sem_seqs": sem_seqs,
        "image_base64": image_to_base64(img_path)
    }

    print("\n" + "=" * 50)
    print(" 正在向大模型发送多模态推理请求...")
    print("=" * 50)

    try:
        # 4. 发送请求
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()

        # 5. 打印对比报告
        print(" 推理成功！获取对比报告：")
        print(f" 正在加载题目: {PROB_ID} ...")
        print(f" 【变量映射】: {json.dumps(result.get('var_map'), ensure_ascii=False)}")
        print("-" * 50)
        print(f" 【抽象算子对比】:")
        print(f"     标准答案 : {true_expression}")
        print(f"     模型预测 : {result.get('raw_expression')}")
        print("-" * 50)

        pred_ans = result.get('final_answer')
        # 简单比对一下对错
        is_correct = " 正确" if str(pred_ans) == str(true_answer) else " 错误"

        print(f" 【最终答案对比】: {is_correct}")
        print(f"     标准答案: {true_answer}")
        print(f"     模型预测: {pred_ans}")
        print("=" * 50)

    except requests.exceptions.RequestException as e:
        print(f" 请求失败，请检查 app.py 是否在运行！错误信息: {e}")


if __name__ == "__main__":
    test_model()