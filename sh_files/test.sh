# ================= 配置区域 =================
# 1. 这里必须填你 log 文件夹里的真实路径
# 请去文件管理器看一眼 log 文件夹下的日期目录名，比如 log/2026-02-03-14-xx-xx/best_model.pth
MODEL_PATH="./log/2026-03-02-21-31-47/best_model.pth"

# 2. 数据集名称 (你训练的是 PGPS9K/Geometry3K)
DATASET="Geometry3K"
# ===========================================

CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch \
--nproc_per_node=1 \
--use_env \
--master_port=12355 \
start.py \
--dataset ${DATASET} \
--use_MLM_pretrain \
--evaluate_only \
--eval_method completion \
--resume_model ${MODEL_PATH} \
--workers 0 \
--batch_size 32