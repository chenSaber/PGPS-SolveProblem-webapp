MODEL_PATH="./log/2026-03-02-12-30-20/320.pth"

CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch \
--nproc_per_node=1 \
--use_env \
--master_port=12355 \
start.py \
--dataset PGPS9K \
--use_MLM_pretrain \
--workers 0 \
--batch_size 32 \
--print_freq 100 \
--resume_model ${MODEL_PATH}