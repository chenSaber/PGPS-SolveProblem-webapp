import torch

# 1. 指向你刚才跑的那个模型路径
input_path = './log/2026-02-08-13-45-52/best_model.pth'
output_path = './log/2026-02-08-13-45-52/best_model_fixed.pth'

print(f"📥 正在加载原始权重: {input_path}")
checkpoint = torch.load(input_path, map_location='cpu')

# 2. 核心脱马甲逻辑
old_state_dict = checkpoint['state_dict']
new_state_dict = {}

for k, v in old_state_dict.items():
    # 强制去掉 module. 前缀
    name = k.replace('module.', '')
    new_state_dict[name] = v

checkpoint['state_dict'] = new_state_dict

# 3. 保存新权重
torch.save(checkpoint, output_path)
print(f"✅ 修复完成！已保存至: {output_path}")