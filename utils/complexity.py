import torch
from thop import profile
from thop import clever_format

def compute_model_complexity(model, input_size=(1, 3, 224, 224), device="cpu"):
    dummy_input = torch.randn(input_size).to(device)
    flops, params = profile(model, inputs=(dummy_input, ), verbose=False)
    flops_fmt, params_fmt = clever_format([flops, params], "%.2f")
    
    return params_fmt, flops_fmt