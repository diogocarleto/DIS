import torch
import sys
from models import *

#good repos to have here
#https://github.com/xuanandsix/DIS-onnxruntime-and-tensorrt-demo
#https://huggingface.co/spaces/ECCV2022/dis-background-removal/tree/main

def main(pth_model_path, onnx_model_path):
    net = ISNetDIS()
    # '../saved_models/isnet.pth'
    net.load_state_dict(torch.load(pth_model_path, map_location=torch.device('cpu')))
    input = torch.randn(1, 3, 1024, 1024, device='cpu')
    # '../saved_models/isnet.onnx'
    torch.onnx.export(net,
                      input,
                      onnx_model_path,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names = ['input'])

if __name__ == "__main__":
    pth_model_path = sys.argv[1]
    onnx_model_path = sys.argv[2]

    main(pth_model_path, onnx_model_path)

# usage example
#python3.10 tensor_to_onnx_model_converter.py ../saved_models/IS-Net-test/gpu_itr_10_traLoss_0.2452_traTarLoss_0.0237_valLoss_0.1456_valTarLoss_0.011_maxF1_0.9961_mae_0.0079_time_2.16715.pth ../saved_models/IS-Net-test/isnet.onnx