import torch


#good repos to have here
#https://github.com/xuanandsix/DIS-onnxruntime-and-tensorrt-demo

def main():
    net = ISNetDIS()
    net.load_state_dict(torch.load('../saved_models/isnet.pth', map_location=torch.device('cpu')))
    input = torch.randn(1, 3, 1024, 1024, device='cpu')
    torch.onnx.export(net, input, '../saved_models/isnet.onnx',
                      export_params=True, opset_version=11, do_constant_folding=True,
                      input_names = ['input'])

if __name__ == "__main__":
    main()