import torch
from torch import nn
from torchvision import models


def main(dev: str = "cuda"):
    device = torch.device(dev)

    # Dummy input for ONNX export (1 image of size 3x224x224)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # Load MobileNetV2 pretrained on ImageNet
    model = models.mobilenet_v2(pretrained=True)

    # Modify the classifier for CIFAR-10 (10 classes)
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    model = model.to(device)

    # Load the custom trained weights for CIFAR-10
    model.load_state_dict(torch.load("mobilenetv2_cifar10.pth"))

    # Set the model to evaluation mode
    model.eval()

    # Export the model to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        "mobilenetv2_cifar10.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print("Model has been successfully exported to ONNX format.")


main("cuda")
