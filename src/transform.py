import torch.onnx
from torch import nn
from torchvision import models


def main(dev: str = "cuda"):
    device = torch.device(dev)
    # Dummy-Eingabe für ONNX-Export (1 Bild von Größe 3x224x224)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, 10)  # CIFAR-10 hat 10 Klassen
    model = model.to(device)
    model.load_state_dict(torch.load("../mobilenetv2_cifar10.pth", weights_only=True))

    model.eval()
    onnx = torch.onnx.dynamo_export(model, dummy_input)
    onnx.save("mobilenetv2_cifar10.onnx")


main("cuda")
