import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

from src.profiler import Profiler

class_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
profiler = Profiler()


# Function to classify a single image
def bench_classification(image_path, dev: str):
    '''
    Hauptfunktion, die das Modell läd, die Vorverarbeitung duchführt und den Profiler startet.

    '''
    device = torch.device(dev)

    # Load the MobileNetV2 model and modify the classifier for your task (e.g., CIFAR-10 with 10 classes)
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 10)  # Assuming 10 output classes (CIFAR-10)
    model.load_state_dict(
        torch.load("model/mobilenetv2_cifar10.pth", weights_only=False, map_location=device))  # Load custom weights
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    # Preprocessing function to match the input format for MobileNetV2
    preprocess = transforms.Compose([
        transforms.Resize(256),  # Resize the image so that the shorter side is 256 pixels
        transforms.CenterCrop(224),  # Crop the center to get a 224x224 image
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(  # Normalize with mean and std of ImageNet dataset (used for pretraining)
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    # Load the image and apply preprocessing
    image = Image.open(image_path).convert('RGB')
    preprocess = preprocess(image).unsqueeze(0)

    profiler.profile(use_model, device, preprocess, model)


def use_model(device, preprocess, model):
    '''
    Inferenzschritt auf dem CPU oder der GPU. Die Laufzeit dieser Funktion wird gemessen.
    '''
    # move to device
    input_tensor = preprocess.to(device)
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
    # Get the predicted class (highest probability)
    _, pred = torch.max(output, 1)
    # Print the predicted class (this is an index)
    print(f'Predicted class index: {int(pred)}')
    # Get the human-readable label
    print(f'Predicted label: {class_labels[pred]}')


def use(img, device):
    bench_classification(img, device)

    avg_time = profiler.get_average_time()
    print(f"Average inference time: {avg_time:.4f} seconds")
