import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image


# Define the device (use GPU if available, otherwise fallback to CPU)


# Function to classify a single image
def classify_image(image_path, dev: str = "cuda"):
    device = torch.device(dev)

    # Load the MobileNetV2 model and modify the classifier for your task (e.g., CIFAR-10 with 10 classes)
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 10)  # Assuming 10 output classes (CIFAR-10)
    model.load_state_dict(torch.load("../model/mobilenetv2_cifar10.pth", weights_only=True))  # Load custom weights
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

    predicted_class = use_model(device, image_path, model, preprocess)

    return predicted_class.item()


def use_model(device, image_path, model, preprocess):
    # Load the image and apply preprocessing
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
    # Get the predicted class (highest probability)
    _, pred = torch.max(output, 1)
    return pred


# Test the function with an image
image_path = '../img/1.jpg'  # Replace with the path to your image
predicted_class = classify_image(image_path, "cuda")

# Print the predicted class (this is an index)
print(f'Predicted class: {predicted_class}')

# If you have a class label mapping (e.g., for CIFAR-10):
# Here is a sample mapping for CIFAR-10
class_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Get the human-readable label
print(f'Predicted label: {class_labels[predicted_class]}')
