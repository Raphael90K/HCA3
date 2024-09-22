import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models


def create_and_train(dev: str = "cuda"):
    device = torch.device(dev)

    # Hyperparameter
    num_epochs = 5
    batch_size = 16
    learning_rate = 0.001

    # CIFAR-10 Dataset (Beispiel)
    transform = transforms.Compose([
        transforms.Resize(224),  # MobileNet erwartet 224x224 Bilder
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # MobileNetV2-Modell laden
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, 10)  # CIFAR-10 hat 10 Klassen
    model = model.to(device)

    # Verlustfunktion und Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')

    # Modell speichern
    torch.save(model.state_dict(), '../mobilenetv2_cifar10.pth')


create_and_train("cuda")
