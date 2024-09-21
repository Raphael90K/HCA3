from hailo_platform import GraphRunner
import numpy as np

# Hailo Modell laden
runner = GraphRunner.load("mobilenetv2_cifar10.hailo")

# Beispiel-Bild laden (Dummy-Bild)
input_image = np.random.rand(1, 3, 224, 224).astype(np.float32)  # Dummy-Bild 1x3x224x224

# Inferenz ausführen
output = runner.run(input_image)

# Ausgabe verarbeiten
print("Inferenz abgeschlossen, Ausgabe:", output)
