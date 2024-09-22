# Heterogeneous Computing Sommersemester 2024 Übung 3

Performancetest einer Klassifizierungsaufgabe. Bilderkennung mit einem Neuronalen Netz. Die Ausführung ist möglich auf 
CPU, GPU und Hailo8l Chip. Getestet wurde i9 13900HX, NVidia GeForce RTX 4070, RaspberryPi 5 CPU und RapsberryPi 5 mit
Hailo8l.

### Ausführung
Die main.py startet die Anwendung. Folgende Parameter sind einstellbar:
- -d, --device: Device für die Ausführung [cpu, cuda, hailo]
- -i, --image:  Pfad zum Bild, das klassifiziert werden soll.