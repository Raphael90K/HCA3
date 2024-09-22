# Heterogeneous Computing Sommersemester 2024 Übung 3

Performancetest einer Klassifizierungsaufgabe. Bilderkennung mit einem Neuronalen Netz. Die Ausführung ist möglich auf
CPU, GPU und Hailo8l Chip. Getestet wurde ein i9 Prozessor, eine Nvidia GeForce RTX Grafikkarte, ein RaspberryPi 5 CPU
und
ein RapsberryPi 5 mit Hailo8l.

### Start der Anwendung:

Die main.py startet die Anwendung. Folgende Parameter sind einstellbar:

- -d, --device: Device für die Ausführung [cpu, cuda, hailo]
- -i, --image:  Pfad zum Bild, das klassifiziert werden soll.

### Messergebnisse:

![Ergebnisse](/results/Grafik.png){:.centered}
<div align="center">
    <img src="/results/Grafik.png">
</div>

