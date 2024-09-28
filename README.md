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

<div align="center">
    <img src="/results/Grafik.png">
</div>

### Fazit:

Das Experiment im Bericht zeigt, dass ein einfaches Neuronales Netz auf den Komponenten CPU, GPU und
Hailo8L Chip für die Erkennung eines Katzenbildes genutzt werden kann. Es wurde in allen Fällen das gleiche Modell
verwendet, welches jedoch für den Hailo8L Chip zunächst in ein spezielles Format umgewandelt werden musste. Im Rahmen
des Experiments wurden dann die benötigten Laufzeiten auf den Hardwarekomponenten Rapsberry Pi CPU, Notebook CPU,
Notebook Grafikkarte und Raspberry Pi mit Hailo8L Chip gemessen und verglichen. Der Hauptprozessor des Rapsberry Pi 5
benötigte hierbei erwartungsgemäß die deutlich längste Laufzeit. Überraschend zeigte sich jedoch, dass der Hailo8L Chip
im Mittel die beste Laufzeit erreichte und somit besser abschnitt, als der relativ leistungsstarke Intel i9 Prozessor
und die Nvidia GeForce RTX 4070 Grafikkarte. Überraschend war ebenfalls, dass die Grafikkarte im Durchschnitt
schlechtere Ergebnisse als der Intel i9 Prozessor erzielte. Auf eine einzelne Iterationen bezogen erreichte die
Grafikkarte zwar die beste Laufzeit, der erwartete theoretische Speedup konnte durch das Experiment jedoch nicht
bestätigt werden.

