from matplotlib import pyplot as plt

device = ["i9 CPU", "GF RTX 4070", "RPi CPU", "Hailo8l"]
avg = [0.0332, 0.0542, 0.1387, 0.0105]

fig, ax = plt.subplots()

ax.bar(device, avg)
ax.set_title('Durchschnittliche Laufzeit pro Hardware')
ax.set_ylabel('Laufzeit in Sekunden')
plt.savefig('Grafik.png')