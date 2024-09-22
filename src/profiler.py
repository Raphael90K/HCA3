import time
import numpy as np


class Profiler:
    def __init__(self, num_iterations=5):
        """
        Initialisiert den Profiler.

        """
        self.num_iterations = num_iterations
        self.times = []

    def profile(self, func, *args, **kwargs):
        """
        Misst die Laufzeit der Funktion.

        :param func: Zu messende Funktion
        :param args: args der Funktion
        :param kwargs: kwags der Funktion
        :return result: Ergebnis der letzten
        """
        print(f"Starting performance profiling for {self.num_iterations} iterations...")
        self.times = []

        for i in range(self.num_iterations):
            start_time = time.time()  # Start the timer
            result = func(*args, **kwargs)  # Run the inference method
            end_time = time.time()  # End the timer

            runtime = end_time - start_time
            self.times.append(runtime)
            print(f"Iteration {i + 1}/{self.num_iterations} took {runtime:.4f} seconds.")

        return result

    def get_average_time(self):
        """
        Gibt die durchnittliche Laufzeit zurück.

        :return: Durchschnittszeit
        """
        if not self.times:
            raise ValueError("No profiling data available. Run `profile` first.")
        return np.mean(self.times)

    def reset(self):
        """
        Setzt die Zeiten zurück.

        """
        self.times = []
