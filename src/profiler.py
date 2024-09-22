import time
import numpy as np


class Profiler:
    def __init__(self, num_iterations=5):
        """
        Initializes the Profiler.
        :param num_iterations: Number of iterations to run for profiling.
        """
        self.num_iterations = num_iterations
        self.times = []

    def profile(self, func, *args, **kwargs):
        """
        Profiles the given inference method over multiple iterations.
        :param func: The function to profile.
        :param args: Positional arguments to pass to the inference method.
        :param kwargs: Keyword arguments to pass to the inference method.
        :return: Inference results from the last iteration.
        """
        print(f"Starting performance profiling for {self.num_iterations} iterations...")
        self.times = []

        for i in range(self.num_iterations):
            start_time = time.time()  # Start the timer
            result = func(*args, **kwargs)  # Run the inference method
            end_time = time.time()  # End the timer

            inference_time = end_time - start_time
            self.times.append(inference_time)
            print(f"Iteration {i + 1}/{self.num_iterations} took {inference_time:.4f} seconds.")

        return result

    def get_average_time(self):
        """
        Returns the average time per inference.
        :return: Average inference time in seconds.
        """
        if not self.times:
            raise ValueError("No profiling data available. Run `profile` first.")
        return np.mean(self.times)

    def reset(self):
        """
        Resets the recorded times.
        """
        self.times = []
