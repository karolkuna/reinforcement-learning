import numpy as np

class MovingAverage:
	def __init__(self, window_length):
		self.window_length = window_length
		self.t = 0
		self.window = np.zeros(window_length)
		self.sum = 0

	def add_value(self, new_value):
		idx = self.t % self.window_length
		old_value = self.window[idx]
		self.window[idx] = new_value
		self.t += 1
		self.sum = self.sum + new_value - old_value

	def get_average(self):
		#return self.sum # TODO: fix it
		return self.window.sum() / self.window_length