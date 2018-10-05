import numpy as np


def calculateP(z):
	"""Given z return softmax p"""
	newZ = z - np.max(z, axis = 0)
	expz = np.exp(newZ)
	sumExpz = np.sum(expz, axis = 0)
	return expz / sumExpz


class crossEntropyLogit:
	def __init__(self):
		self.p = None
		self.j = None
		self.numInstance = None

	def doForward(self, z, y):
		"""Given z and y returns J"""
		self.numInstance = y.shape[1]
		self.p = calculateP(z)
		logP = np.log(self.p)
		yp = y * logP
		self.j = -np.sum(yp) / self.numInstance
		return self.j

	def doBackward(self, y):
		return (self.p - y) / self.numInstance


if __name__ == '__main__':
	np.random.seed(1)
	z = np.random.rand(3, 2)
	CE = crossEntropyLogit()
	y = np.eye(3)[:, :2]
	J1 = CE.doForward(z, y)
	print('J1:', J1)
	J2 = CE.doForward(z + 1000, y)
	print('J2:', J2)
	dz = CE.doBackward(y)
	print(dz)