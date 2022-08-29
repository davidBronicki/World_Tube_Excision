import numpy as np
from typing import Callable, Union
import types


class CoordinateFrame:
	def __init__(self, name: str, dim: int):
		assert(dim > 0), 'non-positive dimension'
		self.name = name
		self.dim = dim

	def getIdentityTransform(self):
		output = Transformation(self, self)

		output.setMap(lambda x: x)
		output.setReverseMap(lambda x: x)
		output.setJacobian(lambda x: np.identity(2))
		output.setInverseJacobian(lambda x: np.identity(2))
		output.setGradJacobian(lambda x: np.zeros((2,2,2)))
		output.setInverse(output)

		return output


class CoordinatePos:
	def __init__(self, frame: CoordinateFrame, coords: np.ndarray):
		assert(coords.shape == (frame.dim,)), 'incorrect coordinate count'
		self.frame = frame
		self.coords = coords

	def _wrongFrameMessage(self, otherFrame: CoordinateFrame):
		return 'Coordinate position not in correct frame. Coordinate in '+\
			self.frame.name + ' but was asserted to be in ' + otherFrame.name

	def inFrame(self,
		frame: CoordinateFrame):

		assert(frame is self.frame), _wrongFrameMessage(frame)


class FramedTensor:
	def __init__(self,
		data: Union[np.ndarray, float],
		pos: CoordinatePos,
		upIndices: int):

		if type(data) != np.ndarray:
			data = np.array([float(data)])
			self.rank = 0
		else:
			self.rank = len(data.shape)

		self.data = data
		self.pos = pos
		self.frame = pos.frame

		self.upIndices = upIndices
		self.downIndices = self.rank - upIndices

	def transform(self, transformation: 'Transformation'):
		jac = transformation.jacobian(self.pos)
		invJac = transformation.inverseJacobian(self.pos)

		newPos = transformation.map(self.pos)
		newData = self.data.copy()
		for i in range(self.upIndices):
			newData = np.tensordot(newData, jac.data, ([0], [0]))
		for i in range(self.downIndices):
			newData = np.tensordot(newData, invJac.data, ([self.upIndices], [0]))

		return FramedTensor(newData, newPos, self.upIndices)


class Transformation:
	def __init__(self,
				 fromFrame: CoordinateFrame,
				 toFrame: CoordinateFrame):

		self.dim = fromFrame.dim
		assert (fromFrame.dim == toFrame.dim), 'different dimensions'
		self.initialFrame = fromFrame
		self.finalFrame = toFrame

	def setInverse(self, inverseTransformation: 'Transformation'):
		def newInverse(_self):
			return inverseTransformation
		self.inverse = types.MethodType(newInverse, self)

	def setMap(self, mapFunct: Callable[[np.ndarray], np.ndarray]):
		def newMap(_self, position: CoordinatePos) -> CoordinatePos:
			position.inFrame(self.initialFrame)
			return CoordinatePos(self.finalFrame, mapFunct(position.coords))
		self.map = types.MethodType(newMap, self)

	def setReverseMap(self, mapFunct: Callable[[np.ndarray], np.ndarray]):
		def newMap(_self, position: CoordinatePos) -> CoordinatePos:
			position.inFrame(self.finalFrame)
			return CoordinatePos(self.initialFrame, mapFunct(position.coords))
		self.reverseMap = types.MethodType(newMap, self)

	def setJacobian(self,
					jacobianFunct: Callable[[np.ndarray], np.ndarray],
					setInverse=False):

		def newJacobian(_self, position: CoordinatePos) -> FramedTensor:
			position.inFrame(self.initialFrame)
			return FramedTensor(jacobianFunct(position.coords), position, 1)
		self.jacobian = types.MethodType(newJacobian, self)

		if setInverse:
			def newInvJacobian(_self, position: CoordinatePos) -> FramedTensor:
				position.inFrame(self.initialFrame,
								 'incorrect coordinate frame')
				return FramedTensor(np.linalg.inv(jacobianFunct(position.coords)), position, 1)
			self.inverseJacobian = types.MethodType(newInvJacobian, self)

	def setInverseJacobian(self,
						   invJacobianFunct: Callable[[np.ndarray], np.ndarray],
						   setJacobian=False):

		def newInvJacobian(_self, position: CoordinatePos) -> FramedTensor:
			position.inFrame(self.initialFrame)
			return FramedTensor(invJacobianFunct(position.coords), position, 1)
		self.inverseJacobian = types.MethodType(newInvJacobian, self)

		if setJacobian:
			def newJacobian(_self, position: CoordinatePos) -> FramedTensor:
				position.inFrame(self.initialFrame,
								 'incorrect coordinate frame')
				return FramedTensor(np.linalg.inv(invJacobianFunct(position.coords)), position, 1)
			self.jacobian = types.MethodType(newJacobian, self)

	def setGradJacobian(self, gradJacFunct: Callable[[np.ndarray], np.ndarray]):
		def newGradJac(_self, position: CoordinatePos) -> FramedTensor:
			position.inFrame(self.initialFrame)
			return FramedTensor(gradJacFunct(position.coords), position, 1)
		self.gradJacobian = types.MethodType(newGradJac, self)

	def inverse(self):
		output = Transformation(
			self.finalFrame,
			self.initialFrame
		)
		output.map = types.MethodType(
			lambda outputSelf, coords: self.reverseMap(coords),
			output)
		output.reverseMap = types.MethodType(
			lambda outputSelf, coords: self.map(coords),
			output)
		output.jacobian = types.MethodType(
			lambda outputSelf, coords: self.inverseJacobian(
				self.reverseMap(coords)),
			output)
		output.inverseJacobian = types.MethodType(
			lambda outputSelf, coords: self.jacobian(self.reverseMap(coords)),
			output)

		def reversedGradJacobian(outputSelf, coordinates: CoordinatePos):
			coordinates = self.reverseMap(coordinates)
			gradJac = self.gradJacobian(coordinates).data
			invJac = self.inverseJacobian(coordinates).data
			return FramedTensor(np.einsum('bca,ak,cj,ib->ijk',
				gradJac, invJac, invJac, invJac), coordinates, 1)
		output.gradJacobian = types.MethodType(
			reversedGradJacobian,
			output)
		return output

	def compose(self, other: 'Transformation'):
		assert(self.initialFrame is other.finalFrame), 'incompatable composition'
		output = Transformation(
			other.initialFrame,
			self.finalFrame
		)
		output.map = types.MethodType(
			lambda outputSelf, coords: self.map(other.map(coords)),
			output)
		output.reverseMap = types.MethodType(
			lambda outputSelf, coords: other.reverseMap(
				self.reverseMap(coords)),
			output)

		def newJacobian(outputSelf, coords: CoordinatePos):
			intermediateCoords = other.map(coords)
			outerJac = self.jacobian(intermediateCoords).data
			innerJac = other.jacobian(coords).data
			return FramedTensor(np.dot(outerJac, innerJac), coords, 1)
		output.jacobian = types.MethodType(newJacobian, output)

		def newInvJacobian(outputSelf, coords: CoordinatePos):
			intermediateCoords = other.map(coords)
			outerJac = self.inverseJacobian(intermediateCoords).data
			innerJac = other.inverseJacobian(coords).data
			return FramedTensor(np.dot(innerJac, outerJac), coords, 1)
		output.inverseJacobian = types.MethodType(newInvJacobian, output)

		def newGradJacobian(outputSelf, coords: CoordinatePos):
			intermediateCoords = other.map(coords)
			innerGradJac = other.gradJacobian(coords).data
			innerJac = other.jacobian(coords).data
			outerGradJac = self.gradJacobian(intermediateCoords).data
			outerJac = self.jacobian(intermediateCoords).data

			innerGradJac = np.einsum('ajk,ia->ijk', innerGradJac, outerJac)
			outerGradJac = np.einsum('iab,bk,aj->ijk',
									 outerGradJac, innerJac, innerJac)

			return FramedTensor(innerGradJac + outerGradJac, coords, 1)
		output.gradJacobian = types.MethodType(newGradJacobian, output)
		return output

	def map(self, coordinates: CoordinatePos) -> CoordinatePos:
		raise(NotImplementedError(
			'user left map undefined: must be set manually'))

	def reverseMap(self, coordinates: CoordinatePos) -> CoordinatePos:
		raise(NotImplementedError(
			'user left reverse map undefined: must be set manually'))

	def jacobian(self, coordinates: CoordinatePos) -> FramedTensor:
		raise(NotImplementedError(
			'user left jacobian undefined: must be set manually'))

	def inverseJacobian(self, coordinates: CoordinatePos) -> FramedTensor:
		coordinates.inFrame(self.initialFrame)
		return np.linalg.inv(self.jacobian(coordinates))

	def gradJacobian(self, coordinates: CoordinatePos) -> FramedTensor:
		# TODO: implement numerical gradient?
		raise(NotImplementedError(
			'user left grad jacobian undefined: must be set manually'))

	def transformTensor(self, tensor: FramedTensor):
		return tensor.transform(self)
		# jac = self.jacobian(tensor.pos)
		# invJac = self.inverseJacobian(tensor.pos)

		# newPos = self.map(tensor.pos)
		# newData = tensor.data.copy()
		# for i in range(tensor.upIndices):
		# 	newData = np.tensordot(newData, jac, ([i], [0]))
		# for i in range(tensor.downIndices):
		# 	newData = np.tensordot(newData, invJac, ([tensor.upIndices + i], [0]))

		# return FramedTensor(newData, newPos, tensor.upIndices)
