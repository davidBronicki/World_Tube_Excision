import numpy as np
from typing import Callable
import types


class CoordinateFrame:
	def __init__(self, name: str, dim: int):
		assert(dim > 0), 'non-positive dimension'
		self.name = name
		self.dim = dim


class CoordinatePos:
	def __init__(self, frame: CoordinateFrame, coords: np.ndarray):
		assert(coords.shape == (frame.dim,)), 'incorrect coordinate count'
		self.frame = frame
		self.coords = coords

	def inFrame(self, frame: CoordinateFrame):
		assert(frame is self.frame), 'coordinate not in correct frame'


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
			position.inFrame(self.initialFrame, 'incorrect coordinate frame')
			return CoordinatePos(self.finalFrame, mapFunct(position.coords))
		self.map = types.MethodType(newMap, self)

	def setReverseMap(self, mapFunct: Callable[[np.ndarray], np.ndarray]):
		def newMap(_self, position: CoordinatePos) -> CoordinatePos:
			position.inFrame(self.finalFrame, 'incorrect coordinate frame')
			return CoordinatePos(self.initialFrame, mapFunct(position.coords))
		self.reverseMap = types.MethodType(newMap, self)

	def setJacobian(self,
					jacobianFunct: Callable[[np.ndarray], np.ndarray],
					setInverse=False):

		def newJacobian(_self, position: CoordinatePos) -> CoordinatePos:
			position.inFrame(self.initialFrame, 'incorrect coordinate frame')
			return jacobianFunct(position.coords)
		self.jacobian = types.MethodType(newJacobian, self)

		if setInverse:
			def newInvJacobian(_self, position: CoordinatePos) -> CoordinatePos:
				position.inFrame(self.initialFrame,
								 'incorrect coordinate frame')
				return np.linalg.inv(jacobianFunct(position.coords))
			self.inverseJacobian = types.MethodType(newInvJacobian, self)

	def setInverseJacobian(self,
						   invJacobianFunct: Callable[[np.ndarray], np.ndarray],
						   setJacobian=False):

		def newInvJacobian(_self, position: CoordinatePos) -> CoordinatePos:
			position.inFrame(self.initialFrame, 'incorrect coordinate frame')
			return invJacobianFunct(position.coords)
		self.inverseJacobian = types.MethodType(newInvJacobian, self)

		if setJacobian:
			def newJacobian(_self, position: CoordinatePos) -> CoordinatePos:
				position.inFrame(self.initialFrame,
								 'incorrect coordinate frame')
				return np.linalg.inv(invJacobianFunct(position.coords))
			self.jacobian = types.MethodType(newJacobian, self)

	def setGradJacobian(self, gradJacFunct: Callable[[np.ndarray], np.ndarray]):
		def newGradJac(_self, position: CoordinatePos) -> CoordinatePos:
			position.inFrame(self.initialFrame, 'incorrect coordinate frame')
			return gradJacFunct(position.coords)
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

		def reversedGradJacobian(outputSelf, coordinates: CoordinatePos) -> np.ndarray:
			coordinates = self.reverseMap(coordinates)
			gradJac = self.gradJacobian(coordinates)
			invJac = self.inverseJacobian(coordinates)
			return np.einsum('bca,ak,cj,ib->ijk',
							 gradJac, invJac, invJac, invJac)
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
			outerJac = self.jacobian(intermediateCoords)
			innerJac = other.jacobian(coords)
			return np.dot(outerJac, innerJac)
		output.jacobian = types.MethodType(newJacobian, output)

		def newInvJacobian(outputSelf, coords: CoordinatePos):
			intermediateCoords = other.map(coords)
			outerJac = self.inverseJacobian(intermediateCoords)
			innerJac = other.inverseJacobian(coords)
			return np.dot(innerJac, outerJac)
		output.inverseJacobian = types.MethodType(newInvJacobian, output)

		def newGradJacobian(outputSelf, coords: CoordinatePos) -> np.ndarray:
			intermediateCoords = other.map(coords)
			innerGradJac = other.gradJacobian(coords)
			innerJac = other.jacobian(coords)
			outerGradJac = self.gradJacobian(intermediateCoords)
			outerJac = self.jacobian(intermediateCoords)

			innerGradJac = np.einsum('ajk,ia->ijk', innerGradJac, outerJac)
			outerGradJac = np.einsum('iab,bk,aj->ijk',
									 outerGradJac, innerJac, innerJac)

			return innerGradJac + outerGradJac
		output.gradJacobian = types.MethodType(newGradJacobian, output)
		return output

	def map(self, coordinates: CoordinatePos) -> CoordinatePos:
		raise(NotImplementedError(
			'user left map undefined: must be set manually'))

	def reverseMap(self, coordinates: CoordinatePos) -> CoordinatePos:
		raise(NotImplementedError(
			'user left reverse map undefined: must be set manually'))

	def jacobian(self, coordinates: CoordinatePos) -> np.ndarray:
		raise(NotImplementedError(
			'user left jacobian undefined: must be set manually'))

	def inverseJacobian(self, coordinates: CoordinatePos) -> np.ndarray:
		coordinates.inFrame(self.initialFrame, 'incorrect coordinate frame')
		return np.linalg.inv(self.jacobian(coordinates))

	def gradJacobian(self, coordinates: CoordinatePos) -> np.ndarray:
		# TODO: implement numerical gradient?
		raise(NotImplementedError(
			'user left grad jacobian undefined: must be set manually'))
