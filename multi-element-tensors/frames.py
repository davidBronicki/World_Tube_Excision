from asyncio import format_helpers
from imp import init_builtin, init_frozen
import numpy as np
from typing import Callable, Union, List, Tuple
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
		output.setJacobian(lambda x: np.identity(self.dim))
		output.setInverseJacobian(lambda x: np.identity(self.dim))
		output.setGradJacobian(lambda x: np.zeros((
			self.dim, self.dim, self.dim)))
		output.setInverse(output)

		return output

	def __str__(self):
		return self.name + ' (' + str(self.dim) + ' dim coord frame)'
		# return 'Frame: ' + self.name + '  Dim: ' + str(self.dim)

class CoordinatePos:
	def __init__(self, frame: CoordinateFrame, coords: np.ndarray):
		assert(coords.shape == (frame.dim,)), 'incorrect coordinate count. expected ' + str((frame.dim,)) + ' got ' + str(coords.shape)
		self.frame = frame
		self.coords = coords

	def _wrongFrameMessage(self, otherFrame: CoordinateFrame):
		return 'Coordinate position not in correct frame. Coordinate in '+\
			self.frame.name + ' but was asserted to be in ' + otherFrame.name

	def assertInFrame(self,
		frame: CoordinateFrame):

		assert(frame is self.frame), self._wrongFrameMessage(frame)

	def __str__(self):
		return str(self.coords) + ' in ' + str(self.frame)
		# return 'Coordinate: ' + str(self.coords) + ' in ' + self.frame.name + ' frame'

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
			newData = np.tensordot(newData, jac.data, ([0], [1]))
		for i in range(self.downIndices):
			newData = np.tensordot(newData, invJac.data, ([0], [0]))

		return FramedTensor(newData, newPos, self.upIndices)

	def _wrongFrameMessage(self, otherFrame: CoordinateFrame):
		return 'Framed tensor not in correct frame. Tensor in '+\
			self.frame.name + ' but was asserted to be in ' + otherFrame.name

	def assertInFrame(self,
		frame: CoordinateFrame):

		assert(frame is self.frame), self._wrongFrameMessage(frame)

	def __str__(self):
		return 'Tensor at ' + str(self.pos) + ':\n' + str(self.data)
		# return 'Tensor: ' + str(self.data) + ' at ' + 

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
			position.assertInFrame(self.initialFrame)
			return CoordinatePos(self.finalFrame, mapFunct(position.coords))
		self.map = types.MethodType(newMap, self)

	def setReverseMap(self, mapFunct: Callable[[np.ndarray], np.ndarray]):
		def newMap(_self, position: CoordinatePos) -> CoordinatePos:
			position.assertInFrame(self.finalFrame)
			return CoordinatePos(self.initialFrame, mapFunct(position.coords))
		self.reverseMap = types.MethodType(newMap, self)

	def setJacobian(self,
					jacobianFunct: Callable[[np.ndarray], np.ndarray],
					setInverse=False):

		def newJacobian(_self, position: CoordinatePos) -> FramedTensor:
			position.assertInFrame(self.initialFrame)
			return FramedTensor(jacobianFunct(position.coords), position, 1)
		self.jacobian = types.MethodType(newJacobian, self)

		if setInverse:
			def newInvJacobian(_self, position: CoordinatePos) -> FramedTensor:
				position.assertInFrame(self.initialFrame,
								 'incorrect coordinate frame')
				return FramedTensor(np.linalg.inv(jacobianFunct(position.coords)), position, 1)
			self.inverseJacobian = types.MethodType(newInvJacobian, self)

	def setInverseJacobian(self,
						   invJacobianFunct: Callable[[np.ndarray], np.ndarray],
						   setJacobian=False):

		def newInvJacobian(_self, position: CoordinatePos) -> FramedTensor:
			position.assertInFrame(self.initialFrame)
			return FramedTensor(invJacobianFunct(position.coords), position, 1)
		self.inverseJacobian = types.MethodType(newInvJacobian, self)

		if setJacobian:
			def newJacobian(_self, position: CoordinatePos) -> FramedTensor:
				position.assertInFrame(self.initialFrame,
								 'incorrect coordinate frame')
				return FramedTensor(np.linalg.inv(invJacobianFunct(position.coords)), position, 1)
			self.jacobian = types.MethodType(newJacobian, self)

	def setGradJacobian(self, gradJacFunct: Callable[[np.ndarray], np.ndarray]):
		def newGradJac(_self, position: CoordinatePos) -> FramedTensor:
			position.assertInFrame(self.initialFrame)
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
		output.setInverse(self)
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
		coordinates.assertInFrame(self.initialFrame)
		return np.linalg.inv(self.jacobian(coordinates))

	def gradJacobian(self, coordinates: CoordinatePos) -> FramedTensor:
		# TODO: implement numerical gradient?
		raise(NotImplementedError(
			'user left grad jacobian undefined: must be set manually'))

	def transformTensor(self, tensor: FramedTensor):
		return tensor.transform(self)

class TranslationLinearTransformation(Transformation):
	"""
	Special implementation of Transformation with special composition rules
	and built in true inverse.
	"""
	def __init__(self,
		initialFrame: CoordinateFrame,
		finalFrame: CoordinateFrame,
		jacobian: np.ndarray,
		translation: np.ndarray):

		dim = initialFrame.dim
		assert(jacobian.shape == (dim, dim)), 'jacobian of incorrect shape. Expected'\
			+ str((dim, dim)) + ' but got ' + str(jacobian.shape)
		assert(translation.shape == (dim,)), 'translation of incorrect shape. Expected '\
			+ str((dim,)) + ' but got ' + str(translation.shape)
		super().__init__(initialFrame, finalFrame)
		self._jacobian = jacobian
		self._invJac: np.ndarray = np.linalg.inv(jacobian)
		self._translation = translation
		self._invTrans: np.ndarray = -np.dot(self._invJac, translation)

		inverseTransform = Transformation(finalFrame, initialFrame)
		inverseTransform._jacobian = self._invJac
		inverseTransform._invJac = self._jacobian
		inverseTransform._translation = self._invTrans
		inverseTransform._invTrans = self._translation
		inverseTransform.__class__ = TranslationLinearTransformation

		self.setInverse(inverseTransform)
		inverseTransform.setInverse(self)

	def map(self, coordinates: CoordinatePos) -> CoordinatePos:
		coordinates.assertInFrame(self.initialFrame)
		return CoordinatePos(self.finalFrame, np.dot(self._jacobian, coordinates.coords) + self._translation)

	def reverseMap(self, coordinates: CoordinatePos) -> CoordinatePos:
		coordinates.assertInFrame(self.finalFrame)
		return CoordinatePos(self.initialFrame, np.dot(self._invJac, coordinates.coords) + self._invTrans)

	def jacobian(self, coordinates: CoordinatePos) -> FramedTensor:
		coordinates.assertInFrame(self.initialFrame)
		return FramedTensor(self._jacobian, coordinates, 1)

	def inverseJacobian(self, coordinates: CoordinatePos) -> FramedTensor:
		coordinates.assertInFrame(self.initialFrame)
		return FramedTensor(self._invJac, coordinates, 1)

	def gradJacobian(self, coordinates: CoordinatePos) -> FramedTensor:
		coordinates.assertInFrame(self.initialFrame)
		return FramedTensor(np.zeros((self.dim, self.dim, self.dim)), coordinates, 1)

	def compose(self, other: Union[Transformation, 'TranslationLinearTransformation']):
		if type(other) is not TranslationLinearTransformation:
			return super().compose(other)
		assert(self.initialFrame is other.finalFrame), 'incompatable composition'
		return TranslationLinearTransformation(
			other.initialFrame,
			self.finalFrame,
			np.dot(self._jacobian, other._jacobian),
			np.dot(self._jacobian, other._translation) + self._translation)

def compose(*transformations: Transformation):
	output = transformations[0]
	for transformation in transformations[1:]:
		output = output.compose(transformation)
	return output

class Rectangle:
	def __init__(self, point1: np.ndarray, point2: np.ndarray):
		self.lowerPoint: np.ndarray = np.array([min(a, b) for a, b in zip(point1, point2)])
		self.upperPoint: np.ndarray = np.array([max(a, b) for a, b in zip(point1, point2)])
		self.dimensions: np.ndarray = self.upperPoint - self.lowerPoint
		self.width: float = self.dimensions[0]
		self.height: float = self.dimensions[1]

def rectilinearTransform(
	initialFrame: CoordinateFrame,
	finalFrame: CoordinateFrame,
	initialRect: Rectangle,
	finalRect: Rectangle):

	jac = np.diag(finalRect.dimensions / initialRect.dimensions)
	translation = -np.dot(jac, initialRect.lowerPoint) + finalRect.lowerPoint

	return TranslationLinearTransformation(
		initialFrame, finalFrame, jac, translation)

def scalingTransform(
	initialFrame: CoordinateFrame,
	finalFrame: CoordinateFrame,
	scaling: Union[np.ndarray, List[float], float]):

	dim = initialFrame.dim

	if type(scaling) is float:
		scaling = dim * [scaling]
	scaling = np.array(scaling)
	return TranslationLinearTransformation(
		initialFrame,
		finalFrame,
		np.diag(scaling),
		np.zeros(dim))

def planarRotation(
	initialFrame: CoordinateFrame,
	finalFrame: CoordinateFrame,
	axis1: int,
	axis2: int,
	angle: float):

	jac = np.identity(initialFrame.dim)
	jac[axis1, axis1] = np.cos(angle)
	jac[axis2, axis2] = np.cos(angle)
	jac[axis1, axis2] =-np.sin(angle)
	jac[axis2, axis1] = np.sin(angle)

	return TranslationLinearTransformation(
		initialFrame, finalFrame, jac, np.zeros(initialFrame.dim))

def translation(
	initialFrame: CoordinateFrame,
	finalFrame: CoordinateFrame,
	translation: np.ndarray):

	return TranslationLinearTransformation(
		initialFrame, finalFrame, np.identity(initialFrame.dim), translation)

def originTranslation(
	initialFrame: CoordinateFrame,
	finalFrame: CoordinateFrame,
	translation: np.ndarray):

	return TranslationLinearTransformation(
		initialFrame, finalFrame, np.identity(initialFrame.dim), -translation)

def trapezoidalTransformation(
	initialFrame: CoordinateFrame,
	finalFrame: CoordinateFrame,
	scalingFactors: Union[np.ndarray, List[float], float]):
	"""
	maps unit (square)/(hypercube) to a (trapezoid)/(shape with two
	parallel hypersurfaces with some rectilinear scaling between them.
	"""
	
	dim = initialFrame.dim

	if type(scalingFactors) is float:
		scalingFactors = [scalingFactors] * (dim - 1)
	scalingFactors = np.array(scalingFactors)
	assert(scalingFactors.shape == (dim - 1,)), 'dim mismatch'
	
	scalingVector = np.array(list(scalingFactors) + [1]) - 1

	# TODO: add gradJac

	def scaleMultiplier(z: float):
		return (1 + scalingVector * z)

	def backwardMap(y: np.ndarray):
		return y / scaleMultiplier(y[-1])

	def forwardMap(x: np.ndarray):
		return x * scaleMultiplier(x[-1])

	def jacOfX(x: np.ndarray):
		output = np.diag(scaleMultiplier(x[-1]))
		output[:,-1] += scalingVector * x
		return output

	def invJacOfY(y: np.ndarray):
		scaling = scaleMultiplier(y[-1])
		output = np.diag(1. / scaling)
		output[:,-1] -= y * scalingVector / scaling**2
		return output

	def jacOfY(y: np.ndarray):
		return jacOfX(backwardMap(y))

	def invJacOfX(x: np.ndarray):
		return invJacOfY(forwardMap(x))

	forward = Transformation(initialFrame, finalFrame)
	forward.setMap(forwardMap)
	forward.setReverseMap(backwardMap)
	forward.setJacobian(jacOfX)
	forward.setInverseJacobian(invJacOfX)

	backward = Transformation(finalFrame, initialFrame)
	backward.setMap(backwardMap)
	backward.setReverseMap(forwardMap)
	backward.setJacobian(invJacOfY)
	backward.setInverseJacobian(jacOfY)
	
	forward.setInverse(backward)
	backward.setInverse(forward)

	return forward
