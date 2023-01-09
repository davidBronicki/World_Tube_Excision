import numpy as np

import elements, frames

from typing import List, Tuple, Union
from functools import reduce

reachCount = 0
def reached():
	global reachCount
	print('reached:', reachCount)
	reachCount += 1

class GridElement:
	def __init__(self, chebOrders: Tuple[int], frame: frames.CoordinateFrame):
		assert(frame.dim == len(chebOrders)), 'grid shape must agree with frame dim'
		self.chebOrders = chebOrders
		self.shape = tuple([order + 1 for order in chebOrders])
		self.frame = frame
		self.spacialDimensions = len(chebOrders)
		self.boundaries: List[GridBoundary] = []
		self.index = -1 # set in grid instantiation
		for axis in range(self.spacialDimensions):
			self.boundaries.append(GridBoundary(self, axis, True))
			self.boundaries.append(GridBoundary(self, axis, False))

	def getBoundary(self, axis: int, rightSide: bool):
		return self.boundaries[2 * axis + (0 if rightSide else 1)]

class GridBoundary:
	def __init__(self, baseElement: GridElement, axis: int, isRightBoundary: bool):
		self.baseElement = baseElement
		self.shape = tuple(baseElement.shape[:axis] + baseElement.shape[axis + 1:])
		self.chebOrders = tuple(baseElement.chebOrders[:axis] + baseElement.chebOrders[axis + 1:])
		self.axis = axis
		self.isRightBoundary = isRightBoundary
		self.index = -1 # set in grid instantiation
		self.interface: Union[GridInterface, None] = None

class GridInterface:
	def __init__(self,
		leftGridBoundary: GridBoundary,
		rightGridBoundary: GridBoundary,
		leftToRightSetterRule = ''):
		
		leftGridBoundary.interface = self
		self.leftGridBoundary = leftGridBoundary

		rightGridBoundary.interface = self
		self.rightGridBoundary = rightGridBoundary

		self.leftToRightSetterRule = leftToRightSetterRule
		if leftToRightSetterRule == '':
			self.rightToLeftSetterRule = ''
		else:
			right, left = leftToRightSetterRule.split('<-')
			self.rightToLeftSetterRule = '<-'.join((left, right))

		self.index = -1 # set in grid instantiation
		self.leftToRightTransform: Union[frames.Transformation, None] = None
		self.leftToRightJacobianField: Union[elements.TensorFieldElement, None] = None
		self.leftToRightInvJacobianField: Union[elements.TensorFieldElement, None] = None
		self.rightToLeftTransform: Union[frames.Transformation, None] = None
		self.rightToLeftJacobianField: Union[elements.TensorFieldElement, None] = None
		self.rightToLeftInvJacobianField: Union[elements.TensorFieldElement, None] = None

class Grid:
	def __init__(self,
		globalToElementTransforms: List[frames.Transformation],
		gridElements: List[GridElement]):

		assert(len(gridElements) == len(globalToElementTransforms)), 'each element must have a coordinate frame'
		self.globalToElementTransforms = globalToElementTransforms
		self.globalFrame = globalToElementTransforms[0].initialFrame
		self.elementFrames: List[frames.CoordinateFrame] = []
		self.elementToGlobalTransforms: List[frames.Transformation] = []
		for t in globalToElementTransforms:
			assert(t.initialFrame is self.globalFrame), 'only one global frame allowed'
			self.elementFrames.append(t.finalFrame)
			self.elementToGlobalTransforms.append(t.inverse())

		self.gridElements = gridElements
		for i in range(len(gridElements)):
			self.gridElements[i].index = i
		self.boundaries: List[GridBoundary] = []
		self.interfaces: List[GridInterface] = []

		for element in gridElements:
			for boundary in element.boundaries:
				if boundary.interface is None:
					self.boundaries.append(boundary)
				else:
					self._gridLevelInterfaceInit(boundary.interface)

	def _gridLevelInterfaceInit(self, interface: GridInterface):
		if interface.index != -1: return

		interface.index = len(self.interfaces)
		self.interfaces.append(interface)
		
		leftBoundary = interface.leftGridBoundary
		rightBoundary = interface.rightGridBoundary
		leftFrame = leftBoundary.baseElement.frame
		rightFrame = rightBoundary.baseElement.frame
		leftIndex = self.elementFrames.index(leftFrame)
		rightIndex = self.elementFrames.index(rightFrame)

		def leftBoundaryParameterizer(x: np.ndarray):
			output = np.zeros(x.shape[0] + 1)
			output[:leftBoundary.axis] = x[:leftBoundary.axis]
			output[leftBoundary.axis + 1:] = x[leftBoundary.axis:]
			output[leftBoundary.axis] = 1. if leftBoundary.isRightBoundary else -1.
			return output
		def rightBoundaryParameterizer(x: np.ndarray):
			output = np.zeros(x.shape[0] + 1)
			output[:rightBoundary.axis] = x[:rightBoundary.axis]
			output[rightBoundary.axis + 1:] = x[rightBoundary.axis:]
			output[rightBoundary.axis] = 1. if rightBoundary.isRightBoundary else -1.
			return output

		interface.leftToRightTransform =\
			self.globalToElementTransforms[rightIndex].compose(
				self.elementToGlobalTransforms[leftIndex])
		interface.leftToRightJacobianField =\
			elements.TensorFieldElement.functionInit(leftBoundary.chebOrders,
				lambda x: interface.leftToRightTransform.jacobian(
					frames.CoordinatePos(leftFrame, leftBoundaryParameterizer(x))).data)
		interface.leftToRightInvJacobianField =\
			elements.TensorFieldElement.functionInit(leftBoundary.chebOrders,
				lambda x: interface.leftToRightTransform.inverseJacobian(
					frames.CoordinatePos(leftFrame, leftBoundaryParameterizer(x))).data)
		interface.rightToLeftTransform =\
			self.globalToElementTransforms[leftIndex].compose(
				self.elementToGlobalTransforms[rightIndex])
		interface.rightToLeftJacobianField =\
			elements.TensorFieldElement.functionInit(rightBoundary.chebOrders,
				lambda x: interface.rightToLeftTransform.jacobian(
					frames.CoordinatePos(rightFrame, rightBoundaryParameterizer(x))).data)
		interface.rightToLeftInvJacobianField =\
			elements.TensorFieldElement.functionInit(rightBoundary.chebOrders,
				lambda x: interface.rightToLeftTransform.inverseJacobian(
					frames.CoordinatePos(rightFrame, rightBoundaryParameterizer(x))).data)

	def locateDomain(self, location: frames.CoordinatePos, epsilon = 1e-10):
		cutoff = 1 + epsilon
		for i, transform in enumerate(self.globalToElementTransforms):
			coordinatePos = transform.map(location)
			found = True
			for x in coordinatePos.coords:
				if x > cutoff or x < -cutoff:
					found = False
					continue
			if found:
				return i
		return -1

################################################################
#####################   precompute logic   #####################
################################################################

class ChebEngine:
	def __init__(self, order: int):
		self.order = order
		self.N = order + 1
		self.collocationPoints = []
		for i in range(0, order + 1):
			self.collocationPoints.append(np.cos(np.pi * i / order))
		
		self.polynomialCoefs = [[1],[1]]
		for n in range(2, order + 1):
			self.polynomialCoefs.append([])
			if n%2 == 0:
				self.polynomialCoefs[n].append(-self.polynomialCoefs[n - 2][0])
				for i in range(1, n // 2):
					self.polynomialCoefs[n].append(
						2 * self.polynomialCoefs[n - 1][i - 1] - self.polynomialCoefs[n - 2][i])
				self.polynomialCoefs[n].append(2 * self.polynomialCoefs[n - 1][n // 2 - 1])
			else:
				for i in range(0, n // 2):
					self.polynomialCoefs[n].append(
						2 * self.polynomialCoefs[n - 1][i] - self.polynomialCoefs[n - 2][i])
				self.polynomialCoefs[n].append(2 * self.polynomialCoefs[n - 1][n // 2])

	def getCollocationPoint(self, n: int) -> float:
		return self.collocationPoints[n]

	def eval(self, n: int, x: float) -> float:
		monomialValue = 1 if n % 2 == 0 else x
		xSquared = x * x
		output = monomialValue * self.polynomialCoefs[n][0]
		for i in range(1, (n + 2) // 2):
			monomialValue *= xSquared
			output += monomialValue * self.polynomialCoefs[n][i]
		return output

class PrecomputePoint:
	"""
	contains the coefficients to quickly convert spectral
	coefficients into a function value at a given point
	"""
	def __init__(self, coefs: np.ndarray):
		self.coefs = coefs
	
	def __call__(self, spectralCoefs: np.ndarray) -> Union[np.ndarray, float]:
		return np.tensordot(self.coefs, spectralCoefs, 2*[list(range(len(self.coefs.shape)))])

class ElementPrecompute:
	def __init__(self, gridElement: 'GridElement', precomputePoints: List[np.ndarray]):
		self.coefficientCount = np.prod(gridElement.shape)
		self.chebEngines = [ChebEngine(N) for N in gridElement.chebOrders]

		self.points = precomputePoints
		self.precomputes = [self._makePrecompute(point) for point in precomputePoints]

	def __call__(self, n: int, spectralCoefs: np.ndarray):
		return self.precomputes[n](spectralCoefs)

	def _makePrecompute(self, point: np.ndarray):
		coefVectors: List[np.ndarray] = []
		for engine, coord in zip(self.chebEngines, point):
			coefVectors.append(np.array([
				engine.eval(n, coord) for n in range(engine.N)]))
		coefs = reduce(np.multiply, np.ix_(*coefVectors))
		return PrecomputePoint(coefs)

	def appendPoint(self, precomputePoint: np.ndarray):
		self.points.append(precomputePoint)
		self.precomputes.append(self._makePrecompute(precomputePoint))
		# add point

	def indexLocation(self, n):
		return self.points[n]

class FieldPrecomputePoint:
	"""
	contains info to quickly access the correct ElementPrecompute object
	and the correct index within this element.
	"""
	def __init__(self, elementID: int, precomputeIndex: int, location: frames.CoordinatePos):
		self.elementID = elementID
		self.precomputeIndex = precomputeIndex
		self.location = location

class FieldPrecompute:
	def __init__(self,
		grid: 'Grid',
		precomputePoints: List[frames.CoordinatePos],
		epsilon = 1e-10):

		self.precomputeElements: List[ElementPrecompute] = []
		self.precomputePointsMetaData: List[FieldPrecomputePoint] = []
		for gridElement in grid.gridElements:
			self.precomputeElements.append(ElementPrecompute(gridElement, []))
		for point in precomputePoints:
			elementID = grid.locateDomain(point, epsilon)
			self.precomputeElements[elementID].appendPoint(
				grid.globalToElementTransforms[elementID].map(point).coords)
			self.precomputePointsMetaData.append(FieldPrecomputePoint(
				elementID,
				len(self.precomputeElements[elementID].precomputes) - 1,
				point))

	def __call__(self, n: int, spectralCoefs: np.ndarray):
		return self.precomputeElements[self.precomputePointsMetaData[n].elementID](
			self.precomputePointsMetaData[n].precomputeIndex, spectralCoefs)

	def __len__(self):
		return len(self.precomputePointsMetaData)

	def indexLocation(self, n: int):
		return self.precomputePointsMetaData[n].location

	def indexElementID(self, n: int):
		return self.precomputePointsMetaData[n].elementID
