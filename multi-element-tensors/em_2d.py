import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import time
from typing import List, Tuple

import sys

import elements, frames, fields, grids, bundles

evoInstance = 0
globalHalfSideLength = 2.

def constructGrid_1(N: int):
	'''
	
	+-------------+
	|             |
	|             |
	|             |
	|             |
	|             |
	+-------------+
	
	'''
	globalFrame = frames.CoordinateFrame('global', 2)

	elementFrame = frames.CoordinateFrame('element', 2)
	globalToElement = frames.scalingTransform(
		globalFrame, elementFrame, 1.0 / globalHalfSideLength)
	elementGrid = grids.GridElement((N, N), elementFrame)

	grid = grids.Grid(
		[globalToElement],
		[elementGrid])

	return grid, globalFrame

def constructGrid_5(N: int):
	'''
	
	+-----------------------------+
	|\                           /|
	| \                         / |
	|  \                       /  |
	|   \                     /   |
	|    \                   /    |
	|     \                 /     |
	|      \               /      |
	|       +-------------+       |
	|       |             |       |
	|       |             |       |
	|       |             |       |
	|       |             |       |
	|       |             |       |
	|       +-------------+       |
	|      /               \      |
	|     /                 \     |
	|    /                   \    |
	|   /                     \   |
	|  /                       \  |
	| /                         \ |
	|/                           \|
	+-----------------------------+
	
	'''

	globalFrame = frames.CoordinateFrame('global', 2)
	centerFrame = frames.CoordinateFrame('center', 2)

	topFrame = frames.CoordinateFrame('top', 2)
	bottomFrame = frames.CoordinateFrame('bottom', 2)
	leftFrame = frames.CoordinateFrame('left', 2)
	rightFrame = frames.CoordinateFrame('right', 2)

	intermediateFrame = frames.CoordinateFrame('intermediate', 2)

	centerHalfSideLength = 0.45

	globalToCenter = frames.scalingTransform(
		centerFrame, globalFrame, centerHalfSideLength).inverse()

	originOffset = (globalHalfSideLength + centerHalfSideLength) / 2
	halfDepth = globalHalfSideLength - originOffset
	halfWidth = originOffset
	scaling = globalHalfSideLength / halfWidth

	globalToTop = frames.trapezoidalTransformation(
		topFrame, intermediateFrame, scaling).inverse().compose(frames.compose(
			frames.scalingTransform(
				intermediateFrame, intermediateFrame, np.array([halfWidth, halfDepth])).inverse(),
			frames.originTranslation(
				intermediateFrame, intermediateFrame, np.array([0, originOffset])),
			frames.planarRotation(
				globalFrame, intermediateFrame, 0, 1, 0)))

	globalToBottom = frames.trapezoidalTransformation(
		bottomFrame, intermediateFrame, scaling).inverse().compose(frames.compose(
			frames.scalingTransform(
				intermediateFrame, intermediateFrame, np.array([halfWidth, halfDepth])).inverse(),
			frames.originTranslation(
				intermediateFrame, intermediateFrame, np.array([0, originOffset])),
			frames.planarRotation(
				globalFrame, intermediateFrame, 0, 1, np.pi)))

	globalToRight = frames.trapezoidalTransformation(
		rightFrame, intermediateFrame, scaling).inverse().compose(frames.compose(
			frames.scalingTransform(
				intermediateFrame, intermediateFrame, np.array([halfWidth, halfDepth])).inverse(),
			frames.originTranslation(
				intermediateFrame, intermediateFrame, np.array([0, originOffset])),
			frames.planarRotation(
				globalFrame, intermediateFrame, 0, 1, np.pi/2)))

	globalToLeft = frames.trapezoidalTransformation(
		leftFrame, intermediateFrame, scaling).inverse().compose(frames.compose(
			frames.scalingTransform(
				intermediateFrame, intermediateFrame, np.array([halfWidth, halfDepth])).inverse(),
			frames.originTranslation(
				intermediateFrame, intermediateFrame, np.array([0, originOffset])),
			frames.planarRotation(
				globalFrame, intermediateFrame, 0, 1, -np.pi/2)))

	centerN = N
	transitionN = N

	centerGrid = grids.GridElement([centerN, centerN], centerFrame)
	topGrid = grids.GridElement([transitionN, centerN], topFrame)
	bottomGrid = grids.GridElement([transitionN, centerN], bottomFrame)
	leftGrid = grids.GridElement([transitionN, centerN], leftFrame)
	rightGrid = grids.GridElement([transitionN, centerN], rightFrame)

	grids.GridInterface(
		centerGrid.getBoundary(1, True),
		topGrid.getBoundary(1, False),
		'i<-i')
	grids.GridInterface(
		centerGrid.getBoundary(1, False),
		bottomGrid.getBoundary(1, False),
		'i<--i')# inverted coordinate diection
	grids.GridInterface(
		centerGrid.getBoundary(0, True),
		rightGrid.getBoundary(1, False),
		'i<--i')# inverted coordinate diection
	grids.GridInterface(
		centerGrid.getBoundary(0, False),
		leftGrid.getBoundary(1, False),
		'i<-i')

	grids.GridInterface(
		topGrid.getBoundary(0, True),
		rightGrid.getBoundary(0, False),
		'i<-i')
	grids.GridInterface(
		rightGrid.getBoundary(0, True),
		bottomGrid.getBoundary(0, False),
		'i<-i')
	grids.GridInterface(
		bottomGrid.getBoundary(0, True),
		leftGrid.getBoundary(0, False),
		'i<-i')
	grids.GridInterface(
		leftGrid.getBoundary(0, True),
		topGrid.getBoundary(0, False),
		'i<-i')

	grid = grids.Grid(
		[globalToCenter, globalToTop, globalToRight, globalToBottom, globalToLeft],
		[centerGrid, topGrid, rightGrid, bottomGrid, leftGrid])

	return grid, globalFrame

def constructGrid_9(sideN: int, innerN: int, outerN: int):
	'''
	
	+-----------------------------+
	|\                           /|
	| \                         / |
	|  \                       /  |
	|   \                     /   |
	|    \                   /    |
	|     \                 /     |
	|      \               /      |
	|       +-------------+       |
	|       |\           /|       |
	|       | \         / |       |
	|       |  +-------+  |       |
	|       |  |       |  |       |
	|       |  |       |  |       |
	|       |  |       |  |       |
	|       |  +-------+  |       |
	|       | /         \ |       |
	|       |/           \|       |
	|       +-------------+       |
	|      /               \      |
	|     /                 \     |
	|    /                   \    |
	|   /                     \   |
	|  /                       \  |
	| /                         \ |
	|/                           \|
	+-----------------------------+
	
	'''

	globalFrame = frames.CoordinateFrame('global', 2)
	centerFrame = frames.CoordinateFrame('center', 2)

	innerTopFrame = frames.CoordinateFrame('inner-top', 2)
	innerBottomFrame = frames.CoordinateFrame('inner-bottom', 2)
	innerLeftFrame = frames.CoordinateFrame('inner-left', 2)
	innerRightFrame = frames.CoordinateFrame('inner-right', 2)

	outerTopFrame = frames.CoordinateFrame('outer-top', 2)
	outerBottomFrame = frames.CoordinateFrame('outer-bottom', 2)
	outerLeftFrame = frames.CoordinateFrame('outer-left', 2)
	outerRightFrame = frames.CoordinateFrame('outer-right', 2)

	intermediateFrame = frames.CoordinateFrame('intermediate', 2)

	centerHalfSideLength = 0.25
	outerHalfSideLength = 0.6

	innerHalfDepth = (outerHalfSideLength - centerHalfSideLength) / 2
	outerHalfDepth = (globalHalfSideLength - outerHalfSideLength) / 2

	innerOffset = centerHalfSideLength + innerHalfDepth
	outerOffset = outerHalfSideLength + outerHalfDepth

	globalToCenter = frames.scalingTransform(
		centerFrame, globalFrame, centerHalfSideLength).inverse()

	def setupTransform(finalFrame, rotation, offset, halfDepth):
		scaling = (offset + halfDepth) / offset
		return frames.trapezoidalTransformation(
			finalFrame, intermediateFrame, scaling).inverse().compose(frames.compose(
				frames.scalingTransform(
					intermediateFrame, intermediateFrame, np.array([offset, halfDepth])).inverse(),
				frames.originTranslation(
					intermediateFrame, intermediateFrame, np.array([0, offset])),
				frames.planarRotation(
					globalFrame, intermediateFrame, 0, 1, rotation)))

	globalToInnerTop = setupTransform(      innerTopFrame,    0.,   innerOffset, innerHalfDepth)
	globalToInnerBottom = setupTransform(innerBottomFrame, np.pi,   innerOffset, innerHalfDepth)
	globalToInnerRight = setupTransform(  innerRightFrame, np.pi/2, innerOffset, innerHalfDepth)
	globalToInnerLeft = setupTransform(    innerLeftFrame,-np.pi/2, innerOffset, innerHalfDepth)

	globalToOuterTop = setupTransform(      outerTopFrame,    0.,   outerOffset, outerHalfDepth)
	globalToOuterBottom = setupTransform(outerBottomFrame, np.pi,   outerOffset, outerHalfDepth)
	globalToOuterRight = setupTransform(  outerRightFrame, np.pi/2, outerOffset, outerHalfDepth)
	globalToOuterLeft = setupTransform(    outerLeftFrame,-np.pi/2, outerOffset, outerHalfDepth)

	centerGrid = grids.GridElement([sideN, sideN], centerFrame)

	innerTopGrid = grids.GridElement(   [sideN, innerN], innerTopFrame)
	innerBottomGrid = grids.GridElement([sideN, innerN], innerBottomFrame)
	innerRightGrid = grids.GridElement( [sideN, innerN], innerRightFrame)
	innerLeftGrid = grids.GridElement(  [sideN, innerN], innerLeftFrame)

	outerTopGrid = grids.GridElement(   [sideN, outerN], outerTopFrame)
	outerBottomGrid = grids.GridElement([sideN, outerN], outerBottomFrame)
	outerRightGrid = grids.GridElement( [sideN, outerN], outerRightFrame)
	outerLeftGrid = grids.GridElement(  [sideN, outerN], outerLeftFrame)

	grids.GridInterface(
		centerGrid.getBoundary(1, True),
		innerTopGrid.getBoundary(1, False),
		'i<-i')
	grids.GridInterface(
		centerGrid.getBoundary(1, False),
		innerBottomGrid.getBoundary(1, False),
		'i<--i')# inverted coordinate diection
	grids.GridInterface(
		centerGrid.getBoundary(0, True),
		innerRightGrid.getBoundary(1, False),
		'i<--i')# inverted coordinate diection
	grids.GridInterface(
		centerGrid.getBoundary(0, False),
		innerLeftGrid.getBoundary(1, False),
		'i<-i')

	def innerOuterInterface(inner: grids.GridElement, outer: grids.GridElement):
		grids.GridInterface(
			inner.getBoundary(1, True),
			outer.getBoundary(1, False),
			'i<-i')

	innerOuterInterface(innerTopGrid,    outerTopGrid)
	innerOuterInterface(innerBottomGrid, outerBottomGrid)
	innerOuterInterface(innerRightGrid,  outerRightGrid)
	innerOuterInterface(innerLeftGrid,   outerLeftGrid)

	def clockwiseInterface(left: grids.GridElement, right: grids.GridElement):
		grids.GridInterface(
			left.getBoundary(0, True),
			right.getBoundary(0, False),
			'i<-i')

	clockwiseInterface(innerTopGrid,    innerRightGrid)
	clockwiseInterface(innerRightGrid,  innerBottomGrid)
	clockwiseInterface(innerBottomGrid, innerLeftGrid)
	clockwiseInterface(innerLeftGrid,   innerTopGrid)

	clockwiseInterface(outerTopGrid,    outerRightGrid)
	clockwiseInterface(outerRightGrid,  outerBottomGrid)
	clockwiseInterface(outerBottomGrid, outerLeftGrid)
	clockwiseInterface(outerLeftGrid,   outerTopGrid)

	grid = grids.Grid(
		[
			globalToCenter,
			globalToInnerTop,
			globalToInnerRight,
			globalToInnerBottom,
			globalToInnerLeft,
			globalToOuterTop,
			globalToOuterRight,
			globalToOuterBottom,
			globalToOuterLeft],
		[
			centerGrid,
			innerTopGrid,
			innerRightGrid,
			innerBottomGrid,
			innerLeftGrid,
			outerTopGrid,
			outerRightGrid,
			outerBottomGrid,
			outerLeftGrid])

	return grid, globalFrame

def performSimulation(grid: grids.Grid, timeInterval: Tuple[float, float]):

	###############################################################
	##################        metric data        ##################
	###############################################################

	def identityFunction(x: frames.CoordinatePos):
		x.assertInFrame(grid.globalFrame)
		return frames.FramedTensor(np.identity(2), x, 1)

	def leviCivitaFunction(x: frames.CoordinatePos):
		x.assertInFrame(grid.globalFrame)
		return frames.FramedTensor(np.array([[
			i - j for i in range(2)] for j in range(2)]), x, 0)

	def metricFunction(x: frames.CoordinatePos):
		x.assertInFrame(grid.globalFrame)
		return frames.FramedTensor(np.identity(2), x, 0)

	def inverseMetricFunction(x: frames.CoordinatePos):
		x.assertInFrame(grid.globalFrame)
		return frames.FramedTensor(np.identity(2), x, 2)

	kroneckerDelta = fields.TensorField.functionInit(grid, identityFunction)
	leviCivita = fields.TensorField.functionInit(grid, leviCivitaFunction)

	metric = fields.TensorField.functionInit(grid, metricFunction)
	gradMetric = metric.commaGrad()
	invMetric = fields.TensorField.functionInit(grid, inverseMetricFunction)

	christoffel_lower = fields.TensorField.defaultInit(grid, gradMetric.tensorShape)
	christoffel_lower.addData('smn<-mns', -gradMetric)
	christoffel_lower.addData('smn<-msn', gradMetric)
	christoffel_lower.addData('smn<-snm', gradMetric)
	christoffel = fields.TensorField.tensorProduct(
		'smn,as->amn', christoffel_lower, invMetric)

	###############################################################
	##################         init data         ##################
	###############################################################

	def bump(x: np.ndarray, x0: np.ndarray, width: float) -> float:
		r = (x - x0) / width * np.pi / 2
		r = np.sqrt(np.dot(r, r))
		return np.cos(r)**2 if r < (np.pi / 2) else 0.

	def chargeDensity(x: frames.CoordinatePos, t: float):
		x.assertInFrame(grid.globalFrame)
		width = 0.3
		amplitude = 0.4
		amplitude /= width**2
		sourceLocation = np.array([0., 0.])
		# return frames.FramedTensor(
		# 	np.dot(x.coords, x.coords) * np.cos(np.arctan2(x.coords[1], x.coords[0])),
		# 	x, 0)
		return frames.FramedTensor(
			bump(x.coords, sourceLocation, width) * amplitude,
			x, 0)
	
	def chargeField(t: float):
		# return fields.TensorField.defaultInit(grid, ())
		return fields.TensorField.functionInit(grid, lambda x: chargeDensity(x, t))


	def initialE(x: frames.CoordinatePos):
		# mask = min((1 - abs(x[0])) * (1 - abs(x[1])), 0.1) * 10
		# r = np.sqrt(x[0]**2 + x[1]**2)
		# return A*r**2 * np.exp(-a * (r - b)) * mask
		return frames.FramedTensor(np.array([0., 0.]), x, 0)
	def initialA(x: frames.CoordinatePos):
		# mask = min((1 - abs(x[0])) * (1 - abs(x[1])), 0.1) * 10
		# r = np.sqrt(x[0]**2 + x[1]**2)
		# return -(2 * A * r * np.exp(-a * (r - b)) * mask - a * A * r**2 * np.exp(-a * (r - b)) * mask)
		return frames.FramedTensor(np.array([0., 0.]), x, 0)

	initE = fields.TensorField.functionInit(grid, initialE)
	initA = fields.TensorField.functionInit(grid, initialA)
	initGamma = fields.trueGradient(initA, christoffel, 1)
	initTrace = fields.TensorField.tensorProduct('ii->', initGamma)
	initGamma = fields.TensorField.tensorProduct('ij,jk->ik', initGamma, invMetric)

	initState = bundles.StateBundle(
		[kroneckerDelta, metric, invMetric, christoffel,
			initE, initTrace, initGamma, initA],
		[0,1,2,3])

	initStateVector = initState.canonicalPack()

	currentState = initState

	###############################################################
	##################      state evolution      ##################
	###############################################################

	def constructBoundarySurfaceGeometryData(
		boundary: bundles.BoundaryBundle):

		gridBoundary = boundary.gridBoundary
		kroneckerElement = boundary.tensorElements[0]
		invMetricElement = boundary.tensorElements[2]

		shape = gridBoundary.shape
		dim = len(shape)
		orthoVector = np.array(
			[0] * gridBoundary.axis + [1] +
			[0] * (dim - gridBoundary.axis))
		orthoVector *= 1 if gridBoundary.isRightBoundary else -1
		def orthoInitFunct(x: np.ndarray):
			return orthoVector
		
		orthoVectorField = elements.TensorFieldElement.functionInit(
			[count - 1 for count in shape], orthoInitFunct)
		normalField = orthoVectorField /\
			elements.TensorFieldElement.tensorProduct(
				'i,j,ij->', orthoVectorField, orthoVectorField, invMetricElement).sqrt_scalar()
		projectionField = kroneckerElement - elements.TensorFieldElement.tensorProduct(
			'i,j,ik->kj', normalField, normalField, invMetricElement)

		return normalField, projectionField

	def findOutflowVectorField(
		boundary: bundles.BoundaryBundle,
		normalField: elements.TensorFieldElement):

		invMetric = boundary.tensorElements[2]
		eField = boundary.tensorElements[4]
		traceField = boundary.tensorElements[5]
		gammaField = boundary.tensorElements[6]

		normalVector = elements.TensorFieldElement.tensorProduct(
			"i,ij->j", normalField, invMetric)

		correctedGammaField = gammaField.copy()
		correctedGammaField.addData(elements.TensorFieldElement.tensorProduct(
			"i,j,->ij", normalVector, normalVector,-traceField))

		gammaProjection = elements.TensorFieldElement.tensorProduct(
			'ij,j->i', correctedGammaField, normalField)
		#up index
		return eField + gammaProjection

	def setFieldQuantities(
		boundary: bundles.BoundaryBundle,
		normalField: elements.TensorFieldElement,
		projectionField: elements.TensorFieldElement,
		outflow: elements.TensorFieldElement,
		inflow: elements.TensorFieldElement = None):

		invMetricField = boundary.tensorElements[2]
		eField = boundary.tensorElements[4]
		traceField = boundary.tensorElements[5]
		gammaField = boundary.tensorElements[6]

		normalVector = elements.TensorFieldElement.tensorProduct(
			"i,ij->j", normalField, invMetricField)

		eField.setData(outflow / 2)
		# gammaField.setData(
		# 	elements.TensorFieldElement.tensorProduct('ij,kj->ik',
		# 		gammaField, projectionField) +
		# 	elements.TensorFieldElement.tensorProduct('i,j,jk->ik',
		# 		outflow / 2, normalField, invMetricField))
		# if inflow is not None:
		# 	eField.addData(inflow / 2)
		# 	gammaField.addData(
		# 		elements.TensorFieldElement.tensorProduct('i,j,jk->ik',
		# 			-inflow / 2, normalField, invMetricField))

		newProjection = elements.TensorFieldElement.tensorProduct('i,j,jk->ik',
			outflow / 2, normalField, invMetricField)
		if inflow is not None:
			eField.addData(inflow / 2)
			newProjection.addData(
				elements.TensorFieldElement.tensorProduct('i,j,jk->ik',
					-inflow / 2, normalField, invMetricField))
		gammaField.setData(
			elements.TensorFieldElement.tensorProduct('ij,kj->ik',
				gammaField, projectionField) + newProjection +
			elements.TensorFieldElement.tensorProduct('i,j,->ij',
				normalVector, normalVector, traceField))

	def handleBoundary(
		boundaryBundle: bundles.BoundaryBundle):

		normal, projection = constructBoundarySurfaceGeometryData(boundaryBundle)
		outflow = findOutflowVectorField(boundaryBundle, normal)
		setFieldQuantities(boundaryBundle, normal, projection, outflow)

	def handleInterface(
		interfaceBundle: bundles.InterfaceBundle):

		leftBoundary = bundles.BoundaryBundle(
			interfaceBundle.gridInterface.leftGridBoundary,
			interfaceBundle.leftTensorElements)

		rightBoundary = bundles.BoundaryBundle(
			interfaceBundle.gridInterface.rightGridBoundary,
			interfaceBundle.rightTensorElements)

		leftNormal, leftProjection = constructBoundarySurfaceGeometryData(leftBoundary)
		rightNormal, rightProjection = constructBoundarySurfaceGeometryData(rightBoundary)

		leftOutflow = findOutflowVectorField(leftBoundary, leftNormal)
		rightOutflow = findOutflowVectorField(rightBoundary, rightNormal)

		leftInflow = elements.TensorFieldElement.tensorProduct('i,ji->j',
			rightOutflow, interfaceBundle.gridInterface.rightToLeftJacobianField)
		leftInflow.setData(leftInflow.copy(),
			gridSettingRule=interfaceBundle.gridInterface.rightToLeftSetterRule)
		
		rightInflow = elements.TensorFieldElement.tensorProduct('i,ji->j',
			leftOutflow, interfaceBundle.gridInterface.leftToRightJacobianField)
		rightInflow.setData(rightInflow.copy(),
			gridSettingRule=interfaceBundle.gridInterface.leftToRightSetterRule)

		setFieldQuantities(
			leftBoundary, leftNormal, leftProjection, leftOutflow, leftInflow)
		setFieldQuantities(
			rightBoundary, rightNormal, rightProjection, rightOutflow, rightInflow)

	def stateDot(time, stateVector):
		currentState.canonicalUnpack(stateVector)

		global evoInstance

		for interface in currentState.interfaces:
			handleInterface(interface)
		for boundary in currentState.boundaries:
			handleBoundary(boundary)

		stateVector[:] = currentState.canonicalPack()

		# kroneckerField = currentState.tensorFields[0]
		# metricField = currentState.tensorFields[1]
		invMetricField = currentState.tensorFields[2]
		christoffelField = currentState.tensorFields[3]

		eField = currentState.tensorFields[4]
		traceField = currentState.tensorFields[5]
		gammaField = currentState.tensorFields[6]
		aField = currentState.tensorFields[7]
		# print(evoInstance, time)
		# print(eField.elements[0].getBoundaryElement(0, False).data)
		# print(eField.elements[4].getBoundaryElement(1, False).data)
		# print()
		# if evoInstance > 400 and evoInstance < 420:
		evoInstance += 1

		chargeDensityField = chargeField(time)

		#up, up
		gradE = fields.TensorField.tensorProduct('ij,jk->ik',
			fields.trueGradient(eField, christoffelField, 1),
			invMetricField)
		#up, up, down
		gradGamma = fields.trueGradient(gammaField, christoffelField, 2)
		#up
		gradTrace = fields.TensorField.tensorProduct('i,ij->j',
			fields.trueGradient(traceField, christoffelField, 0),
			invMetricField)
		#up, down
		# gradA = fields.trueGradient(aField, christoffelField, 1)

		aDot = -eField # + gradPhi

		eDot = gradTrace - fields.TensorField.tensorProduct(
			'jii->j', gradGamma) # - currentDensity

		gammaDot = -gradE # + gradGradPhi
		traceDot = -chargeDensityField # + lapPhi

		stateDot = bundles.StateBundle(
			[eDot, traceDot, gammaDot, aDot])
		
		currentState.canonicalUnpack(stateDot.canonicalPack())
		for boundary in currentState.boundaries:
			handleBoundary(boundary)
		for interface in currentState.interfaces:
			handleInterface(interface)
		# stateDot.canonicalUnpack(currentState.canonicalPack())
			
		return currentState.canonicalPack()

	t1 = time.time()

	solutionSet = scipy.integrate.solve_ivp(
		stateDot, timeInterval, initStateVector, dense_output=True)

	t2 = time.time()

	return solutionSet, t2 - t1, currentState

def makeMesh(grid: grids.Grid, xData: np.ndarray, yData: np.ndarray):
	meshX, meshY = np.meshgrid(xData, yData)
	meshPoints = np.array([meshX, meshY])
	meshPoints = np.reshape(meshPoints, (2, np.prod(meshX.shape)))
	meshPoints = np.transpose(meshPoints)
	meshPoints = [frames.CoordinatePos(grid.globalFrame, point) for point in meshPoints]
	return meshX, meshY, grids.FieldPrecompute(grid, meshPoints)

def em_2d(
	N = 8,
	animationDuration = 2.,
	simDuration = 3.0,
	dt = 0.1,
	display_dx = 0.1):

	tData = np.arange(0, simDuration, dt)

	# grid, globalFrame = constructGrid_1(N)
	grid, globalFrame = constructGrid_5(N)
	
	solutionSet, elapsedTime, stateBundle = performSimulation(grid, (tData[0], tData[-1]))

	print("simulation completed in " + str(elapsedTime) + " seconds")

	###############################################################
	##################   solution visualization   #################
	###############################################################

	t1 = time.time()

	outputDataSet = np.zeros((len(tData), len(solutionSet.y)))
	# chargeFieldSeries: List[fields.TensorField] = []

	for i in range(len(tData)):
		outputDataSet[i] = solutionSet.sol(tData[i])
		# chargeFieldSeries.append(chargeField(tData[i]))

	t2 = time.time()
	print("solution values interpolated in " + str(t2 - t1) + " seconds")

	solver_tData = solutionSet.t
	dtList = []
	for i in range(1, len(solver_tData)):
		dtList.append(solver_tData[i] - solver_tData[i - 1])
	print("average deltaT: " + str(sum(dtList) / len(dtList)))

	t1 = t2

	xData = np.arange(-globalHalfSideLength, globalHalfSideLength + display_dx / 2, display_dx)
	yData = xData.copy()
	meshX, meshY, fieldPrecompute = makeMesh(grid, xData, yData)
	# meshX, meshY = np.meshgrid(xData, yData)
	# meshPoints = np.array([meshX, meshY])
	# meshPoints = np.reshape(meshPoints, (2, np.prod(meshX.shape)))
	# meshPoints = np.transpose(meshPoints)
	# meshPoints = [frames.CoordinatePos(globalFrame, point) for point in meshPoints]
	# fieldPrecompute = grids.FieldPrecompute(grid, meshPoints)

	magEDataSet = []
	magESet: List[fields.TensorField] = []
	# constraintDataSet = []

	t2 = time.time()
	print("precompute evaluated in " + str(t2 - t1) + " seconds")
	t1 = t2
	
	for outputData in outputDataSet:
		stateBundle.canonicalUnpack(outputData)
		metricField = stateBundle.tensorFields[1]
		christoffel = stateBundle.tensorFields[3]
		eField = stateBundle.tensorFields[4]

		magE = fields.TensorField.tensorProduct('i,j,ij->', eField, eField, metricField)
		magE = fields.TensorField.sqrt_scalar(magE)
		magESet.append(magE)
		magEDataSet.append(np.reshape(magE.precomputeEval(fieldPrecompute), meshX.shape))

	# for chargeFieldSlice in chargeFieldSeries:
	# 	magEDataSet.append(np.reshape(chargeFieldSlice.precomputeEval(fieldPrecompute), meshX.shape))

	t2 = time.time()
	print("data evaluated in " + str(t2 - t1) + " seconds")
	# print(magEDataSet[-1])
	t1 = t2

	# minVal = -1.
	maxVal = 0.3
	# maxVal = 4.0
	minVal = 0.0
	# maxVal = 0.5

	fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	ax = fig.add_subplot(111,
		projection='3d',
		autoscale_on=False,
		xlim=(-globalHalfSideLength, globalHalfSideLength),
		ylim=(-globalHalfSideLength, globalHalfSideLength),
		zlim=(minVal-0.01, maxVal+0.01))
	# plot = ax.plot_surface(meshX,meshY,constraintDataSet[0])
	
	# startFrame = 11 * len(tData) // 12
	# startFrame = 5 * len(tData) // 6
	startFrame = 0
	frameCount = len(tData) - startFrame
	plot = ax.plot_surface(meshX,meshY,magEDataSet[startFrame])

	def animate(frame):
		ax.collections.clear()
		# plot = ax.plot_surface(meshX,meshY,constraintDataSet[frame], color='blue')
		plot = ax.plot_surface(meshX,meshY,magEDataSet[frame + startFrame], color='blue')

	ani = animation.FuncAnimation(
		fig, animate, frameCount,
		interval = animationDuration * 1000 / len(tData))

	input()

	plt.show()

if __name__ == '__main__':
	# em_2d(
	# 	animationDuration=6.,
	# 	N=8,
	# 	simDuration=0.03,
	# 	display_dx=0.05,
	# 	dt = 0.0001)
	# em_2d(
	# 	animationDuration=2.,
	# 	N=8,
	# 	simDuration=2.0,
	# 	display_dx=0.1)

	em_2d(
		animationDuration=8.,
		N=16,
		simDuration=6.0,
		dt = 0.03,
		display_dx=0.03)

	# em_2d(
	# 	N = 48,
	# 	animationDuration = 8.,
	# 	simDuration = 6.0,
	# 	dt = 0.03,
	# 	display_dx = 0.03)

	# em_2d(
	# 	animationDuration=8.,
	# 	N=32,
	# 	simDuration=4.,
	# 	display_dx=0.1)
