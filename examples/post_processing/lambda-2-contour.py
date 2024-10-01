# Script for creating a lambda-2 contour for LES turbulence visualization.
# Also calculates the vorticity field for coloring the isosurface.

# For details on the lambda-2 criterion, see pp. 76-77 in:
# J. Jeong and F. Hussain, “On the identification of a vortex,”
#  Journal of Fluid Mechanics, vol. 285, pp. 69–94, 1995,
#  doi: 10.1017/S0022112095000462.

# Script generated using ParaView version 5.11.0

################################################################################

# USAGE:
# 1. Open a ParaView instance and load desired solution file
# 2. Import this script to Paraview by going to Macros -> Import new macro...
#      and then selecting this script in the browser.
# 3. Run the macro by going to Macros -> lambda-2-contour
# 4. You can then view a contour of the lambda-2 parameter in the pipeline
#      browser, and optionally color by the vorticity magnitude.

################################################################################

# Import the simple module from ParaView
from paraview.simple import *
# Disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# Find source file
solutionFile = FindSource('*.exo')

# Merge blocks to allow manipulation of .exo data
mergeBlocks1 = MergeBlocks(registrationName='MergeBlocks1', Input=solutionFile)

UpdatePipeline(time=100.0, proxy=mergeBlocks1)

# Merge velocity components into vector
mergeVectorComponents1 = MergeVectorComponents(
    registrationName='MergeVectorComponents1', Input=mergeBlocks1)
mergeVectorComponents1.XArray = 'velocity_0'
mergeVectorComponents1.YArray = 'velocity_1'
mergeVectorComponents1.ZArray = 'velocity_2'
mergeVectorComponents1.OutputVectorName = 'velocity'

UpdatePipeline(time=100.0, proxy=mergeVectorComponents1)

# Use a programmable filter to calculate lambda2 criterion
programmableFilter1 = ProgrammableFilter(
    registrationName='ProgrammableFilter1', Input=mergeVectorComponents1)
programmableFilter1.Script = """import numpy as np

vvector = inputs[0].PointData['velocity']

vstrain = strain(vvector)
vskew = gradient(vvector) - vstrain

aaa = matmul(vstrain, vstrain) + matmul(vskew, vskew)

lambdas  = np.linalg.eigvals(aaa)
lambdas  = np.real(lambdas )
lambda2 = sort(lambdas)[:,1]

output.DeepCopy(inputs[0].VTKObject)
output.PointData.append(lambda2, 'lambda2')"""
programmableFilter1.RequestInformationScript = ''
programmableFilter1.RequestUpdateExtentScript = ''
programmableFilter1.PythonPath = ''

UpdatePipeline(time=100.0, proxy=programmableFilter1)

# Calculate vorticity field
calculator1 = Calculator(registrationName='Calculator1',
                         Input=programmableFilter1)
calculator1.AttributeType = 'Cell Data'
calculator1.ResultArrayName = 'vorticity'
calculator1.Function = '(GRAD_velocity_2Y - GRAD_velocity_1Z) * iHat + (GRAD_velocity_0Z - GRAD_velocity_2X) * jHat + (GRAD_velocity_1X - GRAD_velocity_0Y) * kHat'

UpdatePipeline(time=100.0, proxy=calculator1)

# Contour by lambda2 = 0
contour1 = Contour(registrationName='Contour1', Input=calculator1)
contour1.ContourBy = ['POINTS', 'lambda2']
contour1.Isosurfaces = [0.0]
contour1.PointMergeMethod = 'Uniform Binning'

UpdatePipeline(time=100.0, proxy=contour1)
