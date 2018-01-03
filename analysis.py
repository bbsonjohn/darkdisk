#========================================================
#
#     Project: Dark Disk
#
#     Desciption : Compare data with prediction
#
#     Author : John Leung
#
#========================================================

import numpy as np
import scipy.integrate as nint
import scipy.special as special
import scipy.linalg as linalg
import scipy.stats as stats
import scipy.optimize as opt
import matplotlib.pyplot as plt
from tqdm import tqdm
import integrator as intt

zRange = 2600
dzStep = 0.1
indexDHalo = 14
indexDDisk = 15 

def PoissonJeansSolve(hDD, SigDD, zRange):
	
	maxIteration = 100
	convergence = 1e-4
	loopCounter = 0
	xspace = np.linspace(0, zRange, zRange/dzStep)
	DMConstraintZ = 2500

	matterParam = intt.parameterMatter(hDD,SigDD)
	sigma = matterParam[0]
	rho = matterParam[1]

	DefaultRhoDHalo = rho[indexDHalo]

	rhoSolved = float("nan")
	rhoTVector = [DefaultRhoDHalo]
	rhoTm1Vector = [0]
	
	while np.absolute(rhoTm1Vector[loopCounter]/rhoTVector[loopCounter]-1.) > convergence:
		sol = intt.PoissonJeansIntegrator(hDD, SigDD, xspace, rhoDH = rhoTVector[loopCounter])
		DHdensity = intt.densBachall(sol[:,0],indexDHalo,rho,sigma,normalized=True)

		for i in range(len(xspace)):
			if xspace[i] > DMConstraintZ:
				DMConstraintRho = np.interp(DMConstraintZ,[xspace[i-1],xspace[i]],[DHdensity[i-1],DHdensity[i]])
				rhoSolved = rho[indexDHalo]/DMConstraintRho
				break			
			if i == len(xspace)-1 :
				print ("Integration range too small!")
				return float['nan']
		
		if np.isnan(rhoSolved):
			print ("Linear solver exception!")
			return float['nan']
		if loopCounter > maxIteration:
			print("Solver does not converge!")
			return float['nan']

		rhoTm1Vector = np.append(rhoTm1Vector, rhoTVector[loopCounter])
		rhoTVector = np.append(rhoTVector, rhoSolved)
		loopCounter = loopCounter + 1		
#	print("Dark matter Density at z = 0:", rhoTVector[len(rhoTVector)-1])
	return sol 

#--------------------------------------------------------------------------------

def diskParamReturn(sol, hDD, SigDD, iComp = indexDDisk):
	hScaleDefine = (np.cosh(0.5))**(-2)
	matterParam = intt.parameterMatter(hDD,SigDD)
	sigma = matterParam[0]
	rho = matterParam[1]
	xspace = np.linspace(0, zRange, zRange/dzStep)

	density = intt.densBachall(sol[:,0],iComp,rho,sigma,normalized=False)
	surfDensity = np.trapz(density, xspace)

	density_norm = intt.densBachall(sol[:,0],iComp,rho,sigma,normalized=True)
	diskHeight = float('nan')
	for i in range(len(density_norm)):
		if density_norm[i] < hScaleDefine:
			diskHeight = np.interp(hScaleDefine, [density_norm[i-1],density_norm[i]], [xspace[i-1],xspace[i]])
			break			
		if i == len(density_norm)-1 :
			print ("Disk height not found!")

	return [diskHeight, surfDensity]	

#----------------------------------------------------------------------------------------------------------------

def fetchWDist(filename, dw=0.01, mean_adj=False):
	columnW = 0
	columnWErr = 1

	w, werr = np.loadtxt(filename, delimiter= ",", skiprows=1, usecols=(columnW, columnWErr), unpack=True)
	if mean_adj:
		w = w - np.mean(w)
	wAbs = np.absolute(w)	


	wspacePos = np.linspace(0., np.amax(wAbs), np.amax(wAbs)/dw )
	wspaceNeg = np.linspace(-np.amax(wAbs), 0., np.amax(wAbs)/dw )
	densityWidth = np.sqrt((np.mean(werr))**2+(len(wspacePos) * 3./4.)**(-2./5.))
	wkernel = stats.gaussian_kde(w, densityWidth)

	wdistPos = wkernel.evaluate(wspacePos)
	wdistNeg = wkernel.evaluate(wspaceNeg)
	wdistNeg = np.flipud(wdistNeg)
	wdist = wdistPos + wdistNeg

#	print(np.trapz(wdist, wspacePos))
#	plt.hist(wAbs, bins=40, normed=1)
#	plt.plot(wspacePos, wdist)

	return (wspacePos, wdist)

#----------------------------------------------------------------------------------------------------------------

def fetchWDist_gaia(filename, dw=0.05, gaus_approx=True, verbose=True):
	columnW = 0
	columnCount = 1
	columnCountErr = 2

	w, count, count_err = np.loadtxt(filename, delimiter= ",", skiprows=1, usecols=(columnW, columnCount, columnCountErr), unpack=True)

	for i, err in enumerate(count_err):
		if np.isnan(err):
			count_err[i] = 1.

	gaus = lambda x, s: (2./(s*np.sqrt(2.*np.pi)) )*np.exp(-0.5*(x/s)**2)
	#mean_loc = 0.
	#gaus = lambda x, s: 2.*stats.norm.pdf(x, mean_loc, s)
	sigma, sigma_err = opt.curve_fit( gaus, w, count,  sigma=count_err)
	if verbose == True:
		print("Best-fit velocity sigma = ", sigma, " +/- ", sigma_err)

	wspacePos = np.linspace(0., np.amax(w), np.amax(w)/dw )

	#	print(np.trapz(wdist, wspacePos))
	#	plt.hist(wAbs, bins=40, normed=1)
	#	plt.plot(wspacePos, wdist)

	return (wspacePos, gaus(wspacePos, sigma))



#-------------------------------------------------------------------------------------------------

def fetchZDist(filename, zspace):

	columnZ = 5
	columnZErr = 16

	z, zerr = np.loadtxt(filename, delimiter= ",", skiprows=1, usecols=(columnZ, columnZErr), unpack=True)	
	densityWidth = np.sqrt((np.mean(zerr))**2+(len(zspace) * 3./4.)**(-2./5.))

	zkernel = stats.gaussian_kde(z, densityWidth)
	zdist = zkernel.evaluate(zspace)
#	np.trapz(zdist,zspace)
	
	return zdist
#--------------------------------------------------------------------------------------------------

def plotWDist(filename, wdist, dw, bins=40):
	columnW = 0

	w = np.loadtxt(filename, delimiter= ",", skiprows=1, usecols=(columnW), unpack=True)	
	w = w - np.mean(w)
	w = np.absolute(w)	

	wspace = np.linspace(0., np.amax(w), np.amax(w)/dw)

	print("normalization ", np.trapz(wdist, wspace))

	plt.xlabel('w')
	plt.ylabel('Normalized unit')
	plt.hist(w, bins, normed=1)
	plt.plot(wspace, wdist)
	plt.show()

	return None

#-------------------------------------------------------------------------------------------------

def fetchZPredict(phi_in, zspace, wdist, wspace):

	phiConversion = 4*np.pi*intt.gnewt*(intt.vConversion**2)
	phi = phiConversion*phi_in
	xLimit = []
	dw = np.mean( (np.roll(wspace, -1) - wspace)[:-1] )

	for i in range(len(zspace)):
		j = np.ceil(np.sqrt(2*phi[i])/dw)		
		if j > len(wdist)-1:
			break
		xLimit = np.append(xLimit, j+1)

	rho_predict = []
	xIntSpace = np.linspace(0, (len(wdist)-1)*dw, len(wdist))

	for z in range(1,len(zspace)):
		if z > len(xLimit)-1:
			rho_predict = np.append(rho_predict, 0)
			continue	
			
		xJacobi = [float(x)*dw/np.sqrt((float(x)*dw)**2 - 2*phi[z])  for x in range(int(xLimit[z]), len(wdist))]

		xJacobi = np.hstack( (np.zeros(int(xLimit[z])), xJacobi) )
		fWPhi = wdist*xJacobi
		rho_predict = np.append(rho_predict, np.trapz( fWPhi , xIntSpace))

	potential_left = np.flipud(rho_predict)
	potential_right = np.hstack( (np.trapz(wdist, xIntSpace),rho_predict) )
	potential = np.hstack( (potential_left,potential_right) )

	return potential

#-------------------------------------------------------------------------------------------------

def plotFunction(funct, space, PlotLabel = "Plot", AxesLabels = ['x','y'], normalized = True):

	try:
		for i in range(funct):
			if normalized :
				funct[i][:] = funct[i][:]/funct[i][int(np.ceil(len(funct[i])/2.))]
			iplotName = 'plot_' + str(i)
			if isinstance(PlotLabel, basestring):
				plt.plot(space, funct[i], label=PlotLabel)
			else :
				try:
					plt.plot(space, funct[i], label=str(PlotLabel[i]) )
				except TypeError:
					plt.plot(space, funct[i], label=str(PlotLabel) )

	except TypeError:
		plt.plot(space, funct, label=PlotLabel)

	plt.grid()
	plt.xlabel(AxesLabels[0])
	plt.ylabel(AxesLabels[1])
	plt.show()
	return

#-------------------------------------------------------------------------------------------------

def likelihoodDensity(zspace, prediction, data, delta_pred, delta_data):

	starUpperZ = 170.
	starLowerZ = -170
	integrateStart = 0
	integrateEnd = 0
	epsilon = 0.00001

	for i in range(len(zspace)):
		if zspace[i] > starLowerZ:
			integrateStart = i
			break
	for i in range(len(zspace)):
		j = len(zspace)-i-1
		if zspace[j] < starUpperZ:
			integrateEnd = j 
			break

	zspace=zspace[integrateStart:integrateEnd]
	prediction=prediction[integrateStart:integrateEnd]
	data=data[integrateStart:integrateEnd]
	delta_data=delta_data[integrateStart:integrateEnd]
	delta_pred=delta_pred[integrateStart:integrateEnd]

	variance = (delta_pred/(prediction+epsilon) )**2 + (delta_data/(data+epsilon) )**2
	data = data/np.sqrt(variance)
	prediction = prediction/np.sqrt(variance)

	likelihood = np.sum( np.log( stats.norm.pdf(np.log(data), np.log(prediction), np.sqrt(variance)) )   )

#	for i in range(len(differ)):
#		print(differ[i], uncertainty[i])
#	return (zspace, likelihood)
	return likelihood
#-------------------------------------------------------------------------------------------------
def bootstrap(funct, space, star_size, times = 1000):

	sample_set = []
	old_norm = np.sum(funct)
	funct_pdf = funct/old_norm
	densityWidth = (len(space) * 3./4.)**(-1./5.)
	for i in range(times):
		sample = sorted(np.random.choice(space, size=star_size ,p=funct_pdf))
		kernel = stats.gaussian_kde(sample, densityWidth)
		dist = kernel.evaluate(space)
		sample_set.append(old_norm*dist/(np.sum(dist)))

	variance = np.zeros(len(space))
	for j in range(times):
		variance = variance + (funct-sample_set[j])**2
		
#	plotFunction([funct,sample_set[4]], zspaceFull,nPlot=2,normalized = False)
	return np.sqrt(variance/times)



#-------------------------------------------------------------------------------------------------
def logLikelihood(binning, data, predict, sigma):

	likelihood = 1./( sigma*np.sqrt(2*np.pi))*np.exp(-0.5*(np.log(data)-np.log(predict))**2/(2*sigma**2))
	llh = np.log( np.prod(likelihood) )
	return llh

