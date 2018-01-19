#========================================================
#
#     Project: Dark Disk
#
#     Desciption : NIntegrating the Stars Distribution
#
#     Author : John Leung
#
#========================================================

import numpy as np
import scipy.integrate as intt
import scipy.special as special
import matplotlib.pyplot as plt

gnewt = 4.516e-30
vConversion = 3.0857e13
totalNumComp = 14

#----------------------------------------------------------------------------------------

def parameterMatter(hDD, SigDD, rhoDHalo = 0.015, sigmaDHalo = 2000.):
   """   sigmaH2 = 4.0; rhoH2 = 0.014;
   sigmaHI1 = 7.0; rhoHI1 = 0.015;
   sigmaHI2 = 9.0; rhoHI2 = 0.005;
   sigmaWGas = 40.0; rhoWGas = 0.0011;
   sigmaGiants = 20.0; rhoGiants = 0.0006;
   sigmaMV2p5 = 7.5; rhoMV2p5 = 0.0018;
   sigma3MV4 = 14.0; rho3MV4 = 0.0018;
   sigma4MV5 = 18.0; rho4MV5 = 0.0029;
   sigma5MV8 = 18.5; rho5MV8 = 0.0072;
   sigmaMV8 = 18.5; rhoMV8 = 0.0216;
   sigmaWDwarfs = 20.0; rhoWDwarfs = 0.0056;
   sigmaBDwarfs = 20.0; rhoBDwarfs = 0.0015;
   sigmaTDisk = 37.0; rhoTDisk = 0.0035;
   sigmaSHalo = 100.0; rhoSHalo = 0.0001;"""

   sigmaH2 = 3.7; rhoH2 = 0.0104;
   sigmaHI1 = 7.1; rhoHI1 = 0.0277;
   sigmaHI2 = 22.1; rhoHI2 = 0.0073;
   sigmaWGas = 39.0; rhoWGas = 0.0005;
   sigmaGiants = 15.5; rhoGiants = 0.0006;
   sigmaMV2p5 = 7.5; rhoMV2p5 = 0.0018;
   sigma3MV4 = 12.0; rho3MV4 = 0.0018;
   sigma4MV5 = 18.0; rho4MV5 = 0.0029;
   sigma5MV8 = 18.5; rho5MV8 = 0.0072;
   sigmaMV8 = 18.5; rhoMV8 = 0.0216;
   sigmaWDwarfs = 20.0; rhoWDwarfs = 0.0056;
   sigmaBDwarfs = 20.0; rhoBDwarfs = 0.0015;



   if SigDD == 0.:
       sigma = [sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants, sigmaMV2p5, sigma3MV4, sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs, sigmaDHalo]
       rho = [rhoH2, rhoHI1, rhoHI2, rhoWGas, rhoGiants, rhoMV2p5, rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs, rhoDHalo]
       return [sigma, rho]

   rhoDDisk = SigDD/(4.*hDD)
   sigmaDDisk = hDD*vConversion*np.sqrt(8*np.pi*gnewt*rhoDDisk)

   sigma = [sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants, sigmaMV2p5, sigma3MV4, sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs, sigmaDHalo, sigmaDDisk]
   rho = [rhoH2, rhoHI1, rhoHI2, rhoWGas, rhoGiants, rhoMV2p5, rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs, rhoDHalo, rhoDDisk]
       
   return [sigma, rho]

#----------------------------------------------------------------------------------------

def densBachall(phi, iComp, rho0, sigma, normalized = False):

   sigmaConversion = 1./(vConversion*np.sqrt(8*np.pi*gnewt))

   if normalized == True:
      rho = np.ones(len(sigma))
   if normalized == False:
      rho = rho0
   a = rho[iComp]*np.exp(-phi/(np.sqrt(2)*sigmaConversion*sigma[iComp])**2)
   return a

#----------------------------------------------------------------------------------------

def functs(y, x, rho0, sigma):
   # phi'= phip
   # phip' = Sum[Rho[i] Exp[-phi[x]/(Sqrt[2] Sigma[[i]])^2], {k, 1, nComp}]

   phi, phip = y
   source = np.array([])
   for i in range(0, len(rho0), 1):
      source = np.append(source, densBachall(phi,i, rho0, sigma) )
   dy = [phip, np.sum(source)]

   return dy

#----------------------------------------------------------------------------------------

def PoissonJeansIntegrator(hDD, SigDD, xspace, rhoDH = 0.008):

   params = parameterMatter(hDD, SigDD, rhoDHalo = rhoDH)
   sigma = params[0]
   rho = params[1]

   initialPhi = 0.0; initialPhiPrime = 0.0;

   initCond = [initialPhi , initialPhiPrime]

   sol = intt.odeint(functs, initCond, xspace, args=(rho, sigma))

   return sol

#----------------------------------------------------------------------------------------

def plotPotential(hDD, SigDD, zRange, dzStep=0.01):
   xspace = np.linspace(0, zRange, zRange/dzStep)
   solution = PoissonJeansIntegrator(hDD, SigDD, xspace)
   plt.plot(xspace, solution[:, 0], label='phi(z)')
   plt.grid()
   plt.xlabel('z / pc')
   plt.ylabel('Phi(z)')
   plt.show()
   return 0

#----------------------------------------------------------------------------------------

def plotDensity(hDD, SigDD, zRange, iComp = [totalNumComp-1], dzStep=0.01, normalized=True):

   xspace = np.linspace(0, zRange, zRange/dzStep)
   solution = PoissonJeansIntegrator(hDD, SigDD, xspace)

   params = parameterMatter(hDD, SigDD)
   sigma = params[0]
   rho = params[1]
   yAxisLabel = "density"

   if normalized == True:
      yAxisLabel = "Normalized unit"

   for i in range(len(iComp)):
      plotName = 'Comp' + str(iComp[i])
      plotLabel = 'rho_' + str(iComp[i]) +'(z)'
      density = densBachall(solution[:,0],iComp[i],rho,sigma,normalized)
      plt.plot(xspace, density, label=plotLabel)

   plt.grid()
   plt.legend(loc='best')
   plt.xlabel('z / pc')
   plt.ylabel(yAxisLabel)
   plt.show()
   return 0   

#----------------------------------------------------------------------------------------


