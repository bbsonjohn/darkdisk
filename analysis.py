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
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from tqdm import tqdm
import integrator as intt

zRange = 2600
zRange_Red = 260
dzStep = 0.1
indexDHalo = 12
indexDDisk = 13 

def PoissonJeansSolve(hDD, SigDD, zRange):
   
   maxIteration = 100
   convergence = 1e-4
   loopCounter = 0
   xspace = np.linspace(0, zRange, zRange/dzStep)
   xspaceFull = np.linspace(-zRange, zRange, 2*zRange/dzStep-1)
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
#   print("Dark matter Density at z = 0:", rhoTVector[len(rhoTVector)-1])
   
   solu = sol[:,0]
   sol_full = np.flip(solu[(-len(solu)+1):],0)
   sol_full = np.append(sol_full,solu)
   sol_interp = interp.interp1d(xspaceFull, sol_full, kind='cubic')

   return sol_interp

#--------------------------------------------------------------------------------

def diskParamReturn(solu, hDD, SigDD, iComp = indexDDisk):
   hScaleDefine = (np.cosh(0.5))**(-2)
   matterParam = intt.parameterMatter(hDD,SigDD)
   sigma = matterParam[0]
   rho = matterParam[1]
   dzStep_int = 0.01
   
   if SigDD == 0.:
       return [hDD, SigDD]   

   xspace = np.linspace(0, zRange, zRange/dzStep)
   sol = solu(xspace)
   density = intt.densBachall(sol,iComp,rho,sigma,normalized=False)
   surfDensity = 2.*np.trapz(density, xspace)

   xspace_Red = np.linspace(0, zRange_Red, zRange_Red/dzStep)
   sol = solu(xspace_Red)
   density_norm = intt.densBachall(sol,iComp,rho,sigma,normalized=True)

   diskHeight = float('nan')
   for i in range(len(density_norm)):
      if density_norm[i] < hScaleDefine:
         diskHeight = np.interp(hScaleDefine, [density_norm[i-1],density_norm[i]], [xspace_Red[i-1],xspace_Red[i]])
         break         
      if i == len(density_norm)-1 :
         print ("Disk height not found!")
   return [diskHeight, surfDensity]   

#----------------------------------------------------------------------------------------------------------------

def fetchWDist(filename, dw=0.01, mean_adj=False, show_plot=False):
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

   if show_plot:
      plotFunction(wdist, wspacePos, PlotLabel="w-distribution", AxesLabels=['w','Count'],normalized=False)

#   print(np.trapz(wdist, wspacePos))
#   plt.hist(wAbs, bins=40, normed=1)
#   plt.plot(wspacePos, wdist)

   return (wspacePos, wdist)

#----------------------------------------------------------------------------------------------------------------

def fetchWDist_gaia(filename, dw=0.05, gaus_approx=True, verbose=True, show_plot=False):
   columnW = 0
   columnCount = 1
   columnCountErr = 2

   w, count, count_err = np.loadtxt(filename, delimiter= ",", skiprows=1, usecols=(columnW, columnCount, columnCountErr), unpack=True)

   for i, w_err in enumerate(count_err):
      if count[i] < 0.7*count[0]:
         initial_guess_p0 = w[i]
      if (w_err == 0.) or np.isnan(w_err):
         count_err[i] = 1.
   
   norm = np.trapz(count,x=w)
   count = count/norm
   count_err = count_err/norm
   initial_guess_p0 = 1.

   gaus = lambda x, s: (2./(s*np.sqrt(2.*np.pi)) )*np.exp(-0.5*(x/s)**2)
   #mean_loc = 0.
   #gaus = lambda x, s: 2.*stats.norm.pdf(x, mean_loc, s)

   sigma, sigma_err = opt.curve_fit( gaus, w, count, sigma=count_err, p0=initial_guess_p0 )
   if verbose == True:
      print("Best-fit velocity sigma = ", sigma, " +/- ", sigma_err)

   wspacePos = np.linspace(0., np.amax(w), np.amax(w)/dw )

   if show_plot:
      plotError( [gaus(w, sigma), count], [0, count_err], w, AxesLabels=['w','Count'], PlotLabel=['gaus fit','data'])


   #   print(np.trapz(wdist, wspacePos))
   #   plt.hist(wAbs, bins=40, normed=1)
   #   plt.plot(wspacePos, wdist)

   return (wspacePos, gaus(wspacePos, sigma))



#-------------------------------------------------------------------------------------------------

def fetchZDist(filename, zspace):

   columnZ = 5
   columnZErr = 16

   z, zerr = np.loadtxt(filename, delimiter= ",", skiprows=1, usecols=(columnZ, columnZErr), unpack=True)   
   densityWidth = np.sqrt((np.mean(zerr))**2+(len(zspace) * 3./4.)**(-2./5.))

   zkernel = stats.gaussian_kde(z, densityWidth)
   zdist = zkernel.evaluate(zspace)
#   np.trapz(zdist,zspace)
   
   return zdist
#--------------------------------------------------------------------------------------------------

def fetchZDistHist(filename, zspace, use_evfs=True, show_plot=False):

   columnZ = 5
   columnEVFS = 6
   columnZErr = 16

   zdist = np.zeros_like(zspace)
   zspacing = stats.mode(np.roll(zspace,-1)-zspace)[0]
   z, evfs_w, zerr = np.loadtxt(filename, delimiter= ",", skiprows=1, usecols=(columnZ, columnEVFS, columnZErr), unpack=True)
   #z_in_bin = np.array(np.zeros_like(), dtype=bool)

   for i, z_coord in enumerate(zspace):
      z_in_bin = (z > (z_coord - zspacing/2.))*(z <= (z_coord + zspacing/2.))
      if use_evfs:
         if np.sum( z_in_bin[z_in_bin] ) == 0:
            zdist[i] = 0
         else:
            zdist[i] = np.sum( z_in_bin[z_in_bin]*(evfs_w[z_in_bin]) )
            #print((np.sum( z_in_bin[z_in_bin]/(evfs_w[z_in_bin]) ), np.sum( z_in_bin[z_in_bin])))
      else:
         zdist[i] = np.sum( z_in_bin )
         
   if show_plot:
      plotFunction(zdist, zspace, PlotLabel="star z-distribution", AxesLabels=['z','Counts'], normalized=False)
#   np.trapz(zdist,zspace)
   
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

def fetchZPredict(phi_in, zspace_phi, zspace_out, wdist, wspace, show_plot=False):

   phiConversion = 4*np.pi*intt.gnewt*(intt.vConversion**2)
   phi = phiConversion*phi_in(zspace_phi)
   xLimit = []
   dw = np.mean( (np.roll(wspace, -1) - wspace)[:-1] )

   xLimit = np.append(xLimit, 1)
   for i in range(1,len(zspace_phi)):
      j = np.ceil(np.sqrt( 2.*phi[i] )/dw)      
      if j > len(wdist)-1:
         break
      xLimit = np.append(xLimit, j+1)

   rho_pos = []
   xIntSpace = np.linspace(0, (len(wdist)-1)*dw, len(wdist))

   for z in range(len(zspace_phi)):
      if z > len(xLimit)-1:
         rho_pos = np.append(rho_pos, 0.)
         continue   
      xJacobi = [float(x)*dw/np.sqrt((float(x)*dw)**2 - 2.*phi[z])  for x in range(int(xLimit[z]), len(wdist))]

      xJacobi = np.hstack( (np.zeros(int(xLimit[z])), xJacobi) )
      fWPhi = wdist*xJacobi
      rho_pos = np.append(rho_pos, np.trapz( fWPhi , xIntSpace))

   rho_neg = np.flipud(rho_pos)
   rho = np.hstack( (rho_neg[:-1],rho_pos) )
   full_zspace = np.hstack( (np.flipud(-1.*zspace_phi)[:-1],zspace_phi) )
   rho_out = interp.interp1d(full_zspace, rho, kind='linear')
   
   if show_plot:
      plotFunction(rho_out(zspace_out), zspace_out, PlotLabel="z density prediction", AxesLabels=['z','Counts'], normalized=False)

   return rho_out(zspace_out)

#-------------------------------------------------------------------------------------------------

def plotFunction(funct, space, PlotLabel = "Plot", AxesLabels = ['x','y'], normalized = True):

   try:
      for i in range(len(funct)):
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
   except IndexError:
      plt.plot(space, funct, label=PlotLabel)
   except TypeError:
      plt.plot(space, funct, label=PlotLabel)
   except ValueError:
      if len(funct) == len(space):
         plt.plot(space, funct, label=PlotLabel)

   plt.grid()
   plt.xlabel(AxesLabels[0])
   plt.ylabel(AxesLabels[1])
   plt.show()
   return

#-------------------------------------------------------------------------------------------------

def plotError(funct, error, space, PlotLabel = "Plot", AxesLabels = ['x','y']):

   try:
      for i in range(len(funct)):
         plt.errorbar(space, funct[i], yerr=error[i], capthick=2, label=PlotLabel[i])
   except TypeError:
      plt.errorbar(space, funct, yerr=error[i], capthick=2)
   except IndexError:
      plt.errorbar(space, funct, yerr=error[i], capthick=2)      
   except ValueError:
      if len(funct) == len(space):
         plt.errorbar(space, funct)

   if len(PlotLabel) == len(funct):
      plt.legend()
   plt.grid()
   plt.xlabel(AxesLabels[0])
   plt.ylabel(AxesLabels[1])
   plt.show()
   return

#-------------------------------------------------------------------------------------------------
def likelihoodDensity(zspace, prediction, data, delta_pred, delta_data, starUpperZ = float('nan'), starLowerZ = float('nan'), plot_dist = False):
   rebinning = 10
   if np.isnan(starUpperZ) or np.isnan(starLowerZ):
       starUpperZ = np.amax(zspace)
       starLowerZ = np.amin(zsapce)
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

   zspace_rebin = np.empty([len(zspace)/rebinning])
   predict_rebin = np.empty([len(zspace)/rebinning])
   data_rebin = np.empty([len(zspace)/rebinning])
   delta_data_rebin = np.empty([len(zspace)/rebinning])
   delta_pred_rebin = np.empty([len(zspace)/rebinning])

   """for i in range(len(zspace)/rebinning):
      zspace_rebin[i] = zspace[int(rebinning*(i+0.5))] 
      predict_rebin[i] = np.sum( [ prediction[rebinning*i+j] for j in range(rebinning)] )
      data_rebin[i] = np.sum( [ data[rebinning*i+j] for j in range(rebinning)] )
      delta_data_rebin[i] = np.sqrt( np.sum( [ delta_data[rebinning*i+j]**2 for j in range(rebinning)] )  )
      delta_pred_rebin[i] = np.sqrt( np.sum( [ delta_pred[rebinning*i+j]**2 for j in range(rebinning)] )  )

      print((str("data: "),prediction[i],data[i]))
      print((delta_pred[i],delta_data[i]))"""

   variance = ((delta_pred)**2 + (delta_data)**2)/(data**2+epsilon) + epsilon

   #likelihood_array =  np.log( stats.norm.pdf(np.log(data), np.log(prediction), 2.*variance) )  
   likelihood_array =  np.log( 1./(np.sqrt(2*np.pi*variance))*np.exp(-(np.log(data) - np.log(prediction))**2)/(2.*variance) )
   #print(likelihood_array)
   valid_indx = ( np.isfinite(likelihood_array) )
   #print(likelihood_array[valid_indx])
   #print((data[valid_indx],prediction[valid_indx],variance[valid_indx]))
   likelihood = np.sum( likelihood_array[valid_indx]  )
   #print(valid_indx)
   #print((str("the likelihood is: ") , likelihood))
   
   if plot_dist == True:
      plotError([prediction, data], [delta_pred, delta_data], zspace, PlotLabel = ["pred", "data"])
      #plotFunction([predict_chisq, data_chisq], zspace, PlotLabel = ["pred", "data"])

#   for i in range(len(differ)):
#      print(differ[i], uncertainty[i])
#   return (zspace, likelihood)
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
      
#   plotFunction([funct,sample_set[4]], zspaceFull,nPlot=2,normalized = False)
   return np.sqrt(variance/times)



#-------------------------------------------------------------------------------------------------
def logLikelihood(binning, data, predict, sigma):

   likelihood = 1./( sigma*np.sqrt(2*np.pi))*np.exp(-0.5*(np.log(data)-np.log(predict))**2/(2*sigma**2))
   llh = np.log( np.prod(likelihood) )
   return llh




