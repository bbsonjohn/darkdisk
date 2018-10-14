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
import integrator as intt

zRange = 2600
zRange_Red = 260
dzStep = 0.1
indexDHalo = 12
indexDDisk = 13

wfile_columnW = 0
wfile_columnCount = 1
wfile_columnCountErr = 2

zfile_columnl = 1
zfile_columnb = 2
zfile_columnPlx = 3
zfile_columnErrPlx = 4
zfile_columnZ = 5
zfile_columnEVFS = 6
zfile_columnPml = 11
zfile_columnPmb = 12
zfile_columnPmlErr = 13
zfile_columnPmbErr = 14
zfile_columnZErr = 16
zfile_columnVR = 17

def PoissonJeansSolve(hDD, SigDD, zRange):
   
   maxIteration = 100
   convergence = 1e-4
   loopCounter = 0
   xspace = np.linspace(0, zRange, zRange/dzStep)
   xspaceFull = np.linspace(-zRange, zRange, 2*zRange/dzStep-1)
   DMConstraintZ = 2500

   sigma, rho  = intt.parameterMatter(hDD,SigDD)

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

def PoissonJeansSolve_gaia(hDD, SigDD, zRange, use_default_density = False, sigma_in = None, rho_in = None):
   
   maxIteration = 100
   convergence = 1e-4
   loopCounter = 0
   xspace = np.linspace(0, zRange, zRange/dzStep)
   xspaceFull = np.linspace(-zRange, zRange, 2*zRange/dzStep-1)
   
   sol = intt.PoissonJeansIntegrator(hDD, SigDD, xspace, use_default_density=use_default_density, sigma = sigma_in, rho = rho_in, use_rect_grid_method = True)
   
   solu = sol[:,0]
   sol_full = np.flip(solu[(-len(solu)+1):],0)
   sol_full = np.append(sol_full,solu)
   sol_interp = interp.interp1d(xspaceFull, sol_full, kind='cubic')

   return sol_interp

#--------------------------------------------------------------------------------

def diskParamReturn(solu, hDD, SigDD, iComp = indexDDisk, use_default_density = False, sigma_in = None, rho_in = None):
   hScaleDefine = (np.cosh(0.5))**(-2)
   dzStep_int = 0.01
   
   sigma, rho  = intt.parameterMatter(hDD, SigDD, use_default_density = use_default_density, sigma_in = sigma_in, rho_in = rho_in)
   
   if SigDD == 0.:
       return (hDD, SigDD)

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
   return (diskHeight, surfDensity)   

#----------------------------------------------------------------------------------------------------------------

def kernelize_hist(x, y, x_err):
   gaus = lambda x, s, x0: (1./(s*np.sqrt(2.*np.pi)) )*np.exp(-0.5*((x-x0)/s)**2)
   
   xdivision = 100.
   xspacing = x[1] - x[0]
   xspacing_fine = xspacing/xdivision
   x_range = np.amax(x) - np.amin(x)
   xspace_fine = np.linspace(np.amin(x)-2*xspacing, np.amax(x)+2*xspacing, x_range/xspacing_fine+4*xdivision+1)
   
   norm = np.sum(y)
   xkernel = np.zeros_like(xspace_fine)
   
   for i in range(len(xspace_fine)):
      indx_vicinity_bin = (np.abs(x-xspace_fine[i]) < 2*xspacing)
      x_reduced = x[indx_vicinity_bin]; x_err_reduced = x_err[indx_vicinity_bin]; y_reduced = y[indx_vicinity_bin]
      xkernel[i] = np.sum( [y_reduced[j]*gaus(x_reduced[j], x_err_reduced[j], xspace_fine[i]) for j in range(len(x_reduced))] )   

   x_edges = np.append(x-xspacing/2., x[-1]+xspacing/2.)
   x_out, _, _ = plt.hist(xspace_fine, x_edges, weights=xkernel)
   x_out = x_out*norm/np.sum(x_out)
   plt.clf()

   return x_out

#----------------------------------------------------------------------------------------------------------------

def fetchWData(filename, wSun = 7.01, b_cut_deg = 5., show_plot = False, vr_from_file = False, to_save_file = False, out_filename = None, verbose=True):

   vKappa = 4.74047 #( km s^-1 mas (mas yr)^-1 )
   uSun = 11.1
   vSun = 12.24
   wNBin = 30.
   w_Range = 45.
   delw = w_Range/wNBin
   
   ldeg, bdeg, plx, pml, pmb, evfs_w, zerr, vr = np.loadtxt(filename, delimiter= ",", skiprows=1, \
       usecols=(zfile_columnl, zfile_columnb, zfile_columnPlx, zfile_columnPml, zfile_columnPmb, zfile_columnEVFS, zfile_columnZErr, zfile_columnVR), unpack=True)
    
   midplane_cut_indx = ( np.abs(bdeg) <= b_cut_deg )
   ldeg = ldeg[midplane_cut_indx]
   bdeg = bdeg[midplane_cut_indx]
   plx = plx[midplane_cut_indx]
   pml = pml[midplane_cut_indx]
   pmb = pmb[midplane_cut_indx]
   evfs_w = evfs_w[midplane_cut_indx]
   zerr = zerr[midplane_cut_indx]
   vr = vr[midplane_cut_indx]

   lRad = ldeg*np.pi/180.
   bRad = bdeg*np.pi/180.

   VRMean = float('nan')
   if vr_from_file:
      VRMean = 0.
   else:
      VRMean = -uSun*np.cos(lRad)*np.cos(bRad) - vSun*np.sin(lRad)*np.cos(bRad) - wSun*np.sin(bRad)
      
   VZ =  wSun + vKappa*pmb*np.cos(bRad)/plx + VRMean*np.sin(bRad)

   w_space = np.linspace(0., w_Range, wNBin+1)
   count, _, _ = plt.hist(np.abs(VZ), w_space, weights=evfs_w)
   count_err, _ , _ = plt.hist(np.abs(VZ), w_space)
   plt.clf()
   
   for i, w_c_err in enumerate(count_err):
      if w_c_err == 0:
         continue
      else :
         count_err[i] = count[i]/np.sqrt(w_c_err)


   w = (w_space + delw/2.)[:-1]

   if to_save_file:
      line_header = "# w bins (km/s), count, count_err"
      np.savetxt(out_filename , np.transpose( np.array([w , count, count_err]) ), delimiter=',', header=line_header, fmt='%10.5f')
      print("Mid-plane velocity saved to: " + out_filename )

   if verbose:
      print("Star count after mid-plane cut: " + str(np.sum(midplane_cut_indx)) )
      print("Star count after mid-plane cut (evfs-normalized): " + str(np.sum(count)) )
      print("mean star velocity: " + str(np.mean(VZ)) + " km/s" )
      print("mean star velocity (evfs-normalized): " + str(np.sum(VZ*evfs_w)/(np.sum(evfs_w)) )  + " km/s")  

   if show_plot:
      w_space_full = np.linspace(-w_Range, w_Range, 2*wNBin+1)
      w_bins_full = (w_space_full + delw/2.)[:-1]
      count_bins, _, _ = plt.hist(VZ, w_space_full, weights=evfs_w)
      plt.clf()
      plt.title("w-space distribution"); plt.xlabel("w"); plt.ylabel("count");
      plt.errorbar(w_bins_full, count_bins, yerr=np.sqrt(count_bins), xerr=delw/2., capthick=2, ls='none')

   return (w, count, count_err)

#----------------------------------------------------------------------------------------------------------------
def fetchWDist_gaia(w_in=None, count_in=None, count_err_in=None, filename=None, wspace_out=None, gaus_approx=False, verbose=True, from_file = False, show_plot=False):

   w, count, count_err = (None, None, None)

   if from_file:
      w, count, count_err = np.loadtxt(filename, delimiter= ",", skiprows=1, usecols=(wfile_columnW, wfile_columnCount, wfile_columnCountErr), unpack=True)
   else:
      w = w_in
      count = count_in
      count_err = count_err_in
   
   norm = np.trapz(count,x=w)
   count = count/norm
   if count_err is not None:
      count_err = count_err/norm

   wspacePos = wspace_out
   gaus = lambda x, s: (2./(s*np.sqrt(2.*np.pi)) )*np.exp(-0.5*(x/s)**2)
   fit_cutoff = len(w);

   if gaus_approx:
      initial_guess_p0 = 1.

      for i, w_err in enumerate(count_err):
         if (count[i-1] > 0.7*count[0]) and (count[i] < 0.7*count[0]):
            initial_guess_p0 = w[i]
         if (w_err == 0.) or np.isnan(w_err):
            fit_cutoff = i
            break


      sigma, sigma_err = opt.curve_fit( gaus, w[:fit_cutoff], count[:fit_cutoff], sigma=count_err[:fit_cutoff], p0=initial_guess_p0 )

      chi_sq = np.sum( (gaus(w,sigma)-count)**2./count_err**2 )/(len(count)-2)
      if verbose == True:
         print("Best-fit velocity sigma = " + str(sigma) + " +/- " + str(sigma_err) )
         print("chi-squared/ndf = " + str(chi_sq) + ", with n dof = " + str(len(count)-2) )

      if show_plot:
         plotError( [gaus(w, sigma), count], [0, count_err], w, AxesLabels=['w','Count'], PlotLabel=['gaus fit','data'])
         
         
      return (wspacePos, gaus(wspacePos, sigma))


   fw_out = interp.interp1d(w, count, kind='linear')
   fw_out_array = np.array([])

   for i, wsteps in enumerate(wspacePos):
      if wsteps < w[0]:
          fw_out_array = np.append(fw_out_array, fw_out(w[0]))
      elif wsteps > w[len(w)-1]:
          fw_out_array = np.append(fw_out_array, 0.)
      else :
          fw_out_array = np.append(fw_out_array, fw_out(wsteps))   

   fw_norm = np.trapz(fw_out_array, x = wspacePos)
   fw_out_array = fw_out_array/fw_norm

   if show_plot:
      delw = np.mean( (np.roll(w,-1) - w)[:-1] )
      plt.errorbar(w, count/fw_norm, yerr=count_err/fw_norm, xerr=delw/2., capthick=2, label='data', ls='none')
      plt.plot(wspacePos, fw_out_array, label='interpolation')
      plt.title("f(|w|)"); plt.xlabel("w"); plt.ylabel("count"); plt.legend();
      plt.show()

   return (wspacePos, fw_out_array)

#-------------------------------------------------------------------------------------------------

def fetchZDist(filename, zspace, zSun = 0., use_evfs=True, show_plot=False):

   z, zerr, evfs_w = np.loadtxt(filename, delimiter= ",", skiprows=1, usecols=(zfile_columnZ, zfile_columnZErr, zfile_columnEVFS), unpack=True)   
   densityWidth = np.sqrt(np.mean(zerr))**2  #. +(len(zspace) * 3./4.)**(-2./5.))
   zspacing = zspace[1]-zspace[0]
   z_evfs = np.zeros_like(zspace)
   zspacing_division = 100.

   z = z - zSun

   if use_evfs == False:
      evfs_w = np.ones_like(z)

   print("mean z position: " + str(np.sum(z*evfs_w)/np.sum(evfs_w)) )      

   zspacing_fine = zspacing/zspacing_division
   z_range = np.amax(zspace) - np.amin(zspace)
   zspace_fine = np.linspace(np.amin(zspace)-2*zspacing, np.amax(zspace)+2*zspacing, z_range/zspacing_fine+4*zspacing_division+1)


   dist_kernels = np.zeros_like(zspace_fine)
   for i, z_coord in enumerate(zspace_fine):
      z_relevant_indx = ( np.abs(z_coord-z) < 3.*zerr )
      z_relevant = z[z_relevant_indx]
      evfs_relevant = evfs_w[z_relevant_indx]
      zerr_relevant = zerr[z_relevant_indx]
      dist_kernels[i] = np.sum( evfs_relevant*gaus( (z_relevant - z_coord), zerr_relevant ) )

   zdist = np.zeros_like(zspace)
   zdist_2 = np.zeros_like(zspace)
   for i, z_coord in enumerate(zspace):
      z_in_bin = (zspace_fine > (z_coord - zspacing/2.))*(zspace_fine <= (z_coord + zspacing/2.))
      zdist[i] = np.sum(dist_kernels[z_in_bin])*zspacing_fine
      
      z_in_bin_2 = (z > (z_coord - zspacing/2.))*(z <= (z_coord + zspacing/2.))
      zdist_2[i] = np.sum( evfs_w[z_in_bin_2] )

   if show_plot:
      plotFunction([zdist, zdist_2], zspace, PlotLabel=["with kernel", "without kernel"] , AxesLabels=['z','Counts'], normalized=False, logScale = True)
   
   return zdist
#--------------------------------------------------------------------------------------------------

def fetchZDistHist(filename, zspace, zSun = None, use_evfs=True, show_plot=False):

   zdist = np.zeros_like(zspace)
   zspacing = np.mean( (np.roll(zspace,-1)-zspace)[:-1] )
   z, evfs_w = np.loadtxt(filename, delimiter= ",", skiprows=1, usecols=(zfile_columnZ, zfile_columnEVFS), unpack=True)
   #z_in_bin = np.array(np.zeros_like(), dtype=bool)

   if zSun is None:
      zSun = np.mean(z)
       
   z = z - zSun

   for i, z_coord in enumerate(zspace):
      z_in_bin = (z > (z_coord - zspacing/2.))*(z <= (z_coord + zspacing/2.))
      if use_evfs:
         if np.sum( z_in_bin ) == 0:
            zdist[i] = 0
         else:
            zdist[i] = np.sum( evfs_w[z_in_bin] )
      else:
         zdist[i] = np.sum( z_in_bin )
         
   if show_plot:
      plotFunction(zdist, zspace, PlotLabel="star z-distribution", AxesLabels=['z','Counts'], normalized=False)
   
   return zdist
#--------------------------------------------------------------------------------------------------

def plotWDist(filename, wdist, dw, bins=40):

   w = np.loadtxt(filename, delimiter= ",", skiprows=1, usecols=(wfile_columnW), unpack=True)   
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

def fetchZPredict(phi_in, zspace_phi, zspace_out, wdist, wspace, use_gaus_kernel = False, show_plot=False):

   phiConversion = 4*np.pi*intt.gnewt*(intt.vConversion**2)
   phi = phiConversion*phi_in(zspace_phi)
   xLimit = []
   dw = wspace[1] - wspace[0]
   
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
      xJacobi = [2.*float(x)*dw/np.sqrt((float(x)*dw)**2 - 2.*phi[z])  for x in range(int(xLimit[z]), len(wdist))]

      xJacobi = np.hstack( (np.zeros(int(xLimit[z])), xJacobi) )
      fWPhi = wdist*xJacobi
      rho_pos = np.append(rho_pos, np.trapz( fWPhi , xIntSpace))

   rho_neg = np.flipud(rho_pos)
   rho = np.hstack( (rho_neg[:-1],rho_pos) )
   full_zspace = np.hstack( (np.flipud(-1.*zspace_phi)[:-1],zspace_phi) )
   rho_out = interp.interp1d(full_zspace, rho, kind='linear')
   
   #--------------------------------------------
   
   if use_gaus_kernel:  
      z_err = sigma_func(zspace_out/1000., tag='150')*1000.
      zdist = kernelize_hist(zspace_out, rho_out(zspace_out), z_err)
      if show_plot:
         plotFunction([np.log(zdist), np.log(rho_out(zspace_out))], zspace_out, PlotLabel=["with kernel", "without kernel"] , AxesLabels=['z','Counts'], normalized=False)
   else:
      zdist = rho_out(zspace_out)
      if show_plot:
         plotFunction(rho_out(zspace_out), zspace_out, PlotLabel="z density prediction", AxesLabels=['z','Counts'], normalized=False)

   return zdist
#-------------------------------------------------------------------------------------------------------
def fetchZPredict_bootstrap(phi_in, zspace_phi, zspace_out, wdata, wSun=0., wspace_in=None, b_cut_deg=5., resamples=100, vr_from_file = False, to_save_file = False, filename = "dens_predict_err.txt", show_plot = True):
   
   vKappa = 4.74047 #( km s^-1 mas (mas yr)^-1 )
   uSun = 11.1
   vSun = 12.24
   wNBin = 30. # number of bins
   w_Range = 45. # range of w in consideration
   delw = w_Range/wNBin # step size of w, calculated from # of bins and range
   
   ldeg, bdeg, plx, pml, pmb, evfs_w, zerr, vr = wdata # galatic coordinates, proper motions, error in z (unused), and rave radial velocity

   # mid-plane cut and implementation of it on the data   
   midplane_cut_indx = ( np.abs(bdeg) <= b_cut_deg ) 
   ldeg = ldeg[midplane_cut_indx]
   bdeg = bdeg[midplane_cut_indx]
   plx = plx[midplane_cut_indx]
   pml = pml[midplane_cut_indx]
   pmb = pmb[midplane_cut_indx]
   evfs_w = evfs_w[midplane_cut_indx]
   zerr = zerr[midplane_cut_indx]
   vr = vr[midplane_cut_indx]

   # convert degrees to radians
   lRad = ldeg*np.pi/180.
   bRad = bdeg*np.pi/180.

   #w-space grid the first one defines the bin edges (for the plt.hist function), the second define the center of bins
   w_space_edge = np.linspace(0., w_Range, wNBin+1)
   w_space = (w_space_edge + delw/2.)[:-1]

   # calculating VZ = w, use v_r from RAVE if available, use mean v_r otherwise 
   VRMean = -uSun*np.cos(lRad)*np.cos(bRad) - vSun*np.sin(lRad)*np.cos(bRad) - wSun*np.sin(bRad)
   
   if vr_from_file:
      V_r = np.nan_to_num(vr) + np.isnan(vr)*VRMean
   else:
      V_r = VRMean
      
   VZ =  wSun + vKappa*pmb*np.cos(bRad)/plx + V_r*np.sin(bRad)

   rho_samples = [] # the list that contains all the density samples
   
   # calculating statistical error of density prediction
   for i_sample in range(resamples): # bootstrap re-sampling
      VZ_sample = np.random.choice(VZ, len(VZ))
      count_sample, _, _ = plt.hist(np.abs(VZ_sample), w_space_edge, weights=evfs_w)
      plt.clf()
      wspace_fine, wfunct = fetchWDist_gaia(w_in=w_space, count_in=count_sample, wspace_out = wspace_in, gaus_approx=False) #smoothing function for f(w)
      rho_sample = fetchZPredict(phi_in, zspace_phi, zspace_out, wfunct, wspace_fine, use_gaus_kernel = True)  # output density function
      rho_samples.append(rho_sample)
   
   rho_samples = np.array(rho_samples) 
   count = np.mean(rho_samples, axis = 0) # the mean density averaged over all densities sample

   #calculating systematic error by generating two densities from f(w>0) and f(w<0).
   indx_pos_VZ = (VZ > 0.)
   indx_neg_VZ = (VZ < 0.)
   count_pos, _, _ = plt.hist(np.abs(VZ[indx_pos_VZ]), w_space_edge, weights=evfs_w[indx_pos_VZ])
   wspace_fine, wfunct = fetchWDist_gaia(w_in=w_space, count_in=count_pos, wspace_out = wspace_in, gaus_approx=False)
   rho_pos_sample = fetchZPredict(phi_in, zspace_phi, zspace_out, wfunct, wspace_fine, use_gaus_kernel = True)
   count_neg, _, _ = plt.hist(np.abs(VZ[indx_neg_VZ]), w_space_edge, weights=evfs_w[indx_neg_VZ])
   wspace_fine, wfunct = fetchWDist_gaia(w_in=w_space, count_in=count_neg, wspace_out = wspace_in, gaus_approx=False)
   rho_neg_sample = fetchZPredict(phi_in, zspace_phi, zspace_out, wfunct, wspace_fine, use_gaus_kernel = True)
   plt.clf()
   
   # generate bin by bin systematic error
   sys_err = np.abs(rho_pos_sample - rho_neg_sample)/((rho_pos_sample+rho_neg_sample)/2.)
   # generate bin by bin statistical error
   count_err = np.mean((rho_samples - count[np.newaxis,:])**2, axis = 0)
   #quadratuing the two errors
   count_err = np.sqrt(  count_err  + (count*sys_err)**2 )

   if to_save_file:
      line_header = "# z bins (pc), count, count_err"
      np.savetxt(filename , np.transpose( np.array([zspace_out , count, count_err]) ), delimiter=',',header=line_header, fmt='%10.5f')
      print("density prediction saved to: " + filename )
      
   
   if show_plot:
      plt.xlabel('z (pc)'); plt.ylabel(r'\rho (z)');
      plt.errorbar(zspace_out, count, yerr=count_err, capthick=2)
      plt.show()
   
   return (count, count_err)

#-------------------------------------------------------------------------------------------------
def fetch_z_systematics(filename, zspace, zdist):
    
   zspacing = stats.mode(np.roll(zspace,-1)-zspace)[0]
   z, zerr, evfs_w = np.loadtxt(filename, delimiter= ",", skiprows=1, usecols=(zfile_columnZ, zfile_columnZErr, zfile_columnEVFS), unpack=True)

   samplings = 1000
   rand_matrix = np.random.normal(0., 1., samplings*len(zerr))

   z_bin_err = np.zeros_like(zspace)
   for j in range(samplings):
      for i, z_coord in enumerate(zspace):
         low_indx= j*len(zerr) ; up_indx = (j+1)*len(zerr);
         gaus_err = rand_matrix[low_indx:up_indx]
         z_in_bin = (z+gaus_err*zerr > (z_coord - zspacing/2.))*(z+gaus_err*zerr <= (z_coord + zspacing/2.))
         z_bin_err[i] = z_bin_err[i]+(zdist[i] - np.sum( evfs_w[z_in_bin] ) )**2
         
   z_bin_err = np.sqrt(z_bin_err)/samplings

   return z_bin_err

#-------------------------------------------------------------------------------------------------
def likelihoodDensity(zspace, prediction, data, delta_pred, delta_data, starUpperZ = float('nan'), starLowerZ = float('nan'), plot_dist = True):
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
         integrateEnd = j+1 
         break

   zspace=zspace[integrateStart:integrateEnd]
   prediction=prediction[integrateStart:integrateEnd]
   data=data[integrateStart:integrateEnd]
   delta_data=delta_data[integrateStart:integrateEnd]
   delta_pred=delta_pred[integrateStart:integrateEnd]

   variance = (delta_pred)**2/((prediction)**2+epsilon) + (delta_data)**2/((data)**2+epsilon) + epsilon

   non_zero_index = (data > 0.)*(prediction > 0.)
   variance = variance[non_zero_index]
   data = data[non_zero_index]
   prediction = prediction[non_zero_index]
   
   likelihood_array =   1./(np.sqrt(2*np.pi*variance))*np.exp(-((np.log(data) - np.log(prediction))**2)/(2.*variance) )
   
   valid_indx = 1 - (likelihood_array > 0. )
   if np.sum(valid_indx) > 0.:
       return np.inf
   
   likelihood = -1.*np.sum( np.log(likelihood_array) ) 

   if plot_dist == True:
      plotError([np.log(prediction), np.log(data)], [delta_pred/prediction, delta_data/data], zspace, PlotLabel = ["pred", "data"], AxesLabels = ['z','count'])
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

   return np.sqrt(variance/times)

#-------------------------------------------------------------------------------------------------

def sigma_func(s, tag='200'): # i/o are in kpc
    if (tag=='tgas'):
        return 0.0013*pow(np.abs(s), 1.76) + 0.40
    elif (tag=='200'):
        return 0.0013*pow(np.abs(s), 1.574) + 0.17
    elif (tag=='150'):
        return 0.0007*pow(np.abs(s), 1.68) + 0.146

#-------------------------------------------------------------------------------------------------

def plotFunction(funct, space, PlotLabel = "Plot", AxesLabels = ['x','y'], normalized = True, logScale = False):
   plt.figure(figsize=(7.5,5))
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

   if len(PlotLabel) == len(funct):
      plt.legend()
   if logScale:
      plt.yscale('log')
   plt.grid()
   plt.xlabel(AxesLabels[0])
   plt.ylabel(AxesLabels[1])
   plt.show()
   return

#-------------------------------------------------------------------------------------------------

def plotError(funct, error, space, PlotLabel = "Plot", AxesLabels = ['x','y']):
   plt.figure(figsize=(7.5,5))
   try:
      for i in range(len(funct)):
         plt.errorbar(space, funct[i], yerr=error[i], capthick=2, label=PlotLabel[i])
         plt.ylim( ymax=(np.amax(funct)+0.1*(np.amax(funct)-np.amin(np.append(funct,0)))), ymin=(np.amin(funct)-0.2*(np.amax(funct)-np.amin(np.append(funct,0)))) );
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



