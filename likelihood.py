import numpy as np
import scipy.integrate as nint
import scipy.interpolate as interp
import scipy.stats as stats
import scipy.optimize as opt
import matplotlib.pyplot as plt
from tqdm import tqdm
import integrator as intt
import analysis as an


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


def fetchWDist_fast(wdata, dw=0.05, verbose=True):

   initial_guess_p0 = 1.
   gaus = lambda x, s: (2./(s*np.sqrt(2.*np.pi)) )*np.exp(-0.5*(x/s)**2)
 
   w, count, count_err = wdata

   norm = np.trapz(count,x=w)
   count = count/norm
   count_err = count_err/norm

   fit_cutoff = len(w);
   wspacePos = np.linspace(0., np.amax(w), np.amax(w)/dw ) 
      
   for i, w_err in enumerate(count_err):
      if (count[i-1] > 0.7*count[0]) and (count[i] < 0.7*count[0]):
         initial_guess_p0 = w[i]
      if (w_err == 0.) or np.isnan(w_err):
         fit_cutoff = i
         break

   sigma, sigma_err = opt.curve_fit( gaus, w[:fit_cutoff], count[:fit_cutoff], sigma=count_err[:fit_cutoff], p0=initial_guess_p0 )

   if verbose:
      chi_sq = np.sum( (gaus(w,sigma)-count)**2./count_err**2 )/(len(count)-1)
      print("Best-fit velocity sigma = " + str(sigma) + " +/- " + str(sigma_err) )
      print("chi-squared/ndf = " + str(chi_sq) + ", with n dof = " + str(len(count)-2) )
         
   return (sigma, sigma_err)

def fetchZDist_fast(z, zerr, evfs_w_in, zspace, zSun = 0., use_evfs = True, verbose = False):

   evfs_w = evfs_w_in
   densityWidth = np.sqrt(np.mean(zerr))**2#. +(len(zspace) * 3./4.)**(-2./5.))
   zspacing = np.mean( (np.roll(zspace,-1)-zspace)[:-1] )
   zspacing_division = 100.

   if use_evfs == False:
      evfs_w = np.ones_like(z)
   
   z = z - zSun
   
   if verbose: 
      print("mean z position: " + str(np.sum(z*evfs_w)/np.sum(evfs_w)) )      

   zspacing_fine = zspacing/zspacing_division
   z_range = np.amax(zspace) - np.amin(zspace)
   zspace_fine = np.linspace(np.amin(zspace)-2*zspacing, np.amax(zspace)+2*zspacing, z_range/zspacing_fine+4*zspacing_division+1)


   gaus = lambda x, s: (1./(s*np.sqrt(2.*np.pi)) )*np.exp(-0.5*(x/s)**2)

   dist_kernels = np.zeros_like(zspace_fine)
   for i, z_coord in enumerate(zspace_fine):
      z_relevant_indx = ( np.abs(z_coord-z) < 3.*zerr )
      z_relevant = z[z_relevant_indx]
      evfs_relevant = evfs_w[z_relevant_indx]
      zerr_relevant = zerr[z_relevant_indx]
      dist_kernels[i] = np.sum( evfs_relevant*gaus( (z_relevant - z_coord), zerr_relevant ) )

   zdist = np.zeros_like(zspace)
   for i, z_coord in enumerate(zspace):
      z_in_bin = (zspace_fine > (z_coord - zspacing/2.))*(zspace_fine <= (z_coord + zspacing/2.))
      zdist[i] = np.sum(dist_kernels[z_in_bin])*zspacing_fine
  
   return zdist

#-------------------------------------------------------------------------------------------------------
def diskParamLlhReturn(param, h_DD, sig_DD, zdata, mean_param=0., err_param=0.):
    
   sigmaDHalo = float('nan')
   systematic_z = 0.03
   
   star_sig, sunZ, rhoDHalo, sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants, sigmaMV2p5, sigma3MV4,\
   sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs, rhoH2, rhoHI1, rhoHI2, rhoWGas, rhoGiants, rhoMV2p5,\
   rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs = param

   sigma = [sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants, sigmaMV2p5, sigma3MV4, sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs, sigmaDHalo]
   rho = [rhoH2, rhoHI1, rhoHI2, rhoWGas, rhoGiants, rhoMV2p5, rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs, rhoDHalo]

   prior_param = np.array([star_sig, rhoDHalo, sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants, sigmaMV2p5, sigma3MV4, \
   sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs, rhoH2, rhoHI1, rhoHI2, rhoWGas, rhoGiants, rhoMV2p5,\
   rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs])

   delw = 0.05
   delz = 20.
   w_Range = 40
   zRangeLarge = 2600.
   zRangeMed = 300.
   zRangeRed = 260.
   nBootstrapSample = 20
   l_range_p = 200.
   l_range_n = -200.

   zspace = np.linspace(0., zRangeRed, int(zRangeRed/delz))
   zspaceMed = np.linspace(0., zRangeMed, int(zRangeMed/delz))
   zspaceFull  = np.linspace(-zRangeRed, zRangeRed, 2*int(zRangeRed/delz)-1)
   zspaceLarge  = np.linspace(-zRangeLarge, zRangeLarge, 2*int(zRangeLarge/delz)-1)
   wspace = np.linspace(0., w_Range, int(w_Range/delw)+1 )

   star_sigma = star_sig
   wfunct = (2./(np.sqrt(2.*np.pi*star_sigma**2)) )*np.exp(-0.5*(wspace/star_sigma)**2)
   
   z, zerr, evfs_w = zdata

   starDensity = fetchZDist_fast(z, zerr, evfs_w, zspaceFull, zSun = sunZ)
   SDDensity = np.sqrt(fetchZDist_fast(z, zerr, evfs_w, zspaceFull, zSun = sunZ, use_evfs = False))
   for i, sd in enumerate(SDDensity):
      if sd == 0:
         SDDensity[i] = 1.
      else:
         SDDensity[i] = starDensity[i]/sd + systematic_z*starDensity[i]
   starDensity_norm = starDensity[int(np.ceil(len(starDensity)/2.-1))]
   #parallax_error = an.fetch_z_systematics(starFile, zspaceFull, starDensity)
   #parallax_error = 0.
   starDensity = starDensity/starDensity_norm
   starDensity_Delta = SDDensity/starDensity_norm

   solu = an.PoissonJeansSolve_gaia(h_DD, sig_DD, zRangeLarge, use_default_density = False, sigma_in = sigma, rho_in = rho)
   hdd_return, Sigdd_return = an.diskParamReturn(solu, h_DD, sig_DD, use_default_density = False, sigma_in = sigma, rho_in = rho)

   predict = an.fetchZPredict(solu, zspaceMed, zspaceFull, wfunct, wspace, show_plot=False)
   SDPredict = [0. for k in range(len(starDensity_Delta))]
   predict_norm = predict[int(np.ceil(len(predict)/2.-1))]
   predict = predict/predict_norm
   predict_Delta = SDPredict/predict_norm
   
   likelihood = an.likelihoodDensity(zspaceFull, predict, starDensity, predict_Delta, starDensity_Delta, l_range_p, l_range_n, plot_dist = False)
   if not np.isfinite(likelihood):
      return (hdd_return, Sigdd_return, np.inf)

   
   prior = lambda x, s: (1./(s*np.sqrt(2.*np.pi)) )*np.exp(-0.5*((x/s)**2))

   likelihood = likelihood - np.sum(np.log( prior( (prior_param - mean_param) , err_param) ))
   
   return (hdd_return, Sigdd_return, likelihood)

#-------------------------------------------------------------------------------------------------------
def loglikelihood(param, h_DD, sig_DD, zdata, mean_param=0., err_param=0.):
   nsigma_range = 3.
   systematic_z = 0.03
   sigmaDHalo = float('nan')

   star_sig, sunZ, rhoDHalo, sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants, sigmaMV2p5, sigma3MV4,\
   sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs, rhoH2, rhoHI1, rhoHI2, rhoWGas, rhoGiants, rhoMV2p5,\
   rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs = param

   sigma = [sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants, sigmaMV2p5, sigma3MV4, sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs, sigmaDHalo]
   rho = [rhoH2, rhoHI1, rhoHI2, rhoWGas, rhoGiants, rhoMV2p5, rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs, rhoDHalo]
   star_sigma = star_sig

   prior_param = np.array([star_sig, rhoDHalo, sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants, sigmaMV2p5, sigma3MV4, \
   sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs, rhoH2, rhoHI1, rhoHI2, rhoWGas, rhoGiants, rhoMV2p5,\
   rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs])

   if np.sum(sigma <= 0.)+np.sum(rho <= 0.)+(star_sigma <= 0.)+(rhoDHalo <= 0.) > 0.:
      return np.inf
   indx_cut_param = (prior_param > (mean_param - nsigma_range*err_param) )*(prior_param < (mean_param + nsigma_range*err_param) )
   
   if np.sum( 1 - indx_cut_param ) > 0.:
      return np.inf

   delw = 0.05
   delz = 20.
   w_Range = 40
   zRangeLarge = 2600.
   zRangeMed = 300.
   zRangeRed = 260.
   nBootstrapSample = 20
   l_range_p = 200.
   l_range_n = -200.

   zspace = np.linspace(0., zRangeRed, int(zRangeRed/delz))
   zspaceMed = np.linspace(0., zRangeMed, int(zRangeMed/delz))
   zspaceFull  = np.linspace(-zRangeRed, zRangeRed, 2*int(zRangeRed/delz)-1)
   zspaceLarge  = np.linspace(-zRangeLarge, zRangeLarge, 2*int(zRangeLarge/delz)-1)
   wspace = np.linspace(0., w_Range, int(w_Range/delw)+1 )


   wfunct = (2./(np.sqrt(2.*np.pi*star_sigma**2)) )*np.exp(-0.5*(wspace/star_sigma)**2)
   
   z, zerr, evfs_w = zdata

   starDensity = fetchZDist_fast(z, zerr, evfs_w, zspaceFull, zSun = sunZ)
   SDDensity = np.sqrt(fetchZDist_fast(z, zerr, evfs_w, zspaceFull, zSun = sunZ, use_evfs = False))
   for i, sd in enumerate(SDDensity):
      if sd == 0:
         SDDensity[i] = 1.
      else:
         SDDensity[i] = starDensity[i]/sd + systematic_z*starDensity[i]
   starDensity_norm = starDensity[int(np.ceil(len(starDensity)/2.-1))]
   #parallax_error = an.fetch_z_systematics(starFile, zspaceFull, starDensity)
   #parallax_error = 0.
   starDensity = starDensity/starDensity_norm
   starDensity_Delta = SDDensity/starDensity_norm

   solu = an.PoissonJeansSolve_gaia(h_DD, sig_DD, zRangeLarge, use_default_density = False, sigma_in = sigma, rho_in = rho)

   predict = an.fetchZPredict(solu, zspaceMed, zspaceFull, wfunct, wspace, show_plot=False)
   SDPredict = [0. for k in range(len(starDensity_Delta))]
   predict_norm = predict[int(np.ceil(len(predict)/2.-1))]
   predict = predict/predict_norm
   predict_Delta = SDPredict/predict_norm
   
   likelihood = an.likelihoodDensity(zspaceFull, predict, starDensity, predict_Delta, starDensity_Delta, l_range_p, l_range_n, plot_dist = False)
   if not np.isfinite(likelihood):
      return np.inf
   
   #prior = lambda x, s: (1./(s*np.sqrt(2*np.pi)))*np.exp(-0.5*((x/s)**2))
   prior = lambda x, s: (1./(s*np.sqrt(2*np.pi)))*np.exp(-0.5*((x/s)**2))
   
   prior_prod = prior( (prior_param - mean_param) , err_param)
   
   valid_indx = 1 - (prior_prod > 0.)
   if np.sum(valid_indx) > 0.:
       return np.inf
   likelihood = likelihood - np.sum(np.log( prior_prod ))
   
   return likelihood



#-------------------------------------------------------------------------------------------------------
def loglikelihood_emcee(param, h_DD, sig_DD, zdata, mean_param=0., err_param=0.):
   systematic_z = 0.03
   nsigma_range = 3.
   sigmaDHalo = float('nan')

   star_sig, sunZ, rhoDHalo, sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants, sigmaMV2p5, sigma3MV4,\
   sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs, rhoH2, rhoHI1, rhoHI2, rhoWGas, rhoGiants, rhoMV2p5,\
   rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs = param

   sigma = [sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants, sigmaMV2p5, sigma3MV4, sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs, sigmaDHalo]
   rho = [rhoH2, rhoHI1, rhoHI2, rhoWGas, rhoGiants, rhoMV2p5, rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs, rhoDHalo]
   star_sigma = star_sig

   prior_param = np.array([star_sig, rhoDHalo, sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants, sigmaMV2p5, sigma3MV4, \
   sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs, rhoH2, rhoHI1, rhoHI2, rhoWGas, rhoGiants, rhoMV2p5,\
   rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs])

   if np.sum(sigma <= 0.)+np.sum(rho <= 0.)+(star_sigma <= 0.)+(rhoDHalo <= 0.) > 0.:
      return -np.inf
   indx_cut_param = (prior_param > (mean_param - nsigma_range*err_param) )*(prior_param < (mean_param + nsigma_range*err_param) )
   if np.sum( 1-indx_cut_param ) > 0.:
      return -np.inf


   delw = 0.05
   delz = 20.
   w_Range = 40
   zRangeLarge = 2600.
   zRangeMed = 300.
   zRangeRed = 260.
   nBootstrapSample = 20
   l_range_p = 200.
   l_range_n = -200.

   zspace = np.linspace(0., zRangeRed, int(zRangeRed/delz))
   zspaceMed = np.linspace(0., zRangeMed, int(zRangeMed/delz))
   zspaceFull  = np.linspace(-zRangeRed, zRangeRed, 2*int(zRangeRed/delz)-1)
   zspaceLarge  = np.linspace(-zRangeLarge, zRangeLarge, 2*int(zRangeLarge/delz)-1)
   wspace = np.linspace(0., w_Range, int(w_Range/delw)+1 )


   wfunct = (2./(np.sqrt(2.*np.pi*star_sigma**2)) )*np.exp(-0.5*(wspace/star_sigma)**2)
   
   z, zerr, evfs_w = zdata

   starDensity = fetchZDist_fast(z, zerr, evfs_w, zspaceFull, zSun = sunZ)
   SDDensity = np.sqrt(fetchZDist_fast(z, zerr, evfs_w, zspaceFull, zSun = sunZ, use_evfs = False))
   for i, sd in enumerate(SDDensity):
      if sd == 0:
         SDDensity[i] = 1.
      else:
         SDDensity[i] = starDensity[i]/sd + systematic_z*starDensity[i]
   starDensity_norm = starDensity[int(np.ceil(len(starDensity)/2.-1))]
   #parallax_error = an.fetch_z_systematics(starFile, zspaceFull, starDensity)
   #parallax_error = 0.
   starDensity = starDensity/starDensity_norm
   starDensity_Delta = SDDensity/starDensity_norm

   solu = an.PoissonJeansSolve_gaia(h_DD, sig_DD, zRangeLarge, use_default_density = False, sigma_in = sigma, rho_in = rho)

   predict = an.fetchZPredict(solu, zspaceMed, zspaceFull, wfunct, wspace, show_plot=False)
   SDPredict = [0. for k in range(len(starDensity_Delta))]
   predict_norm = predict[int(np.ceil(len(predict)/2.-1))]
   predict = predict/predict_norm
   predict_Delta = SDPredict/predict_norm
   
   likelihood = an.likelihoodDensity(zspaceFull, predict, starDensity, predict_Delta, starDensity_Delta, l_range_p, l_range_n, plot_dist = False)
   if not np.isfinite(likelihood):
      return -np.inf

   
   prior = lambda x, s: (1./(s*np.sqrt(2*np.pi)))*np.exp(-0.5*((x/s)**2))
   
   prior_prod = prior( (prior_param - mean_param) , err_param)
   
   valid_indx = 1 - (prior_prod > 0.)
   if np.sum(valid_indx) > 0.:
       return -np.inf
   likelihood = likelihood - np.sum(np.log( prior_prod ))
   
   return (-1.*likelihood)







