import numpy as np
import scipy.integrate as nint
import scipy.interpolate as interp
import scipy.stats as stats
import scipy.optimize as opt
import matplotlib.pyplot as plt
#from tqdm import tqdm
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

#-------------------------------------------------------------------------------------------------------

def fetchZDist_fast(z, zerr, evfs_w_in, zspace, zSun = 0., use_evfs = True, verbose = False):

   evfs_w = evfs_w_in
   densityWidth = np.sqrt(np.mean(zerr))**2#. +(len(zspace) * 3./4.)**(-2./5.))
   zspacing = np.mean( (np.roll(zspace,-1)-zspace)[:-1] )

   if not use_evfs:
      evfs_w = np.ones_like(z)
   
   z = z - zSun
   
   if verbose: 
      print("mean z position: " + str(np.sum(z*evfs_w)/np.sum(evfs_w)) )      

   zdist = np.zeros_like(zspace)
   for i, z_coord in enumerate(zspace):
      z_in_bin = (z > (z_coord - zspacing/2.))*(z <= (z_coord + zspacing/2.))
      zdist[i] = np.sum(evfs_w[z_in_bin])
  
   return zdist

#-------------------------------------------------------------------------------------------------------
def loglikelihood_trueW(param, h_DD, sig_DD, zdata, wdata, wSun, logic, mean_param=0., err_param=0., linear_param=0., linear_width=0., del_w_cont = 0.0005, starCat = "None" , w_file_name = None, print_data = False):

   delw = del_w_cont 
   nsigma_range = 3.
   systematic_z = 0.03
   sigmaDHalo = float('nan')
   llh_infty = False

   epsilon = 1e-4
   
   load_ext_w_err, save_w_err, return_disk_param = logic
   if load_ext_w_err:
      wspace, wfunct, ext_w_err = wdata

   star_norm, sunZ, rhoDHalo, sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants, sigmaMV2p5, sigma3MV4,\
   sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs, rhoH2, rhoHI1, rhoHI2, rhoWGas, rhoGiants, rhoMV2p5,\
   rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs = param

   sigma = [sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants, sigmaMV2p5, sigma3MV4, sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs, sigmaDHalo]
   rho = [rhoH2, rhoHI1, rhoHI2, rhoWGas, rhoGiants, rhoMV2p5, rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs, rhoDHalo]

   prior_param = np.array([sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants, sigmaMV2p5, sigma3MV4, \
   sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs, rhoH2, rhoHI1, rhoHI2, rhoWGas, rhoGiants, rhoMV2p5,\
   rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs])
   
   prior_lin_param = np.array([sunZ, rhoDHalo])

   indx_cut_param = (prior_param > (mean_param - nsigma_range*err_param) )*(prior_param < (mean_param + nsigma_range*err_param) ) 
   indx_cut_lin_param = (prior_lin_param > (linear_param - linear_width) )*(prior_lin_param < (linear_param + linear_width) )

   if (h_DD < epsilon) or (sig_DD < epsilon) or (star_norm < epsilon) or (rhoDHalo < epsilon):
      llh_infty = True
   if ( np.amin(sigma)< epsilon ) or ( np.amin(rho)< epsilon ):
      llh_infty = True
   if (np.sum( 1 - indx_cut_param ) + np.sum(1- indx_cut_lin_param) ) > 0.:
      llh_infty = True
      
   if llh_infty and (not return_disk_param):
      return np.inf
      
   delz = 20.
   w_Range = 45.
   zRangeLarge = 2600.
   zRangeMed = 300.
   zRangeRed = 260.
   nBootstrapSample = 20
   l_range_p = 200.
   l_range_n = -200.

   zspace = np.linspace(0., zRangeRed, int(zRangeRed/delz)+1)
   zspaceMed = np.linspace(0., zRangeMed, int(zRangeMed/delz)+1)
   zspaceFull  = np.linspace(-zRangeRed, zRangeRed, 2*int(zRangeRed/delz)+1)

   zspaceLarge  = np.linspace(-zRangeLarge, zRangeLarge, 2*int(zRangeLarge/delz)-1)
   if not load_ext_w_err:
      wspace = np.linspace(0., w_Range, int(w_Range/delw)+1 )

   z, zerr, evfs_w = zdata

   starDensity = fetchZDist_fast(z, zerr, evfs_w, zspaceFull, zSun = sunZ)
   SDDensity = np.sqrt(fetchZDist_fast(z, zerr, evfs_w, zspaceFull, zSun = sunZ, use_evfs = False))
   for i, sd in enumerate(SDDensity):
      if sd == 0:
         SDDensity[i] = 1.
      else:
         SDDensity[i] = np.sqrt( (starDensity[i]/sd)**2 + (systematic_z*starDensity[i])**2 )
   starDensity_norm = starDensity[int(np.ceil(len(starDensity)/2.-1))]

   starDensity = starDensity/starDensity_norm
   starDensity_Delta = SDDensity/starDensity_norm

   solu = an.PoissonJeansSolve_gaia(h_DD, sig_DD, zRangeLarge, use_default_density = False, sigma_in = sigma, rho_in = rho)
   if return_disk_param:
      hdd_return, Sigdd_return = an.diskParamReturn(solu, h_DD, sig_DD, use_default_density = False, sigma_in = sigma, rho_in = rho)
   if not load_ext_w_err:
      predict, SDPredict = an.fetchZPredict_bootstrap(solu, zspaceMed, zspaceFull, wdata, wSun, wspace_in = wspace, to_save_file = save_w_err, filename=w_file_name, show_plot=False)
   if load_ext_w_err:
      predict = an.fetchZPredict(solu, zspaceMed, zspaceFull, wfunct, wspace, show_plot=False)
      SDPredict = ext_w_err*predict
   predict_norm = star_norm*predict[int(np.ceil(len(predict)/2.-1))]
   predict = predict/predict_norm
   predict_Delta = SDPredict/predict_norm
   
   likelihood = an.likelihoodDensity(zspaceFull, predict, starDensity, predict_Delta, starDensity_Delta, l_range_p, l_range_n, plot_dist = False)
   
   if print_data:
      data_file_name = "grid3/"+starCat+"_hDD_"+str(h_DD)+"_sigDD_"+str(sig_DD)+".txt"
      file_header = "z_space, predict, predict_err, data, data_err"
      np.savetxt(data_file_name, np.transpose( np.array([zspaceFull, predict, predict_Delta, starDensity, starDensity_Delta, solu(zspaceFull) ]) ), delimiter=',',header=file_header, fmt='%10.5f')
   
   if not np.isfinite(likelihood):
      llh_infty = True
   
   prior = lambda x, s: (1./(s*np.sqrt(2*np.pi)))*np.exp(-0.5*((x/s)**2))
   #prior_lin = lambda x, s: (1./(s**2))*(s-np.abs(x))
   prior_lin = lambda x, s: 1.#(1./(s))
   
   prior_prod = prior( (prior_param - mean_param) , err_param)
   prior_lin_prod = prior_lin( (prior_lin_param - linear_param) , linear_width)
   
   valid_indx = 1 - (prior_prod > 0.)
   valid_lin_indx = 1 - (prior_lin_prod > 0.)
   if np.sum(valid_indx)+np.sum(valid_lin_indx) > 0.:
      llh_infty = True

   if llh_infty and (not return_disk_param):
      return np.inf
   if  llh_infty and return_disk_param:
      return (hdd_return, Sigdd_return, np.inf)
      
   likelihood = likelihood - np.sum(np.log( prior_prod )) - np.sum(np.log( prior_lin_prod ))
   
   if return_disk_param:
      return (hdd_return, Sigdd_return, likelihood)
       
   return likelihood
#-------------------------------------------------------------------------------------------------------

def loglikelihood_trueW_allstars(param, h_DD, sig_DD, zdata, wdata, wSun, logic, mean_param=0., err_param=0., linear_param=0., linear_width=0., del_w_cont = 0.01):
 
   delw = del_w_cont
    
   w_file_A = "grid/star_A_hDD_" + str(int(h_DD)) + "_w_data.txt"
   w_file_F = "grid/star_F_hDD_" + str(int(h_DD)) + "_w_data.txt"
   w_file_G = "grid/star_G_hDD_" + str(int(h_DD)) + "_w_data.txt"

   zdata_A = zdata[0]
   zdata_F = zdata[1]
   zdata_G = zdata[2]
   wdata_A = wdata[0]
   wdata_F = wdata[1]
   wdata_G = wdata[2]
   
   nsigma_range = 3.
   systematic_z = 0.03
   sigmaDHalo = float('nan')
   llh_infty = False
   
   load_ext_w_err, save_w_err, return_disk_param = logic
   if load_ext_w_err:
      wspace_A, wfunct_A, ext_w_err_A = wdata_A
      wspace_F, wfunct_F, ext_w_err_F = wdata_F
      wspace_G, wfunct_G, ext_w_err_G = wdata_G


   star_norm_A, star_norm_F, star_norm_G, sunZ, rhoDHalo, sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants, sigmaMV2p5, sigma3MV4,\
   sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs, rhoH2, rhoHI1, rhoHI2, rhoWGas, rhoGiants, rhoMV2p5,\
   rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs = param

   sigma = [sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants, sigmaMV2p5, sigma3MV4, sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs, sigmaDHalo]
   rho = [rhoH2, rhoHI1, rhoHI2, rhoWGas, rhoGiants, rhoMV2p5, rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs, rhoDHalo]

   prior_param = np.array([sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants, sigmaMV2p5, sigma3MV4, \
   sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs, rhoH2, rhoHI1, rhoHI2, rhoWGas, rhoGiants, rhoMV2p5,\
   rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs])
   
   prior_lin_param = np.array([sunZ, rhoDHalo])

   if np.sum(star_norm_A <= 0.8)+np.sum(star_norm_A > 1.2) > 0.:
      llh_infty = True
   if np.sum(star_norm_F <= 0.8)+np.sum(star_norm_F > 1.2) > 0.:
      llh_infty = True
   if np.sum(star_norm_G <= 0.8)+np.sum(star_norm_G > 1.2) > 0.:
      llh_infty = True
   if np.sum(sigma <= 0.) + np.sum(rho <= 0.) > 0.:
      llh_infty = True
   if (rhoDHalo > 0.1)+(rhoDHalo <= 0.) > 0.:
      llh_infty = True
   if (sunZ > 30.)+(sunZ < -30.) > 0.:
      llh_infty = True

   indx_cut_param = (prior_param > (mean_param - nsigma_range*err_param) )*(prior_param < (mean_param + nsigma_range*err_param) ) 
   indx_cut_lin_param = (prior_lin_param > (linear_param - linear_width) )*(prior_lin_param < (linear_param + linear_width) )
   if (np.sum( 1 - indx_cut_param ) + np.sum(1- indx_cut_lin_param) ) > 0.:
      llh_infty = True
      
   if llh_infty and (not return_disk_param):
      return np.inf
      
   delz = 20.
   w_Range = 45.
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

   z_A, zerr_A, evfs_w_A = zdata_A
   z_F, zerr_F, evfs_w_F = zdata_F
   z_G, zerr_G, evfs_w_G = zdata_G

   starDensity_A = fetchZDist_fast(z_A, zerr_A, evfs_w_A, zspaceFull, zSun = sunZ)
   starDensity_F = fetchZDist_fast(z_F, zerr_F, evfs_w_F, zspaceFull, zSun = sunZ)
   starDensity_G = fetchZDist_fast(z_G, zerr_G, evfs_w_G, zspaceFull, zSun = sunZ)
   SDDensity_A = np.sqrt(fetchZDist_fast(z_A, zerr_A, evfs_w_A, zspaceFull, zSun = sunZ, use_evfs = False))
   SDDensity_F = np.sqrt(fetchZDist_fast(z_F, zerr_F, evfs_w_F, zspaceFull, zSun = sunZ, use_evfs = False))
   SDDensity_G = np.sqrt(fetchZDist_fast(z_G, zerr_G, evfs_w_G, zspaceFull, zSun = sunZ, use_evfs = False))
   for i, sd in enumerate(SDDensity_A):
      if sd == 0:
         SDDensity_A[i] = 1.
      else:
         SDDensity_A[i] = np.sqrt( (starDensity_A[i]/sd)**2 + (systematic_z*starDensity_A[i])**2 )
   starDensity_norm_A = starDensity_A[int(np.ceil(len(starDensity_A)/2.-1))]
   starDensity_A = starDensity_A/starDensity_norm_A
   starDensity_Delta_A = SDDensity_A/starDensity_norm_A
   
   for i, sd in enumerate(SDDensity_F):
      if sd == 0:
         SDDensity_F[i] = 1.
      else:
         SDDensity_F[i] = np.sqrt( (starDensity_F[i]/sd)**2 + (systematic_z*starDensity_F[i])**2 )
   starDensity_norm_F = starDensity_F[int(np.ceil(len(starDensity_F)/2.-1))]
   starDensity_F = starDensity_F/starDensity_norm_F
   starDensity_Delta_F = SDDensity_F/starDensity_norm_F
   
   for i, sd in enumerate(SDDensity_G):
      if sd == 0:
         SDDensity_G[i] = 1.
      else:
         SDDensity_G[i] = np.sqrt( (starDensity_G[i]/sd)**2 + (systematic_z*starDensity_G[i])**2 )
   starDensity_norm_G = starDensity_G[int(np.ceil(len(starDensity_G)/2.-1))]
   starDensity_G = starDensity_G/starDensity_norm_G
   starDensity_Delta_G = SDDensity_G/starDensity_norm_G
   
   
   solu = an.PoissonJeansSolve_gaia(h_DD, sig_DD, zRangeLarge, use_default_density = False, sigma_in = sigma, rho_in = rho)
   if return_disk_param:
      hdd_return, Sigdd_return = an.diskParamReturn(solu, h_DD, sig_DD, use_default_density = False, sigma_in = sigma, rho_in = rho)

   if not load_ext_w_err:
      predict_A, SDPredict_A = an.fetchZPredict_bootstrap(solu, zspaceMed, zspaceFull, wdata_A, wSun, wspace_in = wspace, to_save_file = save_w_err, filename=w_file_A, show_plot=False)
      predict_F, SDPredict_F = an.fetchZPredict_bootstrap(solu, zspaceMed, zspaceFull, wdata_F, wSun, wspace_in = wspace, to_save_file = save_w_err, filename=w_file_F, show_plot=False)
      predict_G, SDPredict_G = an.fetchZPredict_bootstrap(solu, zspaceMed, zspaceFull, wdata_G, wSun, wspace_in = wspace, to_save_file = save_w_err, filename=w_file_G, show_plot=False)

   if load_ext_w_err:
      predict_A = an.fetchZPredict(solu, zspaceMed, zspaceFull, wfunct_A, wspace_A, show_plot=False)
      SDPredict_A = ext_w_err_A*predict_A
      predict_F = an.fetchZPredict(solu, zspaceMed, zspaceFull, wfunct_F, wspace_F, show_plot=False)
      SDPredict_F = ext_w_err_F*predict_F
      predict_G = an.fetchZPredict(solu, zspaceMed, zspaceFull, wfunct_G, wspace_G, show_plot=False)
      SDPredict_G = ext_w_err_G*predict_G

   predict_norm_A = star_norm_A*predict_A[int(np.ceil(len(predict_A)/2.-1))]
   predict_A = predict_A/predict_norm_A
   predict_Delta_A = SDPredict_A/predict_norm_A
   predict_norm_F = star_norm_F*predict_F[int(np.ceil(len(predict_F)/2.-1))]
   predict_F = predict_F/predict_norm_F
   predict_Delta_F = SDPredict_F/predict_norm_F
   predict_norm_G = star_norm_G*predict_G[int(np.ceil(len(predict_G)/2.-1))]
   predict_G = predict_G/predict_norm_G
   predict_Delta_G = SDPredict_G/predict_norm_G
   
   likelihood_A = an.likelihoodDensity(zspaceFull, predict_A, starDensity_A, predict_Delta_A, starDensity_Delta_A, l_range_p, l_range_n, plot_dist = False)
   likelihood_F = an.likelihoodDensity(zspaceFull, predict_F, starDensity_F, predict_Delta_F, starDensity_Delta_F, l_range_p, l_range_n, plot_dist = False)
   likelihood_G = an.likelihoodDensity(zspaceFull, predict_G, starDensity_G, predict_Delta_G, starDensity_Delta_G, l_range_p, l_range_n, plot_dist = False)
   likelihood = likelihood_A + likelihood_F + likelihood_G
   if not np.isfinite(likelihood):
      llh_infty = True
   
   prior = lambda x, s: (1./(s*np.sqrt(2*np.pi)))*np.exp(-0.5*((x/s)**2))
   #prior_lin = lambda x, s: (1./(s**2))*(s-np.abs(x))
   prior_lin = lambda x, s: (1./(s))
   
   prior_prod = prior( (prior_param - mean_param) , err_param)
   prior_lin_prod = prior_lin( (prior_lin_param - linear_param) , linear_width)
   
   valid_indx = 1 - (prior_prod > 0.)
   valid_lin_indx = 1 - (prior_lin_prod > 0.)
   if np.sum(valid_indx)+np.sum(valid_lin_indx) > 0.:
      llh_infty = True

   if llh_infty and (not return_disk_param):
      return np.inf
   if  llh_infty and return_disk_param:
      return (hdd_return, Sigdd_return, np.inf)
      
   likelihood = likelihood - np.sum(np.log( prior_prod )) - np.sum(np.log( prior_lin_prod ))
   
   if return_disk_param:
      return (hdd_return, Sigdd_return, likelihood)
       
   return likelihood


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
def loglikelihood_trueW_global(param, h_DD, zdata, wdata, wSun, logic, mean_param=0., err_param=0., linear_param=0., linear_width=0., del_w_cont = 0.01, starCat = "None", w_file_name = "None"):
   sig_DD, star_norm, sunZ, rhoDHalo, sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants, sigmaMV2p5, sigma3MV4,\
   sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs, rhoH2, rhoHI1, rhoHI2, rhoWGas, rhoGiants, rhoMV2p5,\
   rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs = param
   
   load_ext_w_err, save_w_err, return_disk_param = logic
   
   if sig_DD <= 0.:
      if not return_disk_param:
         return np.inf 
      
   short_param = star_norm, sunZ, rhoDHalo, sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants, sigmaMV2p5, sigma3MV4,\
   sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs, rhoH2, rhoHI1, rhoHI2, rhoWGas, rhoGiants, rhoMV2p5,\
   rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs
   
   output = loglikelihood_trueW(short_param, h_DD, sig_DD, zdata, wdata, wSun, logic, mean_param=mean_param, err_param=err_param, linear_param=linear_param, linear_width=linear_width, del_w_cont = 0.01, starCat = starCat, w_file_name = w_file_name)
   return output

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
def loglikelihood_trueW_global_allstars(param, h_DD, zdata, wdata, wSun, logic, mean_param=0., err_param=0., linear_param=0., linear_width=0., del_w_cont = 0.01):
   sig_DD, star_norm_A, star_norm_F, star_norm_G, sunZ, rhoDHalo, sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants, sigmaMV2p5, sigma3MV4,\
   sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs, rhoH2, rhoHI1, rhoHI2, rhoWGas, rhoGiants, rhoMV2p5,\
   rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs = param
   
   load_ext_w_err, save_w_err, return_disk_param = logic
   
   if sig_DD <= 0.:
      if not return_disk_param:
         return np.inf 
      
   short_param = star_norm_A, star_norm_F, star_norm_G, sunZ, rhoDHalo, sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants, sigmaMV2p5, sigma3MV4,\
   sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs, rhoH2, rhoHI1, rhoHI2, rhoWGas, rhoGiants, rhoMV2p5,\
   rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs
   
   output = loglikelihood_trueW_allstars(short_param, h_DD, sig_DD, zdata, wdata, wSun, logic, mean_param=mean_param, err_param=err_param, linear_param=linear_param, linear_width=linear_width, del_w_cont = 0.01)
   return output

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
def loglikelihood_emcee(param, zdata, wdata, wSun, logic, mean_param, err_param, linear_param=0., linear_width=0., del_w_cont = 0.01, starCat = "None" , w_file_name = None):
   h_DD, sig_DD, star_norm, sunZ, rhoDHalo, sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants, sigmaMV2p5, sigma3MV4,\
   sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs, rhoH2, rhoHI1, rhoHI2, rhoWGas, rhoGiants, rhoMV2p5,\
   rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs = param
   
   short_param = star_norm, sunZ, rhoDHalo, sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants, sigmaMV2p5, sigma3MV4,\
   sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs, rhoH2, rhoHI1, rhoHI2, rhoWGas, rhoGiants, rhoMV2p5,\
   rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs
   
   llh = -1.*loglikelihood_trueW(short_param, h_DD, sig_DD, zdata, wdata, wSun, logic, mean_param, err_param, linear_param, linear_width, del_w_cont, starCat, w_file_name)
   return llh



