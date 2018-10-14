import numpy as np
import scipy.integrate as nint
import scipy.interpolate as interp
import scipy.stats as stats
import scipy.optimize as opt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import integrator as intt
import analysis as an
import likelihood as lh
from multiprocessing import cpu_count, Pool
#import emcee as mc
import sys

star_Cat_List = ["A","F","G"]

del_w_cont = 0.0002

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

sunZ_List = [3.721, -0.392, -0.326]
rhoDHalo = 0.03
sunW_List = [6.591, 7.589, 6.395]

density_norm = 1.0

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
sigmaDHalo = float('nan')

err_rhoDHalo = 0.005; err_star_sig = 0.28
err_sigmaH2 = 0.2; err_rhoH2 = 0.00312;
err_sigmaHI1 = 0.5; err_rhoHI1 = 0.00554;
err_sigmaHI2 = 2.4; err_rhoHI2 = 0.0007;
err_sigmaWGas = 4.0; err_rhoWGas = 0.00003;
err_sigmaGiants = 1.6; err_rhoGiants = 0.00006;
err_sigmaMV2p5 = 2.0; err_rhoMV2p5 = 0.00018;
err_sigma3MV4 = 2.4; err_rho3MV4 = 0.00018;
err_sigma4MV5 = 1.8; err_rho4MV5 = 0.00029;
err_sigma5MV8 = 1.9; err_rho5MV8 = 0.00072;
err_sigmaMV8 = 4.0; err_rhoMV8 = 0.0028;
err_sigmaWDwarfs = 5.0; err_rhoWDwarfs = 0.001;
err_sigmaBDwarfs = 5.0; err_rhoBDwarfs = 0.0005;

sigma_err = np.array([err_sigmaH2, err_sigmaHI1, err_sigmaHI2, err_sigmaWGas, err_sigmaGiants, err_sigmaMV2p5, err_sigma3MV4, err_sigma4MV5, err_sigma5MV8, err_sigmaMV8, err_sigmaWDwarfs, err_sigmaBDwarfs])
rho_err = np.array([err_rhoH2, err_rhoHI1, err_rhoHI2, err_rhoWGas, err_rhoGiants, err_rhoMV2p5, err_rho3MV4, err_rho4MV5, err_rho5MV8, err_rhoMV8, err_rhoWDwarfs, err_rhoBDwarfs])

sigma = np.array([sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants, sigmaMV2p5, sigma3MV4, sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs])
rho = np.array([rhoH2, rhoHI1, rhoHI2, rhoWGas, rhoGiants, rhoMV2p5, rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs])


def minimizer(f_args):
   llh_rett_opt, best_sgm, nSgmD, param, h_DD, zdata, wdata, sunW, logic_in, mean_param, err_param, mean_lin_param, width_lin_param, star_Cat, w_file_name = f_args
   print('Current Sgm step: ' + str(nSgmD) )
   load_ext_w_err, save_w_err, reture_disk_param = logic_in
   sig_DD = (nSgmD-0.5)
   reture_disk_param = False
   logic = load_ext_w_err, save_w_err, reture_disk_param
   result = opt.minimize(lh.loglikelihood_trueW, param, args=(h_DD, sig_DD, zdata, wdata, sunW, logic, mean_param, err_param, mean_lin_param, width_lin_param, del_w_cont, star_Cat, w_file_name), method='powell', options={'xtol':1e-4, 'ftol':1e-4})
   #result = opt.minimize(lh.loglikelihood_trueW, param, args=(h_DD, sig_DD, zdata, wdata, sunW, logic, mean_param, err_param, mean_lin_param, width_lin_param, star_Cat, w_file_name), bounds=bounds)
   x_return = result["x"]
   reture_disk_param = True
   logic = load_ext_w_err, save_w_err, reture_disk_param
   hdd_ret, Sigdd_ret, llh_rett = lh.loglikelihood_trueW(x_return, h_DD, sig_DD, zdata, wdata, sunW, logic, mean_param, err_param, mean_lin_param, width_lin_param, del_w_cont, star_Cat, w_file_name)
   print("likelihood: " + str( llh_rett ))
   return_sol = sig_DD, h_DD, llh_rett, llh_rett_opt, best_sgm
   return return_sol


def main():
   arg = sys.argv[1:]
   
   star_in = str(arg[0])
   nHD = float(arg[1])
   ncpu = int(arg[2])
   
   star_found = False
   for i, i_star in enumerate(star_Cat_List):
      if star_in == i_star:
         star_Cat = star_Cat_List[i]
         sunZ = sunZ_List[i]
         sunW = sunW_List[i]
         star_found = True
   if star_found == False:
      print("Argument does not represent a star category!")
      return -1
            
   starFile = star_Cat+'_stars.txt'
   w_file_name = "grid/star_" + star_Cat + "_hDD_" + str(int(nHD)) + "_w_data.txt"


   mean_param = np.append(sigma, rho)
   mean_lin_param = np.array([0., rhoDHalo])

   err_param = np.append(sigma_err, rho_err)
   width_lin_param = np.array([30., 0.03])

   dens_norm = 1.0

   param = dens_norm, sunZ, rhoDHalo, sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants,\
   sigmaMV2p5, sigma3MV4, sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs, rhoH2, rhoHI1, rhoHI2,\
   rhoWGas, rhoGiants, rhoMV2p5, rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs

   h_DD_test = 10.; sig_DD_test = 2.
   load_ext_w_err= False; save_w_err = True; reture_disk_param = False
   logic = load_ext_w_err, save_w_err, reture_disk_param

   wdata_full = np.loadtxt(starFile, delimiter= ",", skiprows=1, usecols=(zfile_columnl, zfile_columnb, zfile_columnPlx, zfile_columnPml, zfile_columnPmb, zfile_columnEVFS, zfile_columnZErr, zfile_columnVR), unpack=True)
   zdata = np.loadtxt(starFile, delimiter= ",", skiprows=1, usecols=(zfile_columnZ, zfile_columnZErr, zfile_columnEVFS), unpack=True)  
   llh_test = lh.loglikelihood_trueW(param, h_DD_test, sig_DD_test, zdata, wdata_full, sunW, logic, mean_param, err_param, mean_lin_param, width_lin_param, del_w_cont, starCat = star_Cat, w_file_name = w_file_name)

   print("llh is: " + str(llh_test))

   p = Pool(ncpu)
   print("CPU count is: " + str(ncpu))
 
   wfile_columnZCoord = 0
   wfile_columnCnt = 1
   wfile_columnErr = 2

   count, count_err = np.loadtxt(w_file_name, delimiter= ",", skiprows=1, usecols=(wfile_columnCnt, wfile_columnErr), unpack=True)
    
   w_Range = 40
   wspace = np.linspace(0., w_Range, int(w_Range/delw)+1 )
   w_binned_space, w_count, w_count_err = an.fetchWData(starFile, wSun = sunW)
   wspace, wfunct = an.fetchWDist_gaia(w_binned_space, w_count, w_count_err, verbose=True, wspace_out = wspace, show_plot=False, gaus_approx = False)
   wdata = wspace, wfunct, count_err/count

   epsilon = 0.0001
   bounds = [(0.8, 1.2), (-30., 30.), (epsilon, 0.06),\
            (epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None),\
            (epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None),\
            (epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None),\
            (epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None)]
   bounds_long = [(0.1, 40), (0.8,1.2), (-30., 30.), (epsilon, 0.06),\
            (epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None),\
            (epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None),\
            (epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None),\
            (epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None)]
            
   load_ext_w_err= True; save_w_err = False;  reture_disk_param = False
   SgmD_step = 2

   h_DD = (nHD-0.5)
   SgmD_trial = 1.0
 
   logic = load_ext_w_err, save_w_err, reture_disk_param

   param_long = SgmD_trial, dens_norm, sunZ, rhoDHalo, sigmaH2, sigmaHI1, sigmaHI2, sigmaWGas, sigmaGiants,\
   sigmaMV2p5, sigma3MV4, sigma4MV5, sigma5MV8, sigmaMV8, sigmaWDwarfs, sigmaBDwarfs, rhoH2, rhoHI1, rhoHI2,\
   rhoWGas, rhoGiants, rhoMV2p5, rho3MV4, rho4MV5, rho5MV8, rhoMV8, rhoWDwarfs, rhoBDwarfs
   result_sig_DD = opt.minimize(lh.loglikelihood_trueW_global, param_long, args=(h_DD, zdata, wdata, sunW, logic, mean_param, err_param, mean_lin_param, width_lin_param, del_w_cont, star_Cat, w_file_name ), method='powell', options={'xtol':1e-3, 'ftol':1e-4})


   reture_disk_param = True
   logic = load_ext_w_err, save_w_err, reture_disk_param
   x_return_opt = result_sig_DD["x"]
   hdd_ret_opt, Sigdd_ret_opt, llh_rett_opt = lh.loglikelihood_trueW_global(x_return_opt, h_DD, zdata, wdata, sunW, logic, mean_param, err_param, mean_lin_param, width_lin_param, del_w_cont, star_Cat, w_file_name)
   best_sgm = x_return_opt[0]

   f_args = [(llh_rett_opt, best_sgm, nSgmD, param, h_DD, zdata, wdata, sunW, logic, mean_param, err_param, mean_lin_param, width_lin_param, del_w_cont, star_Cat, w_file_name) for nSgmD in range(1,21,SgmD_step)]
   result_out = p.map(minimizer, f_args)

   line_header = "Sigma_DD, h_DD, loglikelihood, optimal llh, optimal sgm"
   file_name = "grid/grid_star_"+star_Cat+"_hDD_"+str(int(nHD))+".txt"
   np.savetxt(file_name , result_out, delimiter=',', header=line_header, fmt='%10.5f')
   print( "file output to: " + str(file_name) )
   p.close() 


if __name__ == "__main__":
   main()
   

