#========================================================
#
#     Project: Dark Disk
#
#     Desciption : Generate Exclusion Plot
#
#     Author : John Leung
#
#========================================================


import numpy as np
import scipy.integrate as nint
import scipy.interpolate as interp
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import integrator as intt
import analysis as an


gridFile = '/Users/john/Desktop/john_code/DarkDisk/gridFStarCorr.txt'
raw_gridFile = '/Users/john/Desktop/john_code/DarkDisk/gridFStarCorr_raw.txt'
starFile = '/Users/john/Desktop/john_code/DarkDisk/F_stars.txt'
starWFile = '/Users/john/Desktop/john_code/DarkDisk/F_stars_w_midplane.txt'
delw = 0.01
delz = 5.
zRangeLarge = 2600
zRangeMed = 300
zRangeRed = 260
nBootstrapSample = 20
nStars = len(np.loadtxt(starFile, delimiter= ",", skiprows=1, usecols=(0), unpack=True))

zspace = np.linspace(0, zRangeRed, (zRangeRed/delz))
zspaceMed = np.linspace(0, zRangeMed, (zRangeMed/delz))
zspaceFull  = np.linspace(-zRangeRed, zRangeRed, 2*(zRangeRed/delz)-1)
zspaceLarge  = np.linspace(-zRangeLarge, zRangeLarge, 2*(zRangeLarge/delz)-1)
l_range_p = 220.
l_range_n = -220.
ndf = np.floor((l_range_p-l_range_n)/(zspaceFull[1]-zspaceFull[0]))-1

wspace, wfunct = an.fetchWDist_gaia(starWFile)
starDensity = an.fetchZDistHist(starFile, zspaceFull, use_evfs = True, show_plot = False)
#SDDensity = an.bootstrap(starDensity,zspaceFull, nStars, nBootstrapSample)
SDDensity = np.sqrt(an.fetchZDistHist(starFile, zspaceFull, use_evfs = False))
for i, sd in enumerate(SDDensity):
   if sd == 0:
      SDDensity[i] = 1.
   else:
      SDDensity[i] = starDensity[i]/sd
starDensity_norm = starDensity[int(np.ceil(len(starDensity)/2.))]
#starDensity_norm = np.trapz(starDensity,zspaceFull)
starDensity = starDensity/starDensity_norm
starDensity_Delta = SDDensity/starDensity_norm

#print('Stars read:', nStars)
#print(starDensity[int(np.ceil(len(starDensity)/2.))])
print((str("ndf is: "), ndf))


interpz = []
interpy = []
interpx = []
interpxy = []

zero_solu = an.PoissonJeansSolve(0., 0., zRangeLarge)
zero_predict = an.fetchZPredict(zero_solu, zspaceMed, zspaceFull, wfunct, wspace, show_plot=False)
zero_SDPredict = [0. for k in range(len(starDensity_Delta))]
zero_predict_norm = zero_predict[int(np.ceil(len(zero_predict)/2.))]
zero_predict = zero_predict/zero_predict_norm
zero_predict_Delta = zero_SDPredict/zero_predict_norm
zero_likelihood = an.likelihoodDensity(zspaceFull, zero_predict, starDensity, zero_predict_Delta, starDensity_Delta, l_range_p, l_range_n, plot_dist = False)
zero_likelihood = zero_likelihood/ndf

HD_step = 5
SgmD_step = 1

for nHD in range(5,305,HD_step):
   param = an.diskParamReturn(0., nHD, 0.)
   interpx.append( float(param[1]) )
   interpy.append( float(param[0]) )
   interpxy.append( [float(param[1]),float(param[0])] )
   interpz.append( float(zero_likelihood) )

for nHD in tqdm(range(5,305,HD_step)):
   for nSgmD in tqdm(range(1,61,SgmD_step)):
      solu = an.PoissonJeansSolve(nHD, nSgmD, zRangeLarge)
      param = an.diskParamReturn(solu, nHD, nSgmD)
      predict = an.fetchZPredict(solu, zspaceMed, zspaceFull, wfunct, wspace, show_plot=False)
      #SDPredict = an.bootstrap(predict,zspaceFull, nStars, nBootstrapSample)
      SDPredict = [0. for k in range(len(starDensity_Delta))]
      predict_norm = predict[int(np.ceil(len(predict)/2.))]
      predict = predict/predict_norm
      predict_Delta = SDPredict/predict_norm
      likelihood = an.likelihoodDensity(zspaceFull, predict, starDensity, predict_Delta, starDensity_Delta, l_range_p, l_range_n, plot_dist = False)
      likelihood = likelihood/ndf
      #print(param[1])
      #print(param[0])
      #print(likelihood)
      interpx.append( float(param[1]) )
      interpy.append( float(param[0]) )
      interpxy.append( [float(param[1]),float(param[0])] )
      interpz.append( float(likelihood) )
      #print(("out: ", param[1], param[0], likelihood))

#interpx, interpy, interpz = np.loadtxt(raw_gridFile, delimiter= ",", skiprows=1, usecols=(0,1,2), unpack=True)
#f_likely = interp.interp2d(interpx,interpy,interpz, kind='linear')

#spacex = np.linspace(0,30,30)
#spacey = np.linspace(0,100,25)
spacex, spacey = np.mgrid[0:30:30j,0:100:25j]

f_likely = interp.griddata(np.array(interpxy), np.array(interpz), (spacex,spacey), method='nearest')

outputx = spacex[:,0]
outputy = spacey[0,:]
outputz = np.array([])
opt_y_coord = np.array([])

for i, y in enumerate(outputy):
   peak_f = np.amax( f_likely[:,i] )
   opt_y_coord = np.append(opt_y_coord, peak_f)

#optim_likely = np.transpose( np.array([optim_likely_y for j in range(len(outputx))] )  )

plot_z = np.empty_like(f_likely)

#print(("size: ", plot_z.shape, opt_y_coord.shape, plot_z.shape))
for i, x in enumerate(outputx):
   for j, y in enumerate(outputy):
      plot_z[i,j] = 2.*(f_likely[i,j] - opt_y_coord[j])
   #if np.abs(plot_z[j][i]) > 10:
   #   plot_z[j][i] = 0
   #print((plot_z[i,j], f_likely[i,j], opt_y_coord[i]))

"""
for i, x in enumerate(outputx):
   for j, y in enumerate(outputy):
      out_f = f_likely[i, j]
      peak_f = np.amax( f_likely[i,:] )
      outputz = np.append(outputz, 2.*(peak_f - out_f) )
      #print((out_f,peak_f))
   opt_y_coord = np.append(opt_y_coord, peak_f)

#optim_likely = np.transpose( np.array([optim_likely_y for j in range(len(outputx))] )  )

plot_z = np.empty_like(f_likely)

#print(("size: ", plot_z.shape, opt_y_coord.shape, plot_z.shape))
for i, x in enumerate(outputx):
   for j, y in enumerate(outputy):
      plot_z[i,j] = 2.*(f_likely[i,j] - opt_y_coord[i])
   #if np.abs(plot_z[j][i]) > 10:
   #   plot_z[j][i] = 0
   #print((plot_z[i,j], f_likely[i,j], opt_y_coord[i]))
"""
#plt.pcolor(spacex, spacey, plot_z, vmin=abs(plot_z).min(), vmax=abs(plot_z).max())

#plt.scatter(spacex,spacey,plot_z)
spacex = np.transpose(spacex)
spacey = np.transpose(spacey)
plot_z = np.transpose(plot_z)
plt.pcolor(spacey, spacex, plot_z, vmin=plot_z.min(), vmax=plot_z.max())
plt.colorbar()
plt.contour(spacey, spacex, plot_z, [-1.26])
plt.xlabel('h_D')
plt.ylabel('Sigma_D')
plt.show()


"""with open(gridFile, 'a') as out_file:
   for i,z in enumerate(outputz):
      out_file.write(str(outputx[i]) + ', ' + str(outputy[i]) + ', ' + str(z) + '\n')
      #raw_out = np.column_stack((interpx, interpy, interpz))
      #out = np.column_stack((outputx, outputy, z_out))"""

with open(raw_gridFile, 'a') as out_file2:
   for i,z in enumerate(interpz):
      out_file2.write(str(interpx[i]) + ', ' + str(interpy[i]) + ', ' + str(z) + '\n')
#np.savetxt(raw_gridFile, raw_out, delimiter=',')
#np.savetxt(gridFile, out , delimiter=',' ,fmt='%s')

