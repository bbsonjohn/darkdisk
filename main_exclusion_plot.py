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

gridFile = '/home/john/Programs/DarkDisk/gridAStarCorr.txt'
raw_gridFile = '/home/john/Programs/DarkDisk/gridAStarCorr_raw.txt'

starFile = '/home/john/Programs/DarkDisk/A_stars.txt'
starWFile = '/home/john/Programs/DarkDisk/A_stars_w_midplane.txt'
delw = 0.01
delz = 0.1
zRangeFull = 2600
zRangeRed = 260
nBootstrapSample = 20
nStars = len(np.loadtxt(starFile, delimiter= ",", skiprows=1, usecols=(0), unpack=True))

zspace = np.linspace(0, zRangeRed, (zRangeRed/delz))
zspaceFull  = np.linspace(-zRangeRed, zRangeRed, 2*(zRangeRed/delz)-1)
wspace, wfunct = an.fetchWDist_gaia(starWFile)
starDensity = an.fetchZDist(starFile,zspaceFull)
#SDDensity = an.bootstrap(starDensity,zspaceFull, nStars, nBootstrapSample)
SDDensity = np.sqrt(starDensity)
starDensity_norm = starDensity[int(np.ceil(len(starDensity)/2.))]
starDensity = starDensity/starDensity_norm
starDensity_Delta = SDDensity/starDensity_norm

print('Stars read:', nStars)

interpz = []
interpy = []
interpx = []

"""
for nHD in tqdm(range(5,305,30)):
	for nSgmD in tqdm(range(1,61,6)):
		solu = an.PoissonJeansSolve(nHD, nSgmD, 2600)
		param = an.diskParamReturn(solu, nHD, nSgmD)
		predict = an.fetchZPredict(solu[:,0], zspace, wfunct, wspace)
		#SDPredict = an.bootstrap(predict,zspaceFull, nStars, nBootstrapSample)
		SDPredict = [0. for k in range(len(starDensity_Delta))]
		predict_norm = predict[int(np.ceil(len(predict)/2.))]
		predict = predict/predict_norm
		predict_Delta = SDPredict/predict_norm
		likelihood = an.likelihoodDensity(zspaceFull, predict, starDensity,starDensity_Delta,predict_Delta)/((zspaceFull[1]-zspaceFull[0])*len(zspaceFull))
		#print(param[1])
		#print(param[0])
		#print(likelihood)
		interpx.append( float(param[1]) )
		interpy.append( float(param[0]) )
		interpz.append( float(likelihood) )
"""

interpx, interpy, interpz = np.loadtxt(raw_gridFile, delimiter= ",", skiprows=1, usecols=(0,1,2), unpack=True)
f_likely = interp.interp2d(interpx,interpy,interpz, kind='linear')

spacex = np.linspace(0,30,30)
spacey = np.linspace(0,100,25)

outputx = np.array([])
outputy = np.array([])
outputz = np.array([])
opt_y_coord = np.array([])


for i, x in enumerate(spacex):
	for j, y in enumerate(spacey):
		outputx = np.append(outputx, x)
		outputy = np.append(outputy, y)
		out_f = f_likely(x, y)
		peak_f = np.amax( f_likely(x, spacey) )
		outputz = np.append(outputz, 2.*(peak_f - out_f) )
		if peak_f == out_f:
			opt_y_coord = np.append(opt_y_coord, y)

#optim_likely = np.transpose( np.array([optim_likely_y for j in range(len(outputx))] )  )


plot_z_raw = f_likely(spacex[:-1], spacey[:-1])

plot_z = np.empty_like(plot_z_raw)

for i, x in enumerate(spacex[:-1]):
	for j, y in enumerate(spacey[:-1]):
		plot_z[j][i] = 2.*(plot_z_raw[j][i] - opt_y_coord[i])
		#if np.abs(plot_z[j][i]) > 10:
		#	plot_z[j][i] = 0
		print((plot_z[j][i], plot_z_raw[j][i], opt_y_coord[i]))

plt.pcolor(spacex, spacey, plot_z, vmin=abs(plot_z).min(), vmax=abs(plot_z).max())
plt.colorbar()
plt.show()


with open(gridFile, 'a') as out_file:
	for i,z in enumerate(outputz):
		out_file.write(str(outputx[i]) + ', ' + str(outputy[i]) + ', ' + str(z) + '\n')
		#raw_out = np.column_stack((interpx, interpy, interpz))
		#out = np.column_stack((outputx, outputy, z_out))
"""
with open(raw_gridFile, 'a') as out_file2:
	for i,z in enumerate(interpz):
		out_file2.write(str(interpx[i]) + ', ' + str(interpy[i]) + ', ' + str(z) + '\n')
"""
#np.savetxt(raw_gridFile, raw_out, delimiter=',')
#np.savetxt(gridFile, out , delimiter=',' ,fmt='%s')

