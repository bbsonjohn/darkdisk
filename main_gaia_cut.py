import sys
sys.path.insert(0, "/home/john/Downloads/mwdust-master")
sys.path.insert(0, "/home/john/Downloads/gaia_tools-master")
sys.path.insert(0, "/home/john/Downloads/tgas-completeness-master/py")
sys.path.insert(0, "/home/john/Downloads/isodist-master")

import os
import pylab
import numpy as np
import healpy
import astropy.coordinates as apco
from astropy.io import ascii
from astropy.table import Table, Column
#from apogee.util import localfehdist
import gaia_tools.load, gaia_tools.select
from gaia_tools import xmatch
from galpy.util import bovy_plot, bovy_coords, save_pickles, multi
from matplotlib.colors import LogNorm
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FuncFormatter, LogFormatter
from matplotlib import gridspec
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS
from astropy.coordinates import Galactic
from galpy.util.bovy_coords import cov_pmrapmdec_to_pmllpmbb

import pickle
import tqdm
from scipy import interpolate, optimize, integrate
import mwdust
import effsel
from effsel import main_sequence_cut_r, giant_sequence_cut
from matplotlib.colors import LogNorm
from matplotlib import gridspec
import matplotlib.lines as mlines
import copy
from isodist import Z2FEH, imf, PadovaIsochrone

min_plx= 0.45/0.2
max_dist = 1./min_plx
r_cyl_cut = 0.15
z_cyl_cut = 0.22

#-------------transform from proper motion in ICRS to proper motion in Galactic frame ------------------
def ProperMotionTransform(ra_coord, dec_coord, pmra_coord, pmdec_coord):

#	pmra_coord_cosdec = pmra_coord*np.cos(dec_coord*np.pi/180.)
	pmra_coord_cosdec = pmra_coord

	icrs = ICRS(ra=ra_coord*units.degree, dec=dec_coord*units.degree, pm_ra_cosdec=pmra_coord_cosdec*units.mas/units.yr, pm_dec=pmdec_coord*units.mas/units.yr)
	galactic = icrs.transform_to(Galactic)
	
	pml = (galactic.pm_l_cosb)/np.cos(galactic.b.radian)/(units.mas/units.yr)
	pmb = galactic.pm_b/(units.mas/units.yr)

	return [pml, pmb]

#-------------------------------------------------------------------------------------------------------
def prepare_plot(data, pltRange, nBin, weights = None, normed=False, positive=False, show_plot=False, xlabel=None, ylabel=None, title=None):

	bw = pltRange/nBin #bin width	
	bin_list = np.linspace(-pltRange-bw/2., pltRange+bw/2., nBin)  #list of bins; arange is used for integer bin values ==> histogram
	indx_data_cut = np.array( (data < (pltRange+bw))*(data > (-pltRange-bw)), dtype = bool)

	if positive == True:
		bin_list = np.linspace(0, pltRange+bw, nBin)
		indx_data_cut = np.array( (data < (pltRange+bw))*(data >= 0.), dtype = bool)

	data = data[indx_data_cut]

	try:
		weights = np.array(weights)
		weights = weights[indx_data_cut]
	except TypeError:
		weights = None
		print ("No weights applied.")

	data_n, bins, _ = plt.hist(data, bin_list, color='gray', histtype='stepfilled', alpha=0.3)
	data_err = np.sqrt(data_n)
	data_m, bins, _ = plt.hist(data, bin_list, weights=weights, color='gray', histtype='stepfilled', alpha=0.3)
	data_k = data_m
	if normed == True:
		bin_sep = (    np.roll((bin_list),-1) - (bin_list)    )[:-1]
		last_bin_extrapolate = bin_sep[-1]
		#bin_sep=np.append(bin_sep,last_bin_extrapolate)
		data_k=data_m/(np.sum(data_m*bin_sep))

	data_err = data_err*(data_k/data_n)

	plt.clf()
	mid = 0.5*(bins[1:] + bins[:-1])
	
	if show_plot == True:
		plt.grid()
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.plot(mid, data_k, 'r', label=star_Cat)
		plt.errorbar(mid, data_k, yerr=data_err, fmt='none')
		plt.show()
	return (data_k, data_err, mid)


#

def plot_hist(data, err, bins, color=None, marker=None, normed = True, plotLabel = None, xlabel = None, ylabel = None, title = None):

	for i in range( len(data) ):

		bin_sep = (    np.roll((bins[i]),-1) - (bins[i])    )[:-1]
		last_bin_extrapolate = bin_sep[-1]
		bin_sep=np.append(bin_sep,last_bin_extrapolate)

		norm = 1./np.sum(bin_sep*data[i])

		if normed == False:
			norm = 1.

		if marker == None:
			marker = np.array( 'o' for k in range(len(data)) ) 

		if color == None:
			color = np.array( None for k in range(len(data)) )

		plt.plot(bins[i], norm*data[i],  color=color[i], marker=marker[i] , label='plot')
		plt.errorbar(bins[i], norm*data[i], yerr=norm*err[i], fmt='none')

	plt.legend(plotLabel)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.show()
	return
#-------------------------------------------------------------------------------------------------------
def cyl_vol_func(X,Y,Z,xymin=0.,xymax=0.15,zmin=0.05,zmax=0.15):
    """A function that bins in cylindrical annuli around the Sun"""
    xy= np.sqrt(X**2.+Y**2.)
    out= np.zeros_like(X)
    out[(xy >= xymin)*(xy < xymax)*(Z >= zmin)*(Z < zmax)]= 1.
    return out
#-------------------------------------------------------------------------------------------------------
def is_good_relplx(mj):
    out= np.empty_like(mj)
    out[mj > 5.]= 20.
    out[mj < 0.]= 10.
    out[(mj >= 0)*(mj <= 5.)]= 20.+2.*(mj[(mj >= 0)*(mj <= 5.)]-5.)
    return out

#-------------------------------------------------------------------------------------------------------
def find_jk_boundaries(starCategory, sp):

	sp_stars = lambda star_Cat: np.array([( str(star_Cat) in s) for s in sp['SpT'] ], dtype='bool')
	jk_color_cut_upper = float("nan")
	jk_color_cut_lower = float("nan")

	# First consider the input as main classes
	if starCategory == "A":
		jk_color_cut_upper = (  (sp['JH']+sp['HK'])[ sp_stars('B9V') ] + (sp['JH']+sp['HK'])[ sp_stars('A0V') ]  )/2
		jk_color_cut_lower = (  (sp['JH']+sp['HK'])[ sp_stars('A9V') ] + (sp['JH']+sp['HK'])[ sp_stars('F0V') ]  )/2
	elif starCategory == "F":
		jk_color_cut_upper = (  (sp['JH']+sp['HK'])[ sp_stars('A9V') ] + (sp['JH']+sp['HK'])[ sp_stars('F0V') ]  )/2
		jk_color_cut_lower = (  (sp['JH']+sp['HK'])[ sp_stars('F9V') ] + (sp['JH']+sp['HK'])[ sp_stars('G0V') ]  )/2
	elif starCategory == "G":
		jk_color_cut_upper = (  (sp['JH']+sp['HK'])[ sp_stars('F9V') ] + (sp['JH']+sp['HK'])[ sp_stars('G0V') ]  )/2 
		jk_color_cut_lower = (  (sp['JH']+sp['HK'])[ sp_stars('G4V') ] + (sp['JH']+sp['HK'])[ sp_stars('G5V') ]  )/2
	elif starCategory == "All":
		jk_color_cut_upper = (  (sp['JH']+sp['HK'])[ sp_stars('B9V') ] + (sp['JH']+sp['HK'])[ sp_stars('A0V') ]  )/2
		jk_color_cut_lower = (  (sp['JH']+sp['HK'])[ sp_stars('G4V') ] + (sp['JH']+sp['HK'])[ sp_stars('G5V') ]  )/2		
	else : # then consider the input as a subclass or a list of adjacent subclasses
		for list_i, bo_val in enumerate(sp_stars(starCategory)):
			if bo_val == True:
				jk_color_cut_upper = (  (sp['JH']+sp['HK'])[list_i-1] + (sp['JH']+sp['HK'])[list_i]  )/2
		for list_i, bo_val in enumerate( reversed(sp_stars(starCategory)) ):
			if bo_val == True:
				jk_color_cut_lower = (  (sp['JH']+sp['HK'])[len(sp)-list_i] + (sp['JH']+sp['HK'])[len(sp)-list_i-1]  )/2

	return (jk_color_cut_upper, jk_color_cut_lower)
#---------------------------------------------------------------------------------------------------------------------
def generate_evfs(mj_tight, jk_tight, spt, zRange = 0.2, zNBin = 8, nintt_step = None, filename="default_evfs.txt" , save=True):

	zWidth = zRange/zNBin
	zBins = np.array([])
	evfs = np.array([])

	tesf= gaia_tools.select.tgasEffectiveSelect(tsf,dmap3d=mwdust.Zero(),MJ=mj_tight,JK=jk_tight,maxd=max_dist)
	if nintt_step == None:
		reduction_factor = 4. #my pc cannot handle the Bovy's optimized steps
		nintt_step = (2501*('A' in spt) + 1001 * (True-('A' in spt)))/reduction_factor
	
	for i in range(-zNBin, zNBin):
		i_step = i*zWidth

		zmin = i_step-zWidth/2.
		zmax = i_step+zWidth/2.
		zBins = np.append(zBins, i_step)
		
		evfs = np.append(evfs, tesf.volume(lambda x,y,z: cyl_vol_func(x,y,z,xymax=r_cyl_cut,zmin=zmin,zmax=zmax), ndists=nintt_step,xyz=True,relative=True)   )

	if save == True:
		np.savetxt(filename, np.transpose( np.array([zBins, evfs]) ), delimiter=',',header="z_Bin_center, rel_effective_vol", fmt='%10.5f')
		
	return (zBins, evfs)

#---------------------------------------------------------------------------------------------------------------------

def cut_indx_vol(tgas):

	XYZ= bovy_coords.lbd_to_XYZ(tgas['l'],tgas['b'],1./tgas['parallax'],degree=True)
	r_cyl = np.sqrt(XYZ[:,0]**2.+XYZ[:,1]**2.)
	z_cyl= XYZ[:,2]

	return  (r_cyl < r_cyl_cut)*(np.abs(z_cyl) < z_cyl_cut)

#---------------------------------------------------------------------------------------------------------------------

def cut_flow(data, cuts):
	for i, cut_indx in enumerate(cuts):
		data = data[cut_indx]
	return data

def cut_general(tgas, mj, jk):
	stat_indx = tsf.determine_statistical(tgas,twomass['j_mag'],twomass['k_mag'])
	tgas=tgas[stat_indx]
	mj=mj[stat_indx]
	jk=jk[stat_indx]
#	good_plx_indx = (tgas['parallax']/tgas['parallax_error'] > (is_good_relplx(mj)))*(jk != 0.)*(tgas['parallax'] > min_plx)

	return [stat_indx]
	

def cut_evfs(starCat, sp, jk, mj, tgas):
	tightness = True
	cut_indx = []

	jk_color_cut_upper, jk_color_cut_lower = find_jk_boundaries(starCat, sp)
	color_cut =  (jk > jk_color_cut_upper)*(jk < jk_color_cut_lower)*(tgas['parallax']/tgas['parallax_error'] > (is_good_relplx(mj)))*(jk != 0.)

	jk_tight = jk[color_cut]
	mj_tight = mj[color_cut]

	good_mj_cut_tight = (mj_tight > main_sequence_cut_r(jk_tight,tight=tightness))*(mj_tight < main_sequence_cut_r(jk_tight,low=True,tight=tightness))

	if np.sum(good_mj_cut_tight) < 50:
		good_mj_cut_tight = [True for k in range(len(good_mj_cut_tight))]

	return [color_cut, good_mj_cut_tight]

def cut_stars(starCat, sp, jk, mj, tgas): # category cut and volume cut
	tightness = False
	cut_indx = []

	jk_color_cut_upper, jk_color_cut_lower = find_jk_boundaries(starCat, sp)
	color_cut =  (jk > jk_color_cut_upper)*(jk < jk_color_cut_lower)*(tgas['parallax']/tgas['parallax_error'] > (is_good_relplx(mj)))*(jk != 0.)

	jk = jk[color_cut]
	mj = mj[color_cut]
	tgas = tgas[color_cut]

	good_mj_cut = ( mj > main_sequence_cut_r(jk,tight=tightness))*(mj < main_sequence_cut_r(jk,low=True,tight=tightness))

	jk = jk[good_mj_cut]
	mj = mj[good_mj_cut]
	tgas = tgas[good_mj_cut]

	vol_cut_indx = cut_indx_vol(tgas)

	return [color_cut, good_mj_cut, vol_cut_indx]

def cut_midplane(tgas, b_cut = 5):

	vol_cut_indx = cut_indx_vol(tgas)
	tgas = tgas[vol_cut_indx]
	midplane_indx = (np.abs(tgas['b']) <= b_cut)

	return [vol_cut_indx, midplane_indx]
#-------------------------------------------------------------------------------------------------------



sp= effsel.load_spectral_types()

tgas= gaia_tools.load.tgas()
twomass= gaia_tools.load.twomass()
bv = twomass['b_m_opt']-twomass['vr_m_opt']
jk = twomass['j_mag']-twomass['k_mag']
dm = -5.*np.log10(tgas['parallax'])+10.
mj = twomass['j_mag']-dm

tsf= gaia_tools.select.tgasSelect( max_plxerr=0.4 )
tsf._jmin= 2.
init_cuts = cut_general(tgas, mj, jk)

tgas = cut_flow(tgas, init_cuts)
twomass = cut_flow(twomass, init_cuts)
jk= cut_flow(jk, init_cuts)
dm= cut_flow(dm, init_cuts)
mj= cut_flow(mj, init_cuts)
bv= cut_flow(bv, init_cuts)

star_Category = ["A", "F", "G"]
#b_midplane = [5]
data_list = []
err_list = []
binz_list = []

for i, star_Cat in enumerate(star_Category):
	print('point running: ', i)
	zRange = z_cyl_cut; zNBin = 8; zStepWidth = zRange/zNBin 
	b_cut_deg = 5.

	file_evfs = "evfs_" + star_Cat + ".txt"
	evfs_cuts = cut_evfs(star_Cat, sp, jk, mj, tgas)
	mj_evfs = cut_flow(mj, evfs_cuts)
	jk_evfs = cut_flow(jk, evfs_cuts)


	#-----------------------------------------------------evfs----------------------------------------------------------

	zRangeList, evfs_out = generate_evfs(mj_evfs, jk_evfs, star_Cat, zRange = zRange, zNBin = zNBin, filename=file_evfs, save=True)
	#evfs_list.append(evfs_out)
	#plt.plot( theRange, evfs_list[0], 'r',  theRange, evfs_list[1], 'b',  theRange, evfs_list[2], 'g' )
	#plt.show()

	#-----------------------------------------------------evfs----------------------------------------------------------


	stars_cut = cut_stars(star_Cat, sp, jk, mj, tgas)
	stars_categorized = cut_flow(tgas, stars_cut)

	print("Star count after volume cut: ", len(tgas))

	midplane_cut = cut_midplane(stars_categorized, b_cut=b_cut_deg)
	stars_midplane = cut_flow(stars_categorized, midplane_cut)

	XYZ= bovy_coords.lbd_to_XYZ(stars_midplane['l'],stars_midplane['b'],1./stars_midplane['parallax'],degree=True)
	#r_cyl = np.sqrt(XYZ[:,0]**2.+XYZ[:,1]**2.)
	z_cyl= XYZ[:,2]
	
	zRangeList_upper = zRangeList + (zRange/zNBin)/2
	zRangeList_lower = zRangeList - (zRange/zNBin)/2
	evfs_weight = np.array([])
	for z_i in z_cyl:
		zPosition = (z_i < zRangeList_upper)*(z_i >= zRangeList_lower)
		evfs_weight = np.append( evfs_weight, 1./evfs_out[zPosition] ) 



	radeg = stars_midplane['ra']
	decdeg = stars_midplane['dec']
	pmra = stars_midplane['pmra']
	pmdec = stars_midplane['pmdec']

	print("Current Star Category: ", star_Cat)

	pm = ProperMotionTransform(radeg, decdeg, pmra, pmdec)

	ldeg = stars_midplane['l']
	bdeg = stars_midplane['b']
	plx = stars_midplane['parallax']
	z_pc = XYZ[:,2]*1000.
	pml = pm[0]
	pmb = pm[1]

	lRad = ldeg*np.pi/180.
	bRad = bdeg*np.pi/180.

	vKappa = 4.74047 #( km s^-1 mas (mas yr)^-1 )
	wSun = 7.01
	uSun = 11.1
	vSun = 12.24
	RVMean = -uSun*np.cos(lRad)*np.cos(bRad) - vSun*np.sin(lRad)*np.cos(bRad) - wSun*np.sin(bRad)
	VZ =  wSun + vKappa*pmb*np.cos(bRad)/plx + RVMean*np.sin(bRad)

	VZ_mean = np.mean(VZ)
	print("The mean is: ", VZ_mean)
	
	binning = 30
	plotRange = 40

	countStars = len(VZ)
	title = star_Cat + ' Stars w-profile at b < 5 deg. (' + str(countStars) + ')'
	print("VZ len: ", len(VZ))
	print("evfs len: ", len(evfs_weight))

	data, err, zbins = prepare_plot(np.abs(VZ), plotRange, binning, weights=evfs_weight, normed=True, positive=True, xlabel='w', ylabel='Count', title=title)
	data_list.append(data)
	err_list.append(err)
	binz_list.append(zbins)

	filename = star_Cat + '_gaia_w_midplane.txt'	
	np.savetxt(filename, np.transpose( np.array([zbins, data, err]) ), delimiter=',',header="z_Bin_center, w, w_err", fmt='%10.5f')

print("number of plots: ", len(data_list))
plot_hist(data_list, err_list, binz_list, normed=True, color=['y', 'r', 'm'],marker=['o','o','o'],plotLabel=['A','F','G'],xlabel='w',ylabel='f_0(w)',title='mid-plane w')

