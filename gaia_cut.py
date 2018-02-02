import sys
sys.path.insert(0, "/Users/john/Desktop/bovy_code/mwdust-master")
sys.path.insert(0, "/Users/john/Desktop/bovy_code/gaia_tools-master")
sys.path.insert(0, "/Users/john/Desktop/bovy_code/tgas-completeness-master/py")
sys.path.insert(0, "/Users/john/Desktop/bovy_code/isodist-master")

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

#-------------transform from proper motion in ICRS to proper motion in Galactic frame ------------------
def ProperMotionTransform(ra_coord, dec_coord, pmra_coord, pmdec_coord):

#   pmra_coord_cosdec = pmra_coord*np.cos(dec_coord*np.pi/180.)
   pmra_coord_cosdec = pmra_coord

   icrs = ICRS(ra=ra_coord*units.degree, dec=dec_coord*units.degree, pm_ra_cosdec=pmra_coord_cosdec*units.mas/units.yr, pm_dec=pmdec_coord*units.mas/units.yr)
   galactic = icrs.transform_to(Galactic)
   
   pml = (galactic.pm_l_cosb)/np.cos(galactic.b.radian)/(units.mas/units.yr)
   pmb = galactic.pm_b/(units.mas/units.yr)

   return (pml, pmb)

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


#------------------------------------------------------------------------------------------------------------------------------------

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
    xy = np.sqrt(X**2.+Y**2.)
    out = np.zeros_like(X)
    out[(xy >= xymin)*(xy < xymax)*(Z >= zmin)*(Z < zmax)]= 1.
    return out
#-------------------------------------------------------------------------------------------------------
def is_good_relplx(mj):
    out= np.empty_like(mj)
    out[mj > 5.] = 20.
    out[mj < 0.] = 10.
    out[(mj >= 0)*(mj <= 5.)] = 20.+2.*(mj[(mj >= 0)*(mj <= 5.)]-5.)
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


def find_bv_boundaries(starCategory, sp):

   sp_stars = lambda star_Cat: np.array([( str(star_Cat) in s) for s in sp['SpT'] ], dtype='bool')
   bv_color_cut_upper = float("nan")
   bv_color_cut_lower = float("nan")

   # First consider the input as main classes
   if starCategory == "A":
      bv_color_cut_upper = (  (sp['BV'])[ sp_stars('B9V') ] + (sp['BV'])[ sp_stars('A0V') ]  )/2
      bv_color_cut_lower = (  (sp['BV'])[ sp_stars('A9V') ] + (sp['BV'])[ sp_stars('F0V') ]  )/2
   elif starCategory == "F":
      bv_color_cut_upper = (  (sp['BV'])[ sp_stars('A9V') ] + (sp['BV'])[ sp_stars('F0V') ]  )/2
      bv_color_cut_lower = (  (sp['BV'])[ sp_stars('F9V') ] + (sp['BV'])[ sp_stars('G0V') ]  )/2
   elif starCategory == "G":
      bv_color_cut_upper = (  (sp['BV'])[ sp_stars('F9V') ] + (sp['BV'])[ sp_stars('G0V') ]  )/2 
      bv_color_cut_lower = (  (sp['BV'])[ sp_stars('G4V') ] + (sp['BV'])[ sp_stars('G5V') ]  )/2
   elif starCategory == "All":
      bv_color_cut_upper = (  (sp['BV'])[ sp_stars('B9V') ] + (sp['BV'])[ sp_stars('A0V') ]  )/2
      bv_color_cut_lower = (  (sp['BV'])[ sp_stars('G4V') ] + (sp['BV'])[ sp_stars('G5V') ]  )/2      
   else : # then consider the input as a subclass or a list of adjacent subclasses
      for list_i, bo_val in enumerate(sp_stars(starCategory)):
         if bo_val == True:
            bv_color_cut_upper = (  (sp['BV'])[list_i-1] + (sp['BV'])[list_i]  )/2
      for list_i, bo_val in enumerate( reversed(sp_stars(starCategory)) ):
         if bo_val == True:
            bv_color_cut_lower = (  (sp['BV'])[len(sp)-list_i] + (sp['BV'])[len(sp)-list_i-1]  )/2

   return (bv_color_cut_upper, bv_color_cut_lower)
#---------------------------------------------------------------------------------------------------------------------
def generate_evfs(mj_tight, jk_tight, spt, zspace, nintt_step = None, filename="default_evfs.txt" , save=True):

   zWidth = np.mean( (np.roll(zspace,-1) - zspace)[:-1] )
   evfs = np.array([])

   tesf= gaia_tools.select.tgasEffectiveSelect(tsf,dmap3d=mwdust.Zero(),MJ=mj_tight,JK=jk_tight,maxd=max_dist)
   if nintt_step == None:
      nintt_step = (2501*('A' in spt) + 1001 * (True-('A' in spt)))
   
   for i, z_i in enumerate(zspace):
      zmin = z_i - zWidth/2.
      zmax = z_i + zWidth/2.
      
      evfs = np.append(evfs, tesf.volume(lambda x,y,z: cyl_vol_func(x,y,z,xymax=r_cyl_cut,zmin=zmin,zmax=zmax), ndists=nintt_step,xyz=True,relative=True)   )

   if save == True:
      np.savetxt(filename, np.transpose( np.array([zspace, evfs]) ), delimiter=',',header="z_Bin_center, rel_effective_vol", fmt='%10.5f')
      
   return (zspace, evfs)
#

def load_evfs(filename="default_evfs.txt"):
   ZBINSCOLUMN = 0
   EVFSCOLUMN = 1
   zBins, evfs = np.genfromtxt(filename, delimiter= ",", usecols=(ZBINSCOLUMN, EVFSCOLUMN), unpack=True, skip_header=1)
   return (zBins, evfs)

#---------------------------------------------------------------------------------------------------------------------

def cut_indx_vol(tgas, rcut, zcut):

   XYZ= bovy_coords.lbd_to_XYZ(tgas['l'],tgas['b'],1./tgas['parallax'],degree=True)
   r_cyl = np.sqrt(XYZ[:,0]**2.+XYZ[:,1]**2.)
   z_cyl= XYZ[:,2]

   return  [(r_cyl < r_cyl_cut)*(np.abs(z_cyl) < z_cyl_cut)]

#---------------------------------------------------------------------------------------------------------------------

def cut_flow(data, cuts):
   for i, cut_indx in enumerate(cuts):
      data = data[cut_indx]
   return data

def cut_general(tgas, mj, jk):
   stat_indx = tsf.determine_statistical(tgas,twomass['j_mag'],twomass['k_mag'])
   tgas=tgas[stat_indx]
#   good_plx_indx = (tgas['parallax']/tgas['parallax_error'] > (is_good_relplx(mj)))*(jk != 0.)*(tgas['parallax'] > min_plx)

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

   return [color_cut, good_mj_cut]

def cut_stars_bv(starCat, sp, jk, bv, mj, tgas): # category cut and volume cut
   tightness = False
   cut_indx = []

   bv_color_cut_upper, bv_color_cut_lower = find_bv_boundaries(starCat, sp)
   color_cut =  (bv > bv_color_cut_upper)*(bv < bv_color_cut_lower)*(tgas['parallax']/tgas['parallax_error'] > (is_good_relplx(mj)))*(bv != 0.)

   bv = bv[color_cut]
   jk = jk[color_cut]
   mj = mj[color_cut]
   tgas = tgas[color_cut]

   good_mj_cut = ( mj > main_sequence_cut_r(jk,tight=tightness))*(mj < main_sequence_cut_r(jk,low=True,tight=tightness))

   return [color_cut, good_mj_cut]


def cut_midplane(tgas, b_cut = 5, r_cut = 0.15, z_cut = 0.20):

   vol_cut_indx = cut_indx_vol(tgas, r_cut, z_cut)
   tgas = tgas[vol_cut_indx]
   midplane_indx = (np.abs(tgas['b']) <= b_cut)

   return [vol_cut_indx, midplane_indx]
#-------------------------------------------------------------------------------------------------------

min_plx= 0.45/0.2
max_dist = 1./min_plx
max_plx_error = 0.4
tsf_jmin= 2.

r_cyl_cut = 0.15
z_cyl_cut = 0.22
to_load_evfs = False
dereddening = False

load_dereddened_data = False

#-------------------------------------------------initialization---------------------------------------------------------------
print("loading spectrum...")
sp= effsel.load_spectral_types()

print("loading tgas and twomass data...")
tgas= gaia_tools.load.tgas()
twomass= gaia_tools.load.twomass()
#bv = twomass['b_m_opt']-twomass['vr_m_opt']
jk = twomass['j_mag']-twomass['k_mag']
dm = -5.*np.log10(tgas['parallax'])+10.
mj = twomass['j_mag']-dm
print("executing preliminary cuts...")
tsf= gaia_tools.select.tgasSelect( max_plxerr=max_plx_error )
tsf._jmin= tsf_jmin
init_cuts = cut_general(tgas, mj, jk)

tgas = cut_flow(tgas, init_cuts)
twomass = cut_flow(twomass, init_cuts)
jk= cut_flow(jk, init_cuts)
dm= cut_flow(dm, init_cuts)
mj= cut_flow(mj, init_cuts)
#bv= cut_flow(bv, init_cuts)
#-------------------------------------------------initialization---------------------------------------------------------------
#-------------------------------------------------volumn cut---------------------------------------------------------------
vol_cut_indx = cut_indx_vol(tgas, r_cyl_cut, z_cyl_cut)
jk_cyl = cut_flow(jk, vol_cut_indx)
mj_cyl = cut_flow(mj, vol_cut_indx)
#bv_cyl = cut_flow(bv, vol_cut_indx)
tgas_cyl = cut_flow(tgas, vol_cut_indx)

print("Vol-reduced count: ", len(tgas_cyl))
#-------------------------------------------------volumn cut---------------------------------------------------------------

#-------------------------------------------------extinction---------------------------------------------------------------

#bv_corr = np.empty_like(bv_cyl)
jk_corr = np.zeros_like(jk_cyl)
mj_corr = np.zeros_like(mj_cyl)

if (dereddening) and (not load_dereddened_data):
   print("loading dust map...")
   #dust_combine = mwdust.Combined15()  #b-v filter
   dust_combine_J = mwdust.Combined15("2MASS J")
   dust_combine_K = mwdust.Combined15("2MASS Ks")
   print("Preparing dereddening...")
   for i in tqdm.tqdm(range(len(jk_cyl))):
      ldeg = (tgas_cyl['l'])[i]
      bdeg = (tgas_cyl['b'])[i]
      plx = (tgas_cyl['parallax'])[i]   
      ej = dust_combine_J(ldeg,bdeg,np.divide(1000.,plx))
      ek = dust_combine_K(ldeg,bdeg,np.divide(1000.,plx))
      ejk = ej-ek
      #ebv = dust_combine(ldeg,bdeg,np.divide(1000.,plx))
      #bv_corr[i] = bv_cyl[i] - ebv
      jk_corr[i] = jk_cyl[i] - ejk
      mj_corr[i] = mj_cyl[i] - ej
   dereddened_ouput_file = "temp_ext.txt"
   np.savetxt(dereddened_ouput_file, np.transpose( np.array([jk_corr, mj_corr]) ), delimiter=',',header="B-V, M_j")
   print("dereddening complete! File saved to: ", dereddened_ouput_file)

if dereddening and load_dereddened_data:
   print("loading dereddened data")
   JKCOLUMN = 0; MJCOLUMN = 1
   file_deredden = "temp_ext.txt" 
   jk_corr, mj_corr = np.genfromtxt(file_deredden, delimiter= ",", usecols=(JKCOLUMN, MJCOLUMN), unpack=True, skip_header=1)
   
#-------------------------------------------------extinction---------------------------------------------------------------

star_Category = ["A", "F", "G"]
#star_Category = []
#b_midplane = [5]
data_list = []
err_list = []
binz_list = []

for i, star_Cat in enumerate(star_Category):
   print('point running: ', i)
   print("Current Star Category: ", star_Cat)
   zNBin_evfs = 40
   #-----------------------------------------------------evfs----------------------------------------------------------
   zStepWidth = 2*z_cyl_cut/(zNBin_evfs-1)

   file_evfs = "evfs_" + star_Cat + ".txt"
   
   zRangeList = []
   evfs_out = []
   zspace_evfs = np.linspace(-z_cyl_cut-zStepWidth/2., z_cyl_cut+zStepWidth/2., zNBin_evfs+1)

   if to_load_evfs:
      print("Loaded evfs file: ", file_evfs)
      zRangeList, evfs_out = load_evfs(filename=file_evfs)
   else:  
      evfs_cuts = cut_evfs(star_Cat, sp, jk, mj, tgas)
      mj_evfs = cut_flow(mj, evfs_cuts)
      jk_evfs = cut_flow(jk, evfs_cuts)
      zRangeList , evfs_out = generate_evfs(mj_evfs, jk_evfs, star_Cat, zspace = zspace_evfs, filename=file_evfs, save=True)

   #evfs_list.append(evfs_out)
   #plt.plot( theRange, evfs_list[0], 'r',  theRange, evfs_list[1], 'b',  theRange, evfs_list[2], 'g' )
   #plt.show()
   print("completed!")
   #-----------------------------------------------------evfs---------------------------------------------------------------
   #-------------------------------------------------star density---------------------------------------------------------------

   if (load_dereddened_data == True) or (dereddening == True):
      stars_cut = cut_stars(star_Cat, sp, jk_corr, mj_corr, tgas_cyl)
   else:
      stars_cut = cut_stars(star_Cat, sp, jk_cyl, mj_cyl, tgas_cyl)
   stars_categorized = cut_flow(tgas_cyl, stars_cut)
   #bv_categorized = cut_flow(bv_cyl, stars_cut)   
   jk_categorized = cut_flow(jk_cyl, stars_cut)
   mj_categorized = cut_flow(mj_cyl, stars_cut)
   if load_dereddened_data == True:
      jk_corr_categorized = cut_flow(jk_corr, stars_cut)   
      mj_corr_categorized = cut_flow(mj_corr, stars_cut)

   print("optical cut completed!")   

   radeg = stars_categorized['ra']
   decdeg = stars_categorized['dec']
   pmra = stars_categorized['pmra']
   pmdec = stars_categorized['pmdec']

   pml, pmb = ProperMotionTransform(radeg, decdeg, pmra, pmdec)

   hipID = stars_categorized['hip']
   ldeg = stars_categorized['l']
   bdeg = stars_categorized['b']
   plx = stars_categorized['parallax']
   e_plx = stars_categorized['parallax_error']

   XYZ= bovy_coords.lbd_to_XYZ(ldeg, bdeg, 1./plx, degree=True)
   z_cyl= XYZ[:,2]
   z_pc = XYZ[:,2]*1000.

   zRangeList_upper = zRangeList + zStepWidth/2
   zRangeList_lower = zRangeList - zStepWidth/2
   evfs_weight = np.array([])
   for i, z_i in enumerate(z_cyl):
      zPosition = (z_i < zRangeList_upper)*(z_i >= zRangeList_lower)
      if np.sum(zPosition) == 0:
         print((z_i < zRangeList_upper), (z_i >= zRangeList_lower))
      evfs_weight = np.append(evfs_weight, 1./evfs_out[zPosition] )

   if len(evfs_weight) != len(z_cyl):
      print("Need a more inclusive evfs volumn! (larger z evfs range)")

   err_pml = [0.  for k in range(len(stars_categorized)) ]
   covpm = [0.  for k in range(len(stars_categorized)) ]

   bRad = bdeg*np.pi/180.
   ErrZcoord = np.abs(np.divide(1000.,plx**2)*np.sin(bRad)*e_plx)

   if dereddening:
      data = np.transpose( np.array([hipID,ldeg,bdeg,plx,e_plx,z_pc,evfs_weight, jk_categorized,mj_categorized, \
      mj_corr_categorized,jk_corr_categorized, pml,pmb,err_pml,covpm,covpm,ErrZcoord]) )
   else:
      data = np.transpose( np.array([hipID,ldeg,bdeg,plx,e_plx,z_pc,evfs_weight, jk_categorized,mj_categorized, \
      mj_categorized,jk_categorized, pml,pmb,err_pml,covpm,covpm,ErrZcoord]) )

   print(("evfs length: ", len(evfs_weight) ))
   print(("star length: ", len(z_pc) ))

   save_file_density = star_Cat + '_stars.txt'
   line_header = "# HpID, l (deg), b (deg), plx (mas), err_plx (mas), z_coord (pc), evfs_w, (B-V), AbsMag, AbsMag Corrected, \
   (B-V) Corrected, pm_l (mas/yr), pm_b (mas/yr), err_pml (mas/yr) , err_pmb (mas/yr), cov_pmlpmb, err_z_coord (pc)"
   print(data.dtype)
   np.savetxt(save_file_density, data, delimiter=',',header=line_header)

   print("Data saved to ", save_file_density)

   #-------------------------------------------------star density---------------------------------------------------------------
   #---------------------------------------------------mid-plane---------------------------------------------------------------
   b_cut_deg = 5.

   midplane_cut = cut_midplane(stars_categorized, b_cut=b_cut_deg)
   stars_midplane = cut_flow(stars_categorized, midplane_cut)
   ldeg_midplane = cut_flow(ldeg, midplane_cut)
   bdeg_midplane = cut_flow(bdeg, midplane_cut)
   pml_midplane = cut_flow(pml, midplane_cut)
   pmb_midplane = cut_flow(pmb, midplane_cut)
   plx_midplane = cut_flow(plx, midplane_cut)
   evfs_weight_midplane = cut_flow(evfs_weight, midplane_cut)

   print("Star count after mid-plane cut: ", len(stars_midplane))

   lRad_midplane = ldeg_midplane*np.pi/180.
   bRad_midplane = bdeg_midplane*np.pi/180.

   vKappa = 4.74047 #( km s^-1 mas (mas yr)^-1 )
   wSun = 7.01
   uSun = 11.1
   vSun = 12.24
   RVMean = -uSun*np.cos(lRad_midplane)*np.cos(bRad_midplane) - vSun*np.sin(lRad_midplane)*np.cos(bRad_midplane) - wSun*np.sin(bRad_midplane)
   VZ =  wSun + vKappa*pmb_midplane*np.cos(bRad_midplane)/plx_midplane + RVMean*np.sin(bRad_midplane)
   
   binning = 30
   plotRange = 40

   countStars = len(VZ)
   title = star_Cat + ' Stars w-profile at b < 5 deg. (' + str(countStars) + ')'
   print("midplane w len: ", len(VZ))
   print("midplane evfs len: ", len(evfs_weight_midplane))

   vz_err = np.zeros_like(VZ)

   save_file_w0 = star_Cat + '_gaia_w_midplane_5deg.txt'
   line_header="w, w_err, evfs"
   np.savetxt(save_file_w0, np.transpose( np.array([VZ, vz_err, evfs_weight_midplane]) ), delimiter=',',header=line_header, fmt='%10.5f')

   data, err, wbins = prepare_plot(np.abs(VZ), plotRange, binning, weights=evfs_weight_midplane, normed=False, positive=True, xlabel='w', ylabel='Count', title=title)

   w0data = np.transpose( np.array([wbins, data, err]) )


   save_file_w0 = star_Cat + '_stars_w_midplane.txt'
   line_header = "# w bins (km/s), count, count_err"
   np.savetxt(save_file_w0, w0data, delimiter=',',header=line_header)
   print("Mid-plane velocity saved to: ", save_file_w0)

   data_list.append(data)
   err_list.append(err)
   binz_list.append(wbins)

   #---------------------------------------------------mid-plane---------------------------------------------------------------

print("number of plots: ", len(data_list))
plot_hist(data_list, err_list, binz_list, normed=True, color=['y', 'r', 'm'],marker=['o','o','o'],plotLabel=['A','F','G'],xlabel='w',ylabel='f_0(w)',title='mid-plane w')


