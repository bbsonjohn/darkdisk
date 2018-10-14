###############################################################################
# gaiaSelect.py: Selection function for (part of) the GAIA data set
###############################################################################
###############################################################################
#
# This file is a modified version of Jo Bovy's tgasSelect function.
# It contains routines to compute the selection function of subsets
# of the Gaia DR1 GAIA data. As usual, care should be taken when using this
# set of tools for a subset for which the selection function has not been 
# previously tested.
#
###############################################################################
import os, os.path
import hashlib
import tqdm
import numpy
from scipy import interpolate
import astropy.coordinates as apco
import healpy
import matplotlib.pyplot as plt
from galpy.util import bovy_plot, bovy_coords, multi
from matplotlib import cm
import gaia_tools.load
import scipy.interpolate as interp
from matplotlib import rcParams
from matplotlib import rc

try:
    import mwdust
except ImportError:
    _MWDUSTLOADED= False
else:
    _MWDUSTLOADED= True

# healpix def
_BASE_NSIDE= 2**5
_BASE_NPIX= healpy.nside2npix(_BASE_NSIDE)  
dr_2mass = 'hp' # data set label for 2mass
_njkbin = 3 # number of bins in j - ks

class gaiaSelect(object):
    def __init__(self,
                 min_nobs=8.5, 
                 max_nobs_std=10., 
                 max_plxerr=1.01, 
                 max_scd=0.7, 
                 min_lat=0., 
                 jmin=0.,jmax=13.5,jkmin=-0.05,jkmax=1.05):
        """#
        NAME:
           __init__
        PURPOSE:
           Initialize the GAIA selection function
        INPUT:
           Parameters for determining the 'good' part of the sky (applied at the 2^5 nside pixel level):
              min_nobs - minimum mean number of observations
              max_nobs_std - maximum spread in the number of observations
              max_plerr - maximum mean parallax uncertainty (default: off)
              min_lat - minimum |ecliptic latitude| in degree
           Parameters determining the edges of the color-magnitude considered (don't touch these unless you know what you are doing):
              jmin - Minimum J-band magnitude to consider
              jmax - Maximum J-band magnitude to consider
              jkmin - Minimum J-K_s color to consider
              jkmax - Maximum J-K_s color to consider
        OUTPUT:
           GAIA-selection-function object
        HISTORY:
           2017-01-17 - Started - Bovy (UofT/CCA)
           2018-01-01 - Modified for DR2 - John Leung (Brown)
                 
        """
        # Load the data
        #-----------------------------matching twomass with dr2/tgas---------------------------- 
        self._full_gaia = gaia_tools.load.gaia(dr=2) #  John: dr2/tgas catalog matched with twomass
        self._full_twomass =  gaia_tools.load.gaia(dr=2) #  John: twomass catalog matched with /dr2tgas
        self._complete_twomass  =  gaia_tools.load.twomass('hp') #  John: the whole twomass catalog
        
        
        self._ext_jk = self._complete_twomass['j_mag']-self._complete_twomass['k_mag']
        self._ext_jt = jt(self._ext_jk , self._complete_twomass['j_mag'])
        #----------------------------------------------------------------------------
                
        self._jkmin= jkmin #  John: lower bound in jk space
        self._jkmax= jkmax #  John: upper bound in jk spae
        
        self._full_jk= self._full_twomass['j_mag']-self._full_twomass['k_mag']
        self._full_jt= jt(self._full_jk,self._full_twomass['j_mag'])
        
        indx_jk_notnan = numpy.isfinite(self._full_jk)
        self._full_gaia = self._full_gaia[indx_jk_notnan]
        self._full_twomass = self._full_twomass[indx_jk_notnan]
        self._full_jk = self._full_jk[indx_jk_notnan]
        self._full_jt = self._full_jt[indx_jk_notnan]
        #  John: Some overall statistics to aid in determining the good sky, setup related to statistics of 6 < J < 10
        self._setup_skyonly(min_nobs,max_nobs_std,max_plxerr,max_scd,min_lat) #  John: select the good part of the sky
        self._determine_selection(jmin,jmax,jkmin,jkmax) 
      
        return None


    def _setup_skyonly(self,min_nobs,max_nobs_std,max_plxerr,max_scd,min_lat):
        
        djk = (self._jkmax-self._jkmin)/_njkbin # John: delta jk for the binning
        self._gaia_sid= (self._full_gaia['source_id']/2**(35.\
                               +2*(12.-numpy.log2(_BASE_NSIDE)))).astype('int') #  John: Some 
        self._gaia_sid_skyonlyindx= (self._full_jk > self._jkmin-djk/2.)\
            *(self._full_jk < self._jkmax+djk/2.)\
            *(self._full_twomass['j_mag'] > 0.)\
            *(self._full_twomass['j_mag'] < 13.5)
 
        nstar, e= numpy.histogram(self._gaia_sid[self._gaia_sid_skyonlyindx],
                                  range=[-0.5,_BASE_NPIX-0.5],bins=_BASE_NPIX)
        self._nstar_gaia_skyonly= nstar
        self._nobs_gaia_skyonly= self._compute_mean_quantity_gaia(\
            'astrometric_n_good_obs_al',lambda x: x/9.)
        self._nobsstd_gaia_skyonly= numpy.sqrt(\
            self._compute_mean_quantity_gaia(\
                'astrometric_n_good_obs_al',lambda x: (x/9.)**2.)
            -self._nobs_gaia_skyonly**2.)
        self._plxerr_gaia_skyonly= self._compute_mean_quantity_gaia(\
            'parallax_error')
        tmp_decs, ras= healpy.pix2ang(_BASE_NSIDE,numpy.arange(_BASE_NPIX),
                                      nest=True)
        coos= apco.SkyCoord(ras,numpy.pi/2.-tmp_decs,unit="rad")
        coos= coos.transform_to(apco.GeocentricTrueEcliptic)
        self._eclat_skyonly= coos.lat.to('deg').value
        #self._rv_nan = numpy.isnan(self._full_gaia['radial_velocity'])
        self._exclude_mask_skyonly= \
            (self._nobs_gaia_skyonly < min_nobs)\
            +(numpy.fabs(self._eclat_skyonly) < min_lat)\
            +(self._plxerr_gaia_skyonly > max_plxerr)\
            +(self._nobsstd_gaia_skyonly > max_nobs_std)

        return None

    def _determine_selection(self,jmin,jmax,jkmin,jkmax):
        """Determine the Jt dependence of the selection function in the 'good'
        part of the sky"""
        to_plot = False #John: whether to do the 2d density plot

        #John: note that jt is now j
        djt = 0.1 #John:  delta j for the binning
        self._jmin= jmin # John: setting the min j bin
        self._jmax= jmax # John: setting the max j bin
        djk = (self._jkmax-self._jkmin)/_njkbin # John: delta jk for the binning
        jtbins =  int((jmax-jmin)/djt)+1.  # John: number of bins in j
        
        # coverting from ra and dec to healpix
        theta= numpy.pi/180.*(90.-self._complete_twomass['dec'])
        phi= numpy.pi/180.*self._complete_twomass['ra']
        pix= healpy.ang2pix(_BASE_NSIDE,theta,phi,nest=True)

        findx = (self._ext_jk > jkmin-djk/2.)*(self._ext_jk< jkmax+djk/2.)\
            *(self._ext_jt < jmax+djt/2.)*(self._ext_jt > jmin-djt/2.) # John: cutting the set j, jk space boundary
        
        # John: generate a 3d histogram in j, jk and healpix space for twomass histogramdd([data], [# bins], [bins range])
        nstar2mass, edges  = numpy.histogramdd(\
                    [self._ext_jt[findx], self._ext_jk[findx], pix[findx] ], 
                    bins=[jtbins,_njkbin,_BASE_NPIX],
                    range=[[jmin-djt/2., jmax+djt/2.],
                           [jkmin-djk/2.,jkmax+djk/2.],[-0.5,_BASE_NPIX-0.5]]) 
        
        # John: generate a 3d histogram in j, jk and healpix space for dr2/tgas       
        findx = (self._full_jk > jkmin-djk/2.)*(self._full_jk < jkmax+djk/2.)\
            *(self._full_twomass['j_mag'] < jmax+djt/2.)*(self._full_twomass['j_mag'] > jmin-djt/2.) # cutting the set j, jk space boundary
                           
        nstargaia, edges = numpy.histogramdd(\
            numpy.array([self._full_jt[findx],self._full_jk[findx],\
                             (self._full_gaia['source_id'][findx]\
                                  /2**(35.+2*(12.-numpy.log2(_BASE_NSIDE))))\
                             .astype('int')]).T,
            bins=[jtbins,_njkbin,_BASE_NPIX],
            range=[[jmin-djt/2., jmax+djt/2.],
                   [self._jkmin-djk/2., self._jkmax+djk/2.],[-0.5,_BASE_NPIX-0.5]]) 
        # John: Only 'good' part of the sky
        
        nstar2mass[:,:,self._exclude_mask_skyonly]= numpy.nan # John: imposing the good part of the sky selection
        nstargaia[:,:,self._exclude_mask_skyonly]= numpy.nan
        
        nstar2mass= numpy.nansum(nstar2mass,axis=-1)# John: integrate (summing) over the healpix direction, setting nan = 0 in the sum 
        nstargaia= numpy.nansum(nstargaia,axis=-1)

        # John: show the density plots before fitting
        """if to_plot:
            plt.figure(figsize=(5,5))
            plt.title('TwoMass'); plt.xlabel('$J-K_s$'); plt.ylabel('$J$')
            plt.imshow(numpy.log10(nstar2mass), interpolation='nearest', origin='low', cmap='viridis',
                     extent=[edges[1][0], edges[1][-1], edges[0][0], edges[0][-1]], aspect='auto')
            cbar = plt.colorbar(); cbar.set_label("log count")
            plt.show()
            
            plt.figure(figsize=(5,5))
            plt.title('DR2'); plt.xlabel('$J-K_s$'); plt.ylabel('$J$')
            plt.imshow(numpy.log10(nstargaia), interpolation='nearest', origin='low', cmap='viridis',
                     extent=[edges[1][0], edges[1][-1], edges[0][0], edges[0][-1]], aspect='auto')
            cbar = plt.colorbar(); cbar.set_label("log count")
            plt.show()
            
            plt.figure(figsize=(5,5))
            plt.title('Completeness Volume'); plt.xlabel('$J-K_s$'); plt.ylabel('$J$')
            plotratio = nstargaia/nstar2mass

            for i in range(len(plotratio)):
                for j in range(len(plotratio[i])):
                    if not numpy.isfinite(plotratio[i][j]):
                        plotratio[i][j] = 0.
                    if  plotratio[i][j] > 1.:
                        plotratio[i][j] = 1.
            plt.imshow(plotratio, interpolation='nearest', origin='low', cmap='viridis',
                     extent=[edges[1][0], edges[1][-1], edges[0][0], edges[0][-1]], aspect='auto')
            cbar = plt.colorbar(); cbar.set_label("completeness before spline fit")
            plt.show()"""

    
        exs= 0.5*(numpy.roll(edges[0],1)+edges[0])[:-1] # John: generating the centre of the j-bins from their edges

        sf_splines= [] #John: the fitting spline function
        sf_props= numpy.zeros((_njkbin,3)) #obsolete
        
        #John: In the following loop, we spline-fit over j, iterate over jk, and integrate the healpix
        for ii in range(_njkbin): #John: iterate over jk bins
            # Determine the plateau, out of interest
            """level_indx= (exs > 8.5)*(exs < 9.5)
            sf_props[ii,0]=\
                numpy.nanmedian((nstargaia/nstar2mass)[level_indx,ii])"""
            # Spline interpolate`
            spl_indx= (exs >= 0.00)*(exs <= 13.5)\
                *(True-numpy.isnan((nstargaia/nstar2mass)[:,ii])) # John: cutting the region to calculate completeness and cutting out nan

                   
            #John: calculation the completeness and the uncertainty of completeness
            nstar_ratio = (nstargaia/(nstar2mass))[spl_indx,ii]
            nstar_ratio_err = (numpy.sqrt(nstargaia)/(nstar2mass))[spl_indx,ii]+0.02
            nstar_ratio_err[nstar_ratio == numpy.inf] = (1.+0.02)

            # John: setting completeness = 1.0 for gaia/twomass > 1.0  (meaning complete)
            nstar_ratio[nstar_ratio == numpy.inf] = 1.
            nstar_ratio[nstar_ratio>1.] = 1.
                
            #John: fitting using splines-functions
            #John: UnivariateSpline(xdata, ydata, w = weight for fitting, k = polynomial between nodes, s = convergence valjue)
            """tsf_spline= interpolate.UnivariateSpline(\
                exs[spl_indx],nstar_ratio,
                w= 1./(nstar_ratio_err),
                k=3,ext=1,s=numpy.sum(spl_indx)/4)"""
            # John: Setting s = 0 means we demand the fit must go through the data point. For high jk bins, the fitting is broken unless s = 0
            
            # use generic 1D-fit instead of spline
            tsf_spline = interp.interp1d(exs[spl_indx], nstar_ratio, fill_value = (0,0), bounds_error=False )
            
            sf_splines.append(tsf_spline) # John: append the fitting for the each jk bin and output
            
        #John: show the compoleteness plot
        if to_plot:
            plt.style.use("seaborn-bright")
            rcParams["savefig.dpi"] = 100
            rcParams["figure.dpi"] = 100
            rcParams["font.size"] = 10
            plt.rc('text', usetex=True)
        
            jk_plot =numpy.linspace(-0.1,1.0,_njkbin)
            j_plot = numpy.linspace(0.,14,100)
            jkbin_plot = range(_njkbin)
            #numpy.floor(_njkbin*(jk_plot-self._jkmin)\
                                   #/(self._jkmax-self._jkmin) ).astype('int')
            plt.figure(figsize=(5,5))
            plt.minorticks_on();
            plt.tick_params(axis='both', which='major', labelsize=18)
            plt.tick_params(axis='both', which='minor', labelsize=16)
            plt.xlabel(r'$J-K_s$', fontsize=20); plt.ylabel(r'$J$', fontsize=20)
            plt.title(r'$\rm{Effective \ completeness}$', fontsize=20)
            fit_completeness = []
            for ii in jkbin_plot:
                #if ii == 0 or ii == len(jkbin_plot)-1: continue
                fit_completeness_row = [float(sf_splines[ii]( jt(jk_plot[ii], j_plot[jj])) ) for jj in range(len(j_plot))]
                fit_completeness.append(fit_completeness_row)
            plt.ylim(0.00,13.40); 
            
            plt.imshow(numpy.transpose(fit_completeness), interpolation='nearest', origin='low', cmap='viridis',
                     extent=[jk_plot[0], jk_plot[-1], j_plot[0], j_plot[-1]], aspect='auto')
            cbar = plt.colorbar(); #cbar.set_label(r'$\rm{effective completeness}', fontsize=13)
            cbar.ax.tick_params(labelsize=18)
            saveFilePath = '/Users/john/Dropbox/DarkDisk/JohnCode/draft_plot/'
            plt.savefig( (saveFilePath + 'cmd_fit.pdf') , bbox_inches='tight')
            plt.show()
            
        self._sf_splines= sf_splines
        self._sf_props= sf_props
        return None


    def save_healpix_map(count, edges, filename = "2mc_out.txt"):
        """
        NAME:
           save_healpix_map
        PURPOSE:
           an option to create 2mass healpix density map 
        INPUT:
           count - count at each healpix, cmd coordinate
           edges - bin edges
        """
        dj_bin = edges[0][1]-edges[0][0]
        djk_bin = edges[1][1]-edges[1][0]
        hpix_bin = edges[2][1]-edges[2][0]

        j_mid_bin = (edges[0]+dj_bin/2.)[:-1]
        jk_mid_bin = (edges[1]+djk_bin/2.)[:-1]
        hpix_mid_bin = (edges[2]+hpix_bin/2.)[:-1]
        
        data = []
        for i , fi in enumerate(j_mid_bin):
            for m, fm in enumerate(jk_mid_bin):
                for n, fn in enumerate(hpix_mid_bin):
                    if starCount[i,m,n] > 0.:
                        data.append([fi, fm, fn, starCount[i,m,n]] )
        
        filename="2mc_out.txt"
        numpy.savetxt(filename , data)
        print("2mass 5 hp file saved to " + filename)

        return None
        

    def __call__(self,j,jk,ra,dec):
        """
        NAME:
           __call__
        PURPOSE:
           Evaluate the selection function for multiple (J,J-Ks) 
           and single (RA,Dec)
        INPUT:
           j - apparent J magnitude
           jk - J-Ks color
           ra, dec - sky coordinates (deg)
        OUTPUT
           selection fraction
        HISTORY:
           2017-01-18 - Written - Bovy (UofT/CCA)
        """
        # Parse j, jk
        if isinstance(j,float):
            j= numpy.array([j])
        if isinstance(jk,float):
            jk= numpy.array([jk])
        # Parse RA, Dec
        theta= numpy.pi/180.*(90.-dec)
        phi= numpy.pi/180.*ra
        pix= healpy.ang2pix(_BASE_NSIDE,theta,phi,nest=True)
        if self._exclude_mask_skyonly[pix]:
            return numpy.zeros_like(j)
        jkbin= numpy.floor((jk-self._jkmin)\
                               /(self._jkmax-self._jkmin)*_njkbin).astype('int')
        tjt= jt(jk,j)
        out= numpy.zeros_like(j)
        for ii in range(_njkbin):
            out[jkbin == ii]= self._sf_splines[ii](tjt[jkbin == ii])
        out[out < 0.]= 0.
        out[(j < self._jmin)+(j > self._jmax)]= 0.
        return out

    def determine_statistical(self,data,j,k):
        """
        NAME:
           determine_statistical
        PURPOSE:
           Determine the subsample that is part of the statistical sample
           described by this selection function object
        INPUT:
           data - a GAIA subsample (e.g., F stars)
           j - J magnitudes for data
           k - K_s magnitudes for data
        OUTPUT:
           index array into data that has True for members of the 
           statistical sample
        HISTORY:
           2017-01-18 - Written - Bovy (UofT/CCA)
        """
        # Sky cut
        data_sid= (data['source_id']\
                       /2**(35.+2*(12.-numpy.log2(_BASE_NSIDE)))).astype('int')
        skyindx= True-self._exclude_mask_skyonly[data_sid]
        # Color, magnitude cuts
        cmagindx= (j >= self._jmin)*(j <= self._jmax)\
            *(j-k >= self._jkmin)*(j-k <= self._jkmax)
        return skyindx*cmagindx

    def plot_mean_quantity_gaia(self,tag,func=None,**kwargs):
        """
        NAME:
           plot_mean_quantity_gaia
        PURPOSE:
           Plot the mean of a quantity in the GAIA catalog on the sky
        INPUT:
           tag - tag in the GAIA data to plot
           func= if set, a function to apply to the quantity
           +healpy.mollview plotting kwargs
        OUTPUT:
           plot to output device
        HISTORY:
           2017-01-17 - Written - Bovy (UofT/CCA)
        """
        mq= self._compute_mean_quantity_gaia(tag,func=func)
        cmap= cm.viridis
        cmap.set_under('w')
        kwargs['unit']= kwargs.get('unit',tag)
        kwargs['title']= kwargs.get('title',"")
        healpy.mollview(mq,nest=True,cmap=cmap,**kwargs)
        return None

    def _compute_mean_quantity_gaia(self,tag,func=None):
        """Function that computes the mean of a quantity in the GAIA catalog
        as a function of position on the sky, based on the sample with
        6 < J < 10 and 0 < J-Ks < 0.8"""
        if func is None: func= lambda x: x
        mq, e= numpy.histogram(self._gaia_sid[self._gaia_sid_skyonlyindx],
                               range=[-0.5,_BASE_NPIX-0.5],bins=_BASE_NPIX,
                               weights=func(self._full_gaia[tag]\
                                                [self._gaia_sid_skyonlyindx]))
        mq/= self._nstar_gaia_skyonly
        return mq
        

class gaiaEffectiveSelect(object):
    def __init__(self,gaiaSel,MJ=1.8,JK=0.25,dmap3d=None,
                 maxd=None):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize the effective GAIA selection function for a population of stars
        INPUT:
           gaiaSel - a gaiaSelect object with the GAIA selection function
           MJ= (1.8) absolute magnitude in J or an array of samples of absolute magnitudes in J for the tracer population
           JK= (0.25) J-Ks color or an array of samples of the J-Ks color
           dmap3d= if given, a mwdust.Dustmap3D object that returns the J-band extinction in 3D; if not set use no extinction
           maxd= (None) if given, only consider distances up to this maximum distance (in kpc)
        OUTPUT:
           GAIA-effective-selection-function object
        HISTORY:
           2017-01-18 - Started - Bovy (UofT/CCA)
        """
        self._gaiaSel= gaiaSel
        self._maxd= maxd
        # Parse MJ
        if isinstance(MJ,(int,float)):
            self._MJ= numpy.array([MJ])
        elif isinstance(MJ,list):
            self._MJ= numpy.array(MJ)
        else:
            self._MJ= MJ
        # Parse JK
        if isinstance(JK,(int,float)):
            self._JK= numpy.array([JK])
        elif isinstance(JK,list):
            self._JK= numpy.array(JK)
        else:
            self._JK= JK
        # Parse dust map
        if dmap3d is None:
            if not _MWDUSTLOADED:
                raise ImportError("mwdust module not installed, required for extinction tools; download and install from http://github.com/jobovy/mwdust")
            dmap3d= mwdust.Zero(filter='2MASS J')
        self._dmap3d= dmap3d
        return None

    def __call__(self,dist,ra,dec,MJ=None,JK=None):
        """
        NAME:
           __call__
        PURPOSE:
           Evaluate the effective selection function
        INPUT:
           distance - distance in kpc (can be array)
           ra, dec - sky coordinates (deg), scalars
           MJ= (object-wide default) absolute magnitude in J or an array of samples of absolute  magnitudes in J for the tracer population
           JK= (object-wide default) J-Ks color or an array of samples of the J-Ks color 
        OUTPUT
           effective selection fraction
        HISTORY:
           2017-01-18 - Written - Bovy (UofT/CCA)
        """
        if isinstance(dist,(int,float)):
            dist= numpy.array([dist])
        elif isinstance(dist,list):
            dist= numpy.array(dist)
        MJ, JK= self._parse_mj_jk(MJ,JK)
        distmod= 5.*numpy.log10(dist)+10.
        # Extract the distribution of A_J and A_J-A_Ks at this distance 
        # from the dust map, use twice the radius of a pixel for this
        lcen, bcen= bovy_coords.radec_to_lb(ra,dec,degree=True)
        pixarea, aj= self._dmap3d.dust_vals_disk(\
            lcen,bcen,dist,healpy.nside2resol(_BASE_NSIDE)/numpy.pi*180.)
        totarea= numpy.sum(pixarea)
        ejk= aj*(1.-1./2.5) # Assume AJ/AK = 2.5
        distmod= numpy.tile(distmod,(aj.shape[0],1))
        pixarea= numpy.tile(pixarea,(len(dist),1)).T
        out= numpy.zeros_like(dist)
        for mj,jk in zip(MJ,JK):
            tj= mj+distmod+aj
            tjk= jk+ejk
            out+= numpy.sum(pixarea*self._gaiaSel(tj,tjk,ra,dec),axis=0)
        if not self._maxd is None:
            out[dist > self._maxd]= 0.
        return out/totarea/len(MJ)

    def volume(self,vol_func,xyz=False,MJ=None,JK=None,
               ndists=101,linearDist=False,
               relative=False,
               ncpu=None):
        """
        NAME:
           volume
        PURPOSE:
           Compute the effective volume of a spatial volume under this effective selection function
        INPUT:
           vol_func - function of 
                         (a) (ra/deg,dec/deg,dist/kpc)
                         (b) heliocentric Galactic X,Y,Z if xyz
                      that returns 1. inside the spatial volume under consideration and 0. outside of it, should be able to take array input of a certain shape and return an array with the same shape
           xyz= (False) if True, vol_func is a function of X,Y,Z (see above)
           MJ= (object-wide default) absolute magnitude in J or an array of samples of absolute  magnitudes in J for the tracer population
           JK= (object-wide default) J-Ks color or an array of samples of the J-Ks color 
           relative= (False) if True, compute the effective volume completeness = effective volume / true volume; computed using the same integration grid, so will be more robust against integration errors (especially due to the finite HEALPix grid for the angular integration). For simple volumes, a more precise effective volume can be computed by using relative=True and multiplying in the correct true volume
           ndists= (101) number of distances to use in the distance integration
           linearDist= (False) if True, integrate in distance rather than distance modulus
           ncpu= (None) if set to an integer, use this many CPUs to compute the effective selection function (only for non-zero extinction)
        OUTPUT
           effective volume
        HISTORY:
           2017-01-18 - Written - Bovy (UofT/CCA)
        """           
        # Pre-compute coordinates for integrand evaluation            
        if not hasattr(self,'_ra_cen_4vol') or \
                (hasattr(self,'_ndists_4vol') and 
                 (ndists != self._ndists_4vol or 
                  linearDist != self._linearDist_4vol)):
            theta,phi= healpy.pix2ang(\
                _BASE_NSIDE,numpy.arange(_BASE_NPIX)\
                    [True-self._gaiaSel._exclude_mask_skyonly],nest=True)
            self._ra_cen_4vol= 180./numpy.pi*phi
            self._dec_cen_4vol= 90.-180./numpy.pi*theta
            if linearDist:
                dists= numpy.linspace(0.001,10.,ndists)
                dms= 5.*numpy.log10(dists)+10.
                self._deltadm_4vol= dists[1]-dists[0]
            else:
                dms= numpy.linspace(0.,18.,ndists)
                self._deltadm_4vol= (dms[1]-dms[0])*numpy.log(10.)/5.
            self._dists_4vol= 10.**(0.2*dms-2.)
            self._tiled_dists3_4vol= numpy.tile(\
                self._dists_4vol**(3.-linearDist),(len(self._ra_cen_4vol),1))
            self._tiled_ra_cen_4vol= numpy.tile(self._ra_cen_4vol,
                                                 (len(self._dists_4vol),1)).T
            self._tiled_dec_cen_4vol= numpy.tile(self._dec_cen_4vol,
                                                 (len(self._dists_4vol),1)).T
            lb= bovy_coords.radec_to_lb(phi,numpy.pi/2.-theta)
            l= numpy.tile(lb[:,0],(len(self._dists_4vol),1)).T.flatten()
            b= numpy.tile(lb[:,1],(len(self._dists_4vol),1)).T.flatten()
            XYZ_4vol= \
                bovy_coords.lbd_to_XYZ(l,b,
                   numpy.tile(self._dists_4vol,
                              (len(self._ra_cen_4vol),1)).flatten())
            self._X_4vol= numpy.reshape(XYZ_4vol[:,0],(len(self._ra_cen_4vol),
                                                       len(self._dists_4vol)))
            self._Y_4vol= numpy.reshape(XYZ_4vol[:,1],(len(self._ra_cen_4vol),
                                                       len(self._dists_4vol)))
            self._Z_4vol= numpy.reshape(XYZ_4vol[:,2],(len(self._ra_cen_4vol),
                                                       len(self._dists_4vol)))
        # Cache effective-selection function
        MJ, JK= self._parse_mj_jk(MJ,JK)
        new_hash= hashlib.md5(numpy.array([MJ,JK])).hexdigest()
        if not hasattr(self,'_vol_MJ_hash') or new_hash != self._vol_MJ_hash \
             or (hasattr(self,'_ndists_4vol') and 
                 (ndists != self._ndists_4vol or 
                  linearDist != self._linearDist_4vol)):
            # Need to update the effective-selection function
            if isinstance(self._dmap3d,mwdust.Zero): #easy bc same everywhere
                effsel_4vol= self(self._dists_4vol,
                                  self._ra_cen_4vol[0],
                                  self._dec_cen_4vol[0],MJ=MJ,JK=JK)
                self._effsel_4vol= numpy.tile(effsel_4vol,
                                              (len(self._ra_cen_4vol),1))
            else: # Need to treat each los separately
                if ncpu is None:
                    self._effsel_4vol= numpy.empty((len(self._ra_cen_4vol),
                                                    len(self._dists_4vol)))
                    for ii,(ra_cen, dec_cen) \
                            in enumerate(tqdm.tqdm(zip(self._ra_cen_4vol,
                                                       self._dec_cen_4vol))):
                        self._effsel_4vol[ii]= self(self._dists_4vol,
                                                    ra_cen,dec_cen,MJ=MJ,JK=JK)
                else:
                    multiOut= multi.parallel_map(\
                        lambda x: self(self._dists_4vol,
                                       self._ra_cen_4vol[x],
                                       self._dec_cen_4vol[x],MJ=MJ,JK=JK),
                        range(len(self._ra_cen_4vol)),
                        numcores=ncpu)
                    self._effsel_4vol= numpy.array(multiOut)
            self._vol_MJ_hash= new_hash
            self._ndists_4vol= ndists
            self._linearDist_4vol= linearDist
        out= 0.
        if xyz:
            out= numpy.sum(\
                self._effsel_4vol\
                    *vol_func(self._X_4vol,self._Y_4vol,self._Z_4vol)\
                    *self._tiled_dists3_4vol)
        else:
            out= numpy.sum(\
                self._effsel_4vol\
                    *vol_func(self._ra_cen_4vol,self._dec_cen_4vol,
                              self._dists_4vol)\
                    *self._tiled_dists3_4vol)
        if relative:
            if not hasattr(self,'_gaiaEffSelUniform'):
                gaiaSelUniform= gaiaSelectUniform(comp=1.)
                self._gaiaEffSelUniform= gaiaEffectiveSelect(gaiaSelUniform)
            true_volume= self._gaiaEffSelUniform.volume(vol_func,xyz=xyz,
                                                        ndists=ndists,
                                                        linearDist=linearDist,
                                                        relative=False)
        else:
            true_volume= 1.
        return out*healpy.nside2pixarea(_BASE_NSIDE)*self._deltadm_4vol\
            /true_volume

    def _parse_mj_jk(self,MJ,JK):
        if MJ is None: MJ= self._MJ
        if JK is None: JK= self._JK
        # Parse MJ
        if isinstance(MJ,(int,float)):
            MJ= numpy.array([MJ])
        elif isinstance(MJ,list):
            MJ= numpy.array(MJ)
        # Parse JK
        if isinstance(JK,(int,float)):
            JK= numpy.array([JK])
        elif isinstance(JK,list):
            JK= numpy.array(JK)
        return (MJ,JK)

class gaiaSelectUniform(gaiaSelect):
    """Version of the gaiaSelect code that has uniform completeness
    in a simple part of the sky, for relative effective volume and testing"""
    def __init__(self,comp=1.,ramin=None,ramax=None,keepexclude=False,
                 **kwargs):
        self._comp= comp
        if ramin is None: self._ramin= -1.
        else: self._ramin= ramin
        if ramax is None: self._ramax= 361.
        else: self._ramax= ramax
        gaia_tools.select.gaiaSelect.__init__(self,**kwargs)
        if not keepexclude:
            self._exclude_mask_skyonly[:]= False
        if not ramin is None:
            theta,phi= healpy.pix2ang(2**5,
                                      numpy.arange(healpy.nside2npix(2**5)),
                                      nest=True)
            ra= 180./numpy.pi*phi
            self._exclude_mask_skyonly[(ra < ramin)+(ra > ramax)]= True
        return None

    def __call__(self,j,jk,ra,dec):
        if ra < self._ramin or ra > self._ramax: return numpy.zeros_like(j)
        else: return numpy.ones_like(j)*self._comp

def jt(jk,j):
    #return j
    return j+jk**2.+2.5*jk
