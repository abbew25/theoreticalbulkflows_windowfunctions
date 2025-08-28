
A simple repository for theoretical calculations that relate to galaxy bulk flows. The codes in this repo may be useful for some. Running the makefile will set up a UV virtual environment will all required packages to run the scripts below. Power spectrum calculations rely on CAMB. 

# Some useful codes for plotting bulk flows or window functions.

```calculate_bulkflowvector_varyinggeometries.py```: this computes the BF for any spherical geometry if the window function is given and makes plots of the results for different geometries/scales and angular shapes. Use the ```Settings()``` object to modify the code for the desired survey geometry (there are options for the opening angle of a cone, the radius of the survey, and the galaxy distibution as uniform or having a gaussian distribution, etc). The code will save some plots for the user and also save some data with results for the mean bulk flow amplitude and standard deviation of the bulk flow to .npy files. Since this script can be slow, the user can specify whether to reload previous calculations that have been saved if desired. 

The user should experiment with number of grid cells (gridsize) for the discrete points that are fourier transformed to make sure the resolution is good enough for accurate results. For the given cosmology the Bulk Flow for a uniform distribution with a size r = 150/h Mpc should be approx 130 km/s, for comparison. The user can also compare the results to the script test_code_1dBulkflowintegral_sphere.py, as this should give the answer for a perfect sphere and is much simpler to compute (the calculation is analytical for a top hat geometry). 
	
 
```plot_powerspectrum_forsurveygeometry.py```: for a given radial geometry and angular cut-off will compute the window function in Fourier space for a given galaxy survey distribution. This assumes a perfect survey in a sense. Similar to the above script, the geometry of the window function can be specified in the ```Settings()``` object, and a plot will be saved with the window function for the user to view later. 


## Some more basic stuff for testing / checking results

```test_code_1dBulkflowintegral_sphere.py```: wrote just to quickly compute the 1d integral of the power spectrum with a spherical tophat window function for comparison to results in literature for BF expectation and 1-sigma uncertainty - and to check the expected result for the Bulk Flow amplitude from a fft is correct for more complicated geometries (the script ```calculate_bulkflowvector_varyinggeometries.py``` is capable of more compled window functions). 
	
```testcodefft.py```: I wrote this script to check I have the correct scaling after applying the scipy function for an fft applied to discrete data.


```bulkflowcalc_sphericalsymm.py```: this code gives the expected bulk flow amplitude for a galaxy survey with a top hat window (or can be simply modified for gaussian 3d dist.) for a spherical geometry. The radius of the top hat can be changed in the settings object.


# Acknowledgements 

Thanks to Sam Hinton for improvements to this code. 
