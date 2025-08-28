

# Some useful codes for plotting bulk flows or window functions.

calculate_bulkflowvector_varyinggeometries.py - computes the BF for any spherical geometry if the window function is given and makes plots of the results for different geometries/scales and angular shapes. Seems to be bug free now. User should experiment with number of grid cells (gridsize) for the discrete points that are fourier transformed to make sure the resolution is good enough for accurate results. For the given cosmology the Bulk Flow for a uniform distribution with a size r = 150/h Mpc should be approx 130 km/s, for comparison. Can also compare results to test_code_1dBulkflowintegral_sphere.py, as this should give the answer for a perfect sphere and is much simpler to compute.
	
 
plot_powerspectrum_forsurveygeometry.py - for a given radial geometry and angular cut-off will compute the window function in Fourier space for a given galaxy survey distribution. This assumes a perfect survey in a sense.


## Some more basic stuff for testing / checking results

test_code_1dBulkflowintegral_sphere.py - wrote just to quickly 1d integral over spherical tophat for comparison to results in literature for BF expectation and 1-sigma uncertainty - and to check expected result for Bulk Flow amplitude from fft is correct.
	
testcodefft.py - wrote to check correct scaling after scipy function for fft is applied to discrete data.
	bulkflowcalc_sphericalsymm.py - gives a bulk flow for a top hat (or can be simply modified for gaussian 3d dist.) for a spherical geometry. Radius can be changed in settings object.


# Acknowledgements 

Thanks for Sam Hinton for improvements to this code. 
