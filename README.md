
# shared_codes_astro
Some shared codes in the astro group that we can suggest improvements for.

	calculate_bulkflowvector_varyinggeometries.py - computes the BF for any spherical geometry if the window function is given and makes plots of the results for different geometries/scales and angular shapes. Seems to be bug free now. User should experiment with number of grid cells (gridsize) for the discrete points that are fourier transformed to make sure the resolution is good enough for accurate results. For the given cosmology the Bulk Flow for a uniform distribution with a size r = 150/h Mpc should be approx 130 km/s, for comparison. Can also compare results to test_code_1dBulkflowintegral_sphere.py, as this should give the answer for a perfect sphere and is much simpler to compute.
	plot_powerspectrum_forsurveygeometry.py - for a given radial geometry and angular cut-off will compute the window function in Fourier space for a given galaxy survey distribution. This assumes a perfect survey in a sense.


## Some more basic stuff for testing / checking I know what I'm doing

	test_code_1dBulkflowintegral_sphere.py - wrote just to quickly 1d integral over spherical tophat for comparison to results in literature for BF expectation and 1-sigma uncertainty - and to check expected result for Bulk Flow amplitude from fft is correct.
	testcodefft.py - wrote to check correct scaling after scipy function for fft is applied to discrete data.
	bulkflowcalc_sphericalsymm.py - gives a bulk flow for a top hat (or can be simply modified for gaussian 3d dist.) for a spherical geometry. Radius can be changed in settings object.


## Some codes that were specifically for making plots in my BF paper (not particularly useful for others so you can ignore and I won't bother to lint them nicely):

    code_plot_bulkflows_literature.py
    plot_mocks_amplitude_vs_spheretheory.py
    script_get_matterpower.py - just basic script to set up power spectrum. You can find a tidied up function for matter power spectrum from class in test_code_1dBulkflowintegral_sphere.py
