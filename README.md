# README #

Requires numpy, scipy for the main module nonlinear_mass_faster.py, plus taylor_coeffs.bin and fiducial_rnl.txt

Requires camb to generate the power spectrum files

The main module is nonlinear_mass_faster.py and an example is shown in example.ipynb.
One can simply use nonlinear_mass_faster.py + taylor_coeffs.bin + fiducial_rnl.txt to either fit the polynomial
coefficients, or use the Taylor series in the default Planck18 cosmology.

We also include various helper scripts in case the user wants to re-generate the Taylor
series about a cosmology of their choosing.

"make_pklin.py" and "make_pklin_all_cosmos.py" generate the default linear power spectra,
and the linear power spectra for tests and the Taylor series.

"compute_taylor_coeffs.py" creates the "taylor_coeffs.bin" file from the Taylor power spectra.

"tabulate_fiducial_rnl.py" tabulates RNL in the fiducial cosmology (Planck18), needed
for setting the fitting range.

Finally, we include two scripts that replicate the key tests in Krolewski & Slepian 2019:
"accuracy.py" and "timing_test.py" run the accuracy and timing tests.
