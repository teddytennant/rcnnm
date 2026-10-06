RCNNM is a JAX implementation of reflective convolution nuclear norm minimization (arXiv:2610.04979).
Install with `pip install -e .` and run the tests with `pytest`.
This cut checks the reflective lifting, the weighted Gram identity, and a tiny synthetic recovery. It does not run the BSDS or CAVE experiments and does not claim the paper's 3.26 dB boundary-PSNR gain.
