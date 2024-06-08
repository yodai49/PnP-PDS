# Getting Started
- First, use requirements.txt to install all necessary packages with pip.
- Create a config folder in the root directory and create a "setup.json" file inside it.
- The content of setup.json should be as follows. Please adjust the paths according to your environment.

```json
{
    "path_train": "/Users/hogehoge/",
    "path_test": "/Users/hogehoge/",
    "path_result": "/Users/hogehoge/",
    "pattern_red": "*.JPEG", 
    "root_folder": "/Users/hogehoge/PnP-PDS/"
}
```

  - path_train: Used when training the denoiser. It can be arbitrary if you're only performing image restoration.
  - path_test: Folder containing images to be restored.
  - path_result: Folder where restoration results will be stored.
  - pattern_red: Specify ".JPEG" if it's JPEG. Likewise for other formats. Case sensitivity applies.
  - root_folder: Specify the location where the folder is placed.

For now, try calling eval_restoration from test.py.

# Parameters for eval_restoration
 - gaussian_nl: Standard deviation of Gaussian noise. If adding, values like 0.005 to 0.02 are suitable.
 - sp_nl: Ratio of sparse noise overlaid. Range from 0 to 1.
 - poisson_alpha: Scaling factor for Poisson noise. Around 100 is appropriate.
 - gamma1, gamma2: Step sizes for PnP-PDS.
 - alpha_n: Coefficient for the data constraint term in constrained image restoration problems. Theoretically, 1 is appropriate, but around 0.9 yields better results.
 - alpha_s: Coefficient for the sparse noise term in constrained image restoration problems. Similar to alpha_n, around 0.9 is good.
 - myLambda: Coefficient for the data term in additive formulation. Increasing emphasizes the data term, while decreasing emphasizes the regularization term. The appropriate value varies depending on the task, so it's a trial-and-error process.
 - architecture: Name of the denoiser.
 - deg_op: You can specify one of "blur", "random_sampling", or "Id". This corresponds to Phi in the paper.
 - method: Restoration method. Please refer to pds.py for details. For Poisson noise, it would be either ours-C or comparisonC-1, comparisonC-2, comparisonC-3. For Gaussian noise only, it would be ours-A or comparisonA-X, and for Gaussian + sparse, it would be ours-B or comparisonB-X.
 - m1, m2, gammaInADMMStep1: Parameters used in ADMM. You can find details in pds.py.
