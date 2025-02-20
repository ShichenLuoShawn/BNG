import nilearn
from nilearn import datasets
import numpy as np


# path = ''
if download:
    datasets.fetch_abide_pcp(path,band_pass_filtering=True,global_signal_regression=True,quality_checked=True,derivatives='rois_cc200')
