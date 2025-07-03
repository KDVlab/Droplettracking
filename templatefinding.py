# -*- coding: utf-8 -*-
"""
Preprocess and extract a template from an image sequence.
"""

# === Import necessary libraries ===
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from skimage.filters import unsharp_mask
from skimage.feature import match_template, peak_local_max
import pandas as pd
import pims
import trackpy as tp
from scipy.optimize import curve_fit
from derivative import dxdt
import multiprocessing
from numpy import diff
from matplotlib import image as mpimg
import imageio

# Set default plot settings
mpl.rc('figure', figsize=(10, 6))
mpl.rc('image', cmap='gray')

# === Define image preprocessing function ===
@pims.pipeline
def preprocess_img(frame):
    # Crop image to region of interest
    frame = frame[0:1200, 0:1300]
    # Convert to 8-bit format
    frame = frame.astype(np.uint8)
    # Sharpen image
    frame = unsharp_mask(frame, radius=2, amount=5)
    # Normalize brightness
    frame *= 255.0 / frame.max()
    return frame

# === Load and preprocess image sequence ===
directory = '--/'
run = '--/'
prefix = '*.tif'

# Load and process each frame in the sequence
frames = preprocess_img(pims.ImageSequence(os.path.join(directory + run + prefix)))

# Display the first processed frame
plt.imshow(frames[0])
plt.show()

# === Extract template from first frame ===
template_img = frames[0]

# Crop a section of the image to use as a template (y1:y2, x1:x2)
template = template_img[425:555, 1100:1200]

# Show the cropped template
plt.imshow(template)
plt.show()

# Save the template to a file
imageio.imwrite(directory + 'template.tif', template)
