'''Compliled by Anish Narkar
References:
-------------

[1] https://www.kaggle.com/tomahim/image-manipulation-augmentation-with-skimage
[2] https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
[3] https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_adapthist
[4] https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_hist
[5] http://paulbourke.net/miscellaneous/equalisation/

Operations Included:
----------------------

1. Rotations
2. Color Changes
3. Inverted Image
4. Contrast Check
5. Modify Contrast
6. Corrections - Gamma, Logartithmic, Sigmoid
7. Noise - Gauss, Salt&Pepper, Poisson, Speckle
8. Flip
9. Blur
10. Histogram Equalizations and matching

Notes - Remove cv2 dependency
'''
import numpy as np
import os
import cv2
from skimage.color import rgb2gray
from skimage import exposure
from scipy import ndimage
from skimage.transform import rotate


def add_rotations(image, degree, resize=True):    
    transform = rotate(image, degree, resize)
    return transform

def change_color(image):    
    gray_scale_image = rgb2gray(image)
    return gray_scale_image

def invert(image):    
    inverted_image = np.invert(image)
    return inverted_image

def check_contrast(image,fraction_threshold=0.05, lower_percentile=1, 
                   upper_percentile=99, method='linear'):
    
    return exposure.is_low_contrast(image, fraction_threshold, lower_percentile, upper_percentile)

def modify_contrast(image, prop):    
    v_min, v_max = np.percentile(image, (prop, 100-prop))
    modified_contrast = exposure.rescale_intensity(image, in_range=(v_min, v_max))
    return modified_contrast

def corrections(image, gamma, gain, gamma_correction=0, 
               log_correction=0, sigmoid_correction=0):
    
    if(gamma_correction):
        image = exposure.adjust_gamma(image, gamma, gain)
    
    if(log_correction):
        image = exposure.adjust_log(image)
    
    if(sigmoid_correction):
        image = exposure.adjust_sigmoid(image)    
    return image

def add_noise(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
   
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out
    
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

    
def flip(image, horizontal=0, vertical=0):
    if(horizonatal):
        flipped_image = image[:, ::-1]
    if(vertical):
        flipped_image = image[::-1, :]
    
    return flipped_image


def add_blur(image,size=(11, 11, 1)):
    blured_image = ndimage.uniform_filter(image, size)
    return blured_image


def histogram_equalization(image, clahe=0, clip_limit=0.01,
                           nbins=256, hist_equal =0, mask=None):
    
    if(clahe):
        equalized_image = exposure.equalize_adapthist(image, kernel_size, clip_limit, nbins)
    
    if(hist_equal):
        equalized_image = equexposure.equalize_hist(image, nbins, mask)
    return equalized_image


def histogram_match(image,reference,multichannel=False):
    match = exposure.match_histograms(image, reference, multichannel=False)
    return match
    
