# -*- coding: utf-8 -*-
"""
Simple code to perform a 2D gaussian fit. The original code was found on the
Scipy Cookbook and was modified to support more fit-parameters:
1.  fitting starting centered on the 2D data or on the position of the
    maximum value of the 2D data
2. fitting a circular gaussian (width is the same in x and y)
Also, the full width half maximum (useful for circular fits) can be obtained.
Created on Wed Aug 20 16:20:07 2014
@author: Original Code found on the Scipy Cookbook, modified by
Nikolay Kladt, Image & Data Analyst, CECAD Imaging Facility
"""

from scipy import optimize
import numpy as np

LOWER_BOUND = -100
UPPER_BOUND = 100


def gaussian(height, center_x, center_y, sigma_x, sigma_y, circular=False):
    """ Returns a gaussian function with the given parameters"""
    if circular:
        return lambda x,y: height*np.exp(-(((center_x-x)/sigma_x)**2+((center_y-y)/sigma_x)**2)/2)
    else:
        return lambda x,y: height*np.exp(-(((center_x-x)/sigma_x)**2+((center_y-y)/sigma_y)**2)/2)

def gaussian_array(params, circular=False):
  return gaussian(*params, circular)

def moments(data, circular=False, centered=False):
    """ Returns (height, x, y, width_x, width_y, circular)
    the gaussian parameters of a 2D distribution by calculating its moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    if centered:
        x = float(data.shape[0]/2)
        y = float(data.shape[1]/2)
    else:
        x = (X*data).sum()/total
        y = (Y*data).sum()/total
    col = data[:, int(y)]
    sigma_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    sigma_y = np.sqrt(abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    if circular:
        sigma_x = (sigma_x + sigma_y)/2
        sigma_y = (sigma_x + sigma_y)/2

    return height, x, y, sigma_x, sigma_y


def fitgaussian(data, circular=False, centered=False):
    """ Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data, circular=circular, centered=centered)
    errorfunction = lambda p: np.ravel(gaussian_array(p, circular=circular)(*np.indices(data.shape)) - data)
    results = optimize.least_squares(errorfunction, params, bounds=(LOWER_BOUND,UPPER_BOUND))
    p = results.x
    if circular: # make sure that we have something sensible to sigma_y
        p[4] = p[3]
    return p


def fwhm(sigma):
    """ Calculates the full width half maximum for a given width
    only makes sense for circular gaussians """
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma # standard fcn, see web
    return fwhm

def gaussGrid(shape, height, center_x, center_y, sigma_x, sigma_y):
  g = gaussian(height, center_x, center_y, sigma_x, sigma_y)
  gauss_out = np.zeros(shape)
  for i in range(shape[0]):
    for j in range(shape[1]):
      gauss_out[i, j] = g(i, j)
  return gauss_out