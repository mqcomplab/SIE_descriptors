"""
Q1R1 - Q8R2 are the 16 charge transfer models for predicting specific ion effect (SIE) properties. 

Parameters
-----------
(all floats)
X : two values of HOMO and LUMO energies
m : fitting parameter, slope in the linear regression
b : fitting parameter, y-intercept in the linear regression
alpha : fitting parameter, size descriptor
gamma : fitting parameter for charge descriptor
b : fitting parameter for charge descriptor
a : fitting parameter for charge descriptor
xi : fitting parameter for charge descriptor
h : fitting parameter for charge descriptor

Return
------------
The predicted SIE property value for each model. 
"""

import numpy as np
def Q1R1(X, m, b):
  homo, lumo = X
  return m * (-(-(-homo - lumo)/2)/(-homo + lumo))/((-homo + lumo) ** -1) + b

def Q1R2(X, m, b, alpha):
  homo, lumo = X
  return m * (-(-(-homo - lumo)/2)/(-homo + lumo))/((-homo + lumo) ** alpha) + b

def Q2R1(X, m, b, gamma):
  homo, lumo = X
  return m * (-(-(-homo * gamma - lumo)/(1 + gamma))/(-homo + lumo))/((-homo + lumo) ** -1) + b

def Q2R2(X, m, b, gamma, alpha):
  homo, lumo = X
  return m * (-(-(-homo * gamma - lumo)/(1 + gamma))/(-homo + lumo))/((-homo + lumo) ** alpha) + b

def Q3R1(X, m, b):
  homo, lumo, homo1 = X
  return m * (- (-(-homo-lumo)/2/(-homo+lumo)) - ((-(-homo-lumo)/2) ** 2 / ( 2 * (-homo + lumo)) ** 3) * (-2*homo + lumo + homo1))/((-homo+lumo) ** -1) + b

def Q3R2(X, m, b, alpha):
  homo, lumo, homo1 = X
  return m * (- (-(-homo-lumo)/2/(-homo+lumo)) - ((-(-homo-lumo)/2) ** 2 / ( 2 * (-homo + lumo)) ** 3) * (-2*homo + lumo + homo1))/((-homo+lumo) ** alpha) + b

def Q4R1(X, m, b, h):
  homo, lumo = X
  return m * ( -(-(-homo - lumo)/2)/(-homo + lumo) - h * ((-(-homo - lumo)/2) ** 2 /(2 * (-homo + lumo) ** 3)) ) / ((-homo+lumo) ** -1) + b

def Q4R2(X, m, b, h, alpha):
  homo, lumo = X
  return m * ( -(-(-homo - lumo)/2)/(-homo + lumo) - h * ((-(-homo - lumo)/2) ** 2 /(2 * (-homo + lumo) ** 3)) ) / ((-homo+lumo) ** alpha) + b

def Q5R1(X, m, b, a):
  homo, lumo = X
  return m * ( -(-(-homo - lumo)/2)/(-homo + lumo) - a * ((-(-homo - lumo)/2) ** 2 /(2 * (-homo + lumo) ** 2)) ) / ((-homo+lumo) ** -1) + b

def Q5R2(X, m, b, a, alpha):
  homo, lumo = X
  return m * ( -(-(-homo - lumo)/2)/(-homo + lumo) - a * ((-(-homo - lumo)/2) **2 /(2 * (-homo + lumo) ** 2)) ) / ((-homo+lumo) ** alpha) + b

def Q6R1(X, m, b, gamma, xi):
  homo, lumo, homo1 = X
  return m * ((-(-(-homo*gamma-lumo)/(1+gamma))/(xi * (-homo+lumo))) -((-(-homo*gamma-lumo)/(2)) ** 2 /( 2 * xi **3 * (-homo+lumo) **3)) * (-2*homo + lumo + homo1))/((-homo+lumo) ** -1) + b

def Q6R2(X, m, b, gamma, xi, alpha):
  homo, lumo, homo1 = X
  return m * ((-(-(-homo*gamma-lumo)/(1+gamma))/(xi * (-homo+lumo))) -((-(-homo*gamma-lumo)/(2)) ** 2 /( 2 * xi **3 * (-homo+lumo) **3)) * (-2*homo + lumo + homo1))/((-homo+lumo) ** alpha) + b

def Q7R1(X, m, b, gamma, xi, h):
  homo, lumo = X
  return m * ((-(-(-homo*gamma-lumo)/(1+gamma))/(xi * (-homo+lumo))) - h * ((-(-homo*gamma-lumo)/(2)) ** 2 /( 2 * xi **3 * (-homo+lumo) **3))) /((-homo+lumo) ** -1) + b

def Q7R2(X, m, b, gamma, xi, h, alpha):
  homo, lumo = X
  return m * ((-(-(-homo*gamma-lumo)/(1+gamma))/(xi * (-homo+lumo))) - h * ((-(-homo*gamma-lumo)/(2)) ** 2 /( 2 * xi **3 * (-homo+lumo) **3))) /((-homo+lumo) ** alpha) + b

def Q8R1(X, m, b, gamma, xi, a):
  homo, lumo = X
  return m * ((-(-(-homo*gamma-lumo)/(1+gamma))/(xi * (-homo+lumo))) - a * ((-(-homo*gamma-lumo)/(2)) ** 2 /( 2 * xi **2 * (-homo+lumo) **2))) /((-homo+lumo) ** -1) + b

def Q8R2(X, m, b, gamma, xi, a, alpha):
  homo, lumo = X
  return m * ((-(-(-homo*gamma-lumo)/(1+gamma))/(xi * (-homo+lumo))) - a * ((-(-homo*gamma-lumo)/(2)) ** 2 /( 2 * xi **2 * (-homo+lumo) **2)) )/((-homo+lumo) ** alpha) + b

def calculate_aicc(n, mse, num_params):
    aic = n * np.log(mse) + 2 * num_params
    aic_c = aic + (2 * num_params * (num_params + 1))/(n - num_params - 1)
    return (aic_c)
