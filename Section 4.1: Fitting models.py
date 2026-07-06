import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import minimize
from functions import grid, hyp, hyp_gaussian, phi_model, phi_full


# %%PARAMETERS (all in meters)

lmd = 632e-9              #wavelength of laser
D = 0.4                   #distance between fibers and detector
B = 6e-3                  #basline: distance between fibers
pix_size = 4.4e-6         #pixel size

#%%st w/ grid
sig_obs = 0.2

npixx = 8000
Nx = 400

npixy = 5000
Ny = 250

# Unpack the coordinate arrays and pitches
x_coords, pitch_x = grid(npixx, Nx, pix_size)
y_coords, pitch_y = grid(npixy, Ny, pix_size)

# Create 2D coordinate grid
xx, yy = np.meshgrid(x_coords, y_coords)

# Take central horizontal row
row = Ny // 2

x_dat = xx[row, :].flatten()
y_dat = yy[row, :].flatten()

phi_true = 0.3
k = 2*np.pi/lmd

def straight_model(coords, phi):
    xcoord, ycoord = coords

    # Straight-line / small-angle path-difference approximation
    dL_st = xcoord * B / D

    return np.cos(k * dL_st + phi)

# True straight model on the selected row
st_true = straight_model((x_dat, y_dat), phi_true)

np.random.seed(0)
st_obs = st_true + np.random.normal(scale=sig_obs, size=st_true.shape)

# Fit only phi, because B, D, and lmd are fixed
params, params_covariance = optimize.curve_fit(
    straight_model,
    (x_dat, y_dat),
    st_obs,
    p0=[0.0]
)

phi_fit = params[0]
st_fit = straight_model((x_dat, y_dat), phi_fit)

yerr = np.full_like(st_obs, sig_obs)

plt.errorbar(x_dat, st_obs, yerr, fmt='o', label='Simulated Data')
plt.plot(x_dat, st_true, label='True Model')
plt.plot(x_dat, st_fit, '--', label='Curve Fit', color='black')
plt.title('Straight model curve fit')
plt.xlabel('x [m]')
plt.ylabel('Signal')
plt.legend()
plt.show()

print("True phi:", phi_true)
print("Fitted phi:", phi_fit)
print("x effective pitch:", pitch_x)
print("y effective pitch:", pitch_y)
#%%st final
sig_obs = 0.2

npix_min = -100000
npix_max = 100000
N = 100

npix = np.linspace(npix_min/2, npix_max/2, N)
x = y = npix * pix_size
x, y = np.meshgrid(x, y)

row = N -1

x_dat = x[row].flatten()
y_dat = y[row].flatten()

phi_true = 0.3
k = 2*np.pi/lmd

def straight_model(coords, phi):
    xcoord, ycoord = coords

    # Straight-line / small-angle path-difference approximation
    dL_st = xcoord * B / D

    # Intensity model, matching the hyperbolic sin^2 form
    return np.cos(k * dL_st + phi)

# True straight model on the same coordinate grid
st_true = straight_model((x_dat, y_dat), phi_true)

np.random.seed(0)
st_obs = st_true + np.random.normal(scale=sig_obs, size=st_true.shape)

# Fit only phi, because B, D, and lmd are fixed
params, params_covariance = optimize.curve_fit(
    straight_model,
    (x_dat, y_dat),
    st_obs,
    p0=[0.0]
)

phi_fit = params[0]
st_fit = straight_model((x_dat, y_dat), phi_fit)

yerr = np.full_like(st_obs, sig_obs)

plt.errorbar(x_dat, st_obs, yerr, fmt='o', label='Simulated Data')
plt.plot(x_dat, st_true, label='True Model')
plt.plot(x_dat, st_fit, '--', label='Curve Fit', color='black')
plt.title('Straight model curve fit ')
plt.xlabel('x(m)')
plt.ylabel('Intensity')
plt.legend()
plt.show()

print("True phi:", phi_true)
print("Fitted phi:", phi_fit)
#%% hyp model to hyp fringes ini

x1 = B/2 
x2 = B/2 

def hyp():
    L1 = np.sqrt((x + x1)**2 + y**2 + D**2)
    L2 = np.sqrt((x - x2)**2 + y**2 + D**2)

    dL_hyp = L2 - L1
    hyp = np.cos((2*np.pi)/lmd * dL_hyp)

    return dL_hyp, hyp, L1, L2

dL_hyp, hyp, L1, L2 = hyp()

hyp_true = hyp[row].flatten()
x_dathyp = x[row].flatten()
y_dathyp = y[row].flatten()

np.random.seed(0)
hyp_obs = hyp_true + np.random.normal(scale=sig_obs, size=hyp_true.shape)

def hyp_model(coords, phi):
    xcoord, ycoord = coords

    L1_line = np.sqrt((xcoord + x1)**2 + ycoord**2 + D**2)
    L2_line = np.sqrt((xcoord - x2)**2 + ycoord**2 + D**2)

    dL = L2_line - L1_line

    return np.cos((2*np.pi)/lmd * dL + phi)

params, params_covariance = optimize.curve_fit(
    hyp_model,
    (x_dathyp, y_dathyp),
    hyp_obs,
    p0=[0]
)

phi_fit = params[0]
hyp_fit = hyp_model((x_dathyp, y_dathyp), phi_fit)

yerr = np.full_like(hyp_obs, sig_obs)

plt.errorbar(x_dathyp, hyp_obs, yerr, fmt='o', label='Simulated Data')
plt.plot(x_dathyp, hyp_true, label='True Model')
plt.plot(x_dathyp, hyp_fit, '--', label='Curve Fit', color = 'black')
plt.xlabel('x(m)')
plt.ylabel('Intensity')
plt.title('Hyperbolic model curve fit')
plt.legend()
plt.show()


#%% grid

npixx = 2000
Nx = 200

npixy = 5000
Ny = 500

x_coords, pitch_x = grid(npixx, Nx, pix_size)
y_coords, pitch_y = grid(npixy, Ny, pix_size)

x, y = np.meshgrid(x_coords, y_coords)
extent = [0, npixx, 0, npixy]

x_pix = np.linspace(0, npixx, Nx)
y_pix = np.linspace(0, npixy, Ny)

xx_pix, yy_pix = np.meshgrid(x_pix, y_pix)
#%% 2D fit for straight fringes

sig_obs = 0.2

# Constants
k = 2*np.pi/lmd

# True parameters
kx_true = k * B / D
phi_true = 0.3

# Straight model
def straight_model(coords, kx, phi):
    xcoord, ycoord = coords
    return np.cos(kx*xcoord + phi)

# Create noiseless data
straight_true = straight_model((x, y), kx_true, phi_true)

# Add noise
np.random.seed(0)
straight_noisy = straight_true + np.random.normal(
    scale=sig_obs,
    size=straight_true.shape
)

# Flatten data for curve_fit
flat_coords = np.vstack((x.ravel(), y.ravel()))
flat_noisy = straight_noisy.ravel()

# Initial guess
p0 = [kx_true*0.9, 0.3]

# Fit
params, params_cov = optimize.curve_fit(
    straight_model,
    flat_coords,
    flat_noisy,
    p0=p0
)

kx_fit, phi_fit = params

# Evaluate fitted model on 2D grid
straight_fit = straight_model((x, y), kx_fit, phi_fit)

# Plot true model
plt.figure()
im = plt.imshow(
    straight_true,
    cmap='viridis',
    origin='lower',
    extent=extent
)
plt.colorbar(im, label='Signal')
plt.xlabel('x [pix]')
plt.ylabel('y [pix]')
plt.title('Straight model')
plt.show()

# Plot noisy data
plt.figure()
im = plt.imshow(
    straight_noisy,
    cmap='viridis',
    origin='lower',
    extent=extent
)
plt.colorbar(im, label='Signal')
plt.xlabel('x [pix]')
plt.ylabel('y [pix]')
plt.title('Straight noisy data')
plt.show()

# Plot fitted contours
plt.figure()
plt.contour(xx_pix, yy_pix, straight_fit)
plt.xlabel('x [pix]')
plt.ylabel('y [pix]')
plt.title('Straight fitted contours')
plt.show()

# Print results
print("Straight model fit")
print("True kx:", kx_true)
print("Fitted kx:", kx_fit)
print("True phi:", phi_true)
print("Fitted phi:", phi_fit)

row = Ny -1

plt.figure()
plt.errorbar(
    x_pix,
    straight_noisy[row, :],
    yerr=sig_obs,
    fmt='o',
    markersize=3,
    label='Noisy data'
)

plt.plot(
    x_pix,
    straight_true[row, :],
    label='True model'
)

plt.plot(
    x_pix,
    straight_fit[row, :],
    '--',
    color='black',
    label='2D fitted model'
)

plt.xlabel('x [pix]')
plt.ylabel('Signal')
plt.title('1D slice from 2D straight fit')
plt.legend()
plt.show()
#%% 2D fit for hyperbolic fringes

sig_obs = 0.2

# Constants
k = 2*np.pi/lmd

x1 = B/2
x2 = -B/2

# True parameter
phi_true = 0.3

# Hyperbolic model
def hyp_model(coords, phi):
    xcoord, ycoord = coords

    L1 = np.sqrt((xcoord + x1)**2 + ycoord**2 + D**2)
    L2 = np.sqrt((xcoord + x2)**2 + ycoord**2 + D**2)

    dL = L2 - L1

    return np.cos(k*dL + phi)

# Create noiseless data
hyp_true = hyp_model((x, y), phi_true)

# Add noise
np.random.seed(0)
hyp_noisy = hyp_true + np.random.normal(
    scale=sig_obs,
    size=hyp_true.shape
)

# Flatten data for curve_fit
flat_coords = np.vstack((x.ravel(), y.ravel()))
flat_noisy = hyp_noisy.ravel()

# Initial guess
p0 = [0.0]

# Fit
params, params_cov = optimize.curve_fit(
    hyp_model,
    flat_coords,
    flat_noisy,
    p0=p0
)

phi_fit = params[0]

# Evaluate fitted model on 2D grid
hyp_fit = hyp_model((x, y), phi_fit)

# Plot true model
plt.figure()
im = plt.imshow(
    hyp_true,
    cmap='viridis',
    origin='lower',
    extent=extent
)
plt.colorbar(im, label='Signal')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Hyperbolic model')
plt.show()

# Plot noisy data
plt.figure()
im = plt.imshow(
    hyp_noisy,
    cmap='viridis',
    origin='lower',
    extent=extent
)
plt.colorbar(im, label='Intensity')
plt.xlabel('x [pix]')
plt.ylabel('y [pix]')
plt.title('Hyperbolic noisy data')
plt.show()

# Plot noisy data with fitted contours
plt.figure()
'''
plt.imshow(
    hyp_noisy,
    cmap='viridis',
    origin='lower',
    extent=extent
)

plt.colorbar()
'''
plt.contour(xx_pix, yy_pix, hyp_fit)

plt.xlabel('x [pix]')
plt.ylabel('y [pix]')
plt.title('Hyperbolic fitted contours')
plt.show()

# Print results
print("Hyperbolic model fit")
print("True phi:", phi_true)
print("Fitted phi:", phi_fit)

# 1D slice from the 2D hyperbolic fit

row = Ny // 2   # central horizontal slice

plt.figure()
plt.errorbar(
    x_pix,
    hyp_noisy[row, :],
    yerr=sig_obs,
    fmt='o',
    markersize=3,
    label='Noisy data'
)

plt.plot(
    x_pix,
    hyp_true[row, :],
    label='True model'
)

plt.plot(
    x_pix,
    hyp_fit[row, :],
    '--',
    color='black',
    label='2D fitted model'
)

plt.xlabel('x [pix]')
plt.ylabel('Signal')
plt.title('1D slice from 2D hyperbolic fit')
plt.legend()
plt.show()
