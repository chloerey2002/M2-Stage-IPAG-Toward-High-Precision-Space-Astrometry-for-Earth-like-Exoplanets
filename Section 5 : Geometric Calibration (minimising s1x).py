import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import optimize
from functions import grid, hyp, phi_model, phi_full

# %%PARAMETERS (all in meters)

lmd = 632e-9              #wavelength of laser
D = 0.4                   #distance between fibers and detector
B = 6e-3                  #basline: distance between fibers
pix_size = 4.4e-6         #pixel size


s1x = B / 2
s2x = -B / 2
s1y = 0
s2y = 0

#%%GRID 
'''

npixx = 8000
Nx = 800
npixy =5000
Ny = 500

x, pitch, facx = grid(npixx, Nx, pix_size)
y, pitch, facy = grid(npixy, Ny, pix_size)

xx, yy = np.meshgrid(x, y)

extent = [0, npixx, 0, npixy]

fac_x = npixx / Nx
pitch_x = pix_size * fac_x
'''
interfringe_m = lmd * D / B
interfringe_pix = interfringe_m / pix_size
#interfringe_pix_scaled = interfringe_m / pitch_x

print(interfringe_m)
print(interfringe_pix)
#print(interfringe_pix_scaled)


npix_div2 = 100
npix_min = -npix_div2
npix_max = npix_div2
N = npix_max - npix_min
npix = np.linspace(npix_min, npix_max, N)

x = y = npix * pix_size  # position on screen in m
xx, yy = np.meshgrid(x, y)
extent = [0, npix_max*2, 0, npix_max*2]

#%%FRINGES

dL_hyp, phi, hyp_int = hyp(s1x, s2x, s1y, s2y, xx, yy, B, D, lmd)

plt.figure()
plt.title('Hyperbolic')
plt.imshow(hyp_int, cmap='viridis', origin='lower', extent = extent)
plt.colorbar(label='Intensity')

#%%Dark & bright fringe masks
dphi_dy, dphi_dx = np.gradient(phi, y, x)
dL_hyp, phi, hyp_int = hyp(s1x, s2x, s1y, s2y, xx, yy, B, D, lmd)

I = hyp_int  
dI_dx = np.gradient(I, axis=1)
d2I_dx2 = np.gradient(dI_dx, axis=1)

mask = 0.2
dark_hyp = ((np.abs(dI_dx) < mask) & (d2I_dx2 > 0))
bright_hyp = ((np.abs(dI_dx) < mask) & (d2I_dx2 < 0))

plt.figure()

plt.contour(bright_hyp, colors='red', linewidths=1, extent=extent)
plt.contour(dark_hyp, colors='black', linewidths=1.2, extent=extent)

legend_elements = [
    Line2D([0], [0], color='red', lw=1, label='Bright fringes'),
    Line2D([0], [0], color='black', lw=1.2, label='Dark fringes')
]

plt.legend(handles=legend_elements)
plt.title('Fringe Contours')
plt.show()


plt.figure()
plt.title('1D cut of Signal with contours')
plt.plot(bright_hyp[100, 75:125], label = 'Bright Contours')
plt.plot(hyp_int[100, 75:125], label = 'Signal')
plt.ylabel('Intensity')
plt.xlabel('pixels')
plt.legend()


plt.figure()
plt.title('Hyperbolic w/ dark contours')
plt.contour(dark_hyp, colors='red', linewidths=1, extent = extent)
plt.imshow(hyp_int, cmap='viridis', origin='lower', extent = extent)
plt.colorbar(label='Intensity')

#%% labelling fringes
rows_cols = np.where(dark_hyp)

row_fringe = []  
col_fringe = []  

interfringe = interfringe_pix

for row_curr, col_curr in zip(rows_cols[0], rows_cols[1]):
    flag_fringe_exists_already = False
    
    for k in range(len(row_fringe)):
        if np.min(np.sqrt((row_curr - row_fringe[k])**2 + (col_curr - col_fringe[k])**2)) < 0.9 * interfringe:
            row_fringe[k] = np.append(row_fringe[k], row_curr)
            col_fringe[k] = np.append(col_fringe[k], col_curr)
            flag_fringe_exists_already = True
            break
    
    if not flag_fringe_exists_already:
        row_fringe.append(row_curr)
        col_fringe.append(col_curr)

#Fringe coords plots
fringe_coords = {}

for k in range(len(row_fringe)):
    rows = row_fringe[k]
    cols = col_fringe[k]

    fringe_coords[k] = {"rows": rows,"cols": cols}
    
plt.figure()
plt.imshow(hyp_int, cmap="viridis", origin="lower")

plt.figure()
for k in fringe_coords:
    rows = fringe_coords[k]["rows"]
    cols = fringe_coords[k]["cols"]

    plt.scatter(cols, rows, s=0.5)

plt.title('K coords')
plt.xlabel("x / column")
plt.ylabel("y / row")


#%%phi plots
plt.figure()
plt.imshow(phi_full(B/2, xx, yy, B, D, lmd))
plt.title('Phi_full')
plt.colorbar()


plt.figure()
for k, coords in fringe_coords.items():
    rows = coords["rows"]
    cols = coords["cols"]

    phi = phi_model(s1x, rows, cols, xx, yy, B, D, lmd)
    #plt.imshow(phi,cmap="viridis")
    plt.scatter(cols, rows, c=phi/np.pi, s = 0.5, cmap="viridis")

plt.colorbar(label=r"phi / pi")
plt.title("phi_model")
plt.xlabel("column")
plt.ylabel("row")
plt.show()

#%% OPTIMIZE METHOD

def cost_fn(s1x):
    total = 0
    for k, coords in fringe_coords.items():#sum over k
        rows = coords["rows"]
        cols = coords["cols"]
        total = total + sum((phi_model(s1x, rows, cols, xx, yy, B, D, lmd) - 2*k*np.pi )**2)#sum over r
    return total

theta0 = [B/2]
result = optimize.minimize(cost_fn, theta0, method='Nelder-Mead')
print(result)


chi2 = result.fun
N = sum(len(coords["rows"]) for coords in fringe_coords.values())
chi2_red = chi2 / N
print('red_chi2 = ', chi2_red)

#%% VISUALISIG COST FUNCTION
s1x_loop = np.linspace(-5e-03, 5e-03,100)
s1x_costfn = []

for i in s1x_loop : 
    s1x_costfn = np.append(s1x_costfn, cost_fn(i))
    
plt.figure()    
plt.plot(s1x_loop, s1x_costfn)
plt.axvline(result.x, color='r', ls='--')
plt.title(r"$s_{Ax}$ Cost function")
plt.xlabel(r"$s_{Ax}$(m)")
plt.ylabel('Cost function')



