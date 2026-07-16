import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, ndimage
from functions import grid, hyp, phi_model, phi_full
from matplotlib.lines import Line2D

#%%PARAMETERS (all in meters)--------------------------------------------------------------------

lmd = 632e-9              #wavelength of laser
D = 0.4                   #distance between fibers and detector
B = 6e-3                  #basline: distance between fibers
pix_size = 4.4e-6         #pixel size

#fibre positions
s1x = B / 2# + 1e-4
s2x = -B / 2
s1y = 0
s2y = 0


#%%GRID def-------------------------------------------------------------------------------------------

npixx = 8000
Nx = 8000
npixy =5000
Ny = 500

x, pitchx, facx = grid(npixx, Nx, pix_size)
y, pitchy, facy = grid(npixy, Ny, pix_size)

xx, yy = np.meshgrid(x, y)

extent = [0, npixx, 0, npixy]

fac_x = npixx / Nx
pitch_x = pix_size * fac_x

interfringe_m = lmd * D / B
interfringe_pix = interfringe_m / pix_size
interfringe_pix_scaled = interfringe_m / pitch_x

print('Interfringe in m=',interfringe_m)
print('Interfringe in pix =', interfringe_pix)
print('Interfringe in sampled pix=', interfringe_pix_scaled)

#%%Fringes-------------------------------------------------------------------------------------------

dL_hpy, phi, hyp_int = hyp(s1x, s2x, s1y, s2y, xx, yy, B, D, lmd)

plt.figure()
plt.title('Hyperbolic')
plt.imshow(hyp_int, cmap='viridis', origin='lower', extent = extent)
plt.colorbar(label='Intensity')
#%%graidnets
dphi_dy, dphi_dx = np.gradient(phi, y, x)

#%%Dark & bright fringe masks-------------------------------------------------------------------------------------------

I = hyp_int  
dI_dx = np.gradient(I, axis=1)
d2I_dx2 = np.gradient(dI_dx, axis=1)


mask = 0.3
dark_hyp = ((np.abs(dI_dx) < mask) & (d2I_dx2 > 0))
bright_hyp = ((np.abs(dI_dx) < mask) & (d2I_dx2 < 0))


#FRINGE CONTOURS
plt.figure()
plt.contour(bright_hyp, colors='red', linewidths=1, extent=extent)
plt.contour(dark_hyp, colors='black', linewidths=1.2, extent=extent)
legend_elements = [Line2D([0], [0], color='red', lw=1, label='Bright fringes'), Line2D([0], [0], color='black', lw=1.2, label='Dark fringes')] #dummy lines to create legend
plt.legend(handles=legend_elements)
plt.title('Fringe Contours')
plt.show()


#1D CUT BRIGHT
plt.figure()
plt.title('1D cut of Signal with bright contours')
plt.plot(bright_hyp[100, 75:200], label = 'Bright Contours')
plt.plot(hyp_int[100, 75:200], label = 'Signal')
plt.ylabel('Intensity')
plt.xlabel('pixels')
plt.legend()


#1D CUT DARK
plt.figure()
plt.title('1D cut of Signal with dark contours')
plt.plot(dark_hyp[100, 75:200], label = 'Dark Contours')
plt.plot(-hyp_int[100, 75:200], label = 'Signal')
plt.ylabel('Intensity')
plt.xlabel('pixels')
#plt.gca().invert_yaxis()
plt.legend()


#CONTOURS OVER IMAGE
plt.figure()
plt.title('Hyperbolic w/ dark contours')
plt.contour(dark_hyp, colors='red', linewidths=1, extent = extent)
plt.imshow(hyp_int, cmap='viridis', origin='lower', extent = extent)
plt.colorbar(label='Intensity')

#%% labelling fringes-------------------------------------------------------------------------------------------
kind = "dark"

if kind == "dark":
    fringe_mask = dark_hyp
elif kind == "bright":
    fringe_mask = bright_hyp

def label_fringe_mask(fringe_mask, min_size=20):
    """
    Labels connected fringe components.
    
    fringe_mask: dark_hyp or bright_hyp
    min_size :   the minimum # of connected pixels needed to be considered an object ie a fringe
    """
    structure = np.ones((3, 3), dtype=int)                               #def conected, all pixels in a 3x3 block are conected
    labelled, n_labels = ndimage.label(fringe_mask, structure)           #ndimage labels defined objects with k indicies now stored in labelled
                                                                         #n_labels - number connected components
    components = []

    for lab in range(1, n_labels + 1):                                   #range starts from 1 to number of connected components
        rows, cols = np.where(labelled == lab)                           #coords for one fringe


        components.append({ "rows": rows, "cols": cols, "mean_col": np.mean(cols)})     #mean cols allows for sorting later, leftmost fringes have smaller mean
        

    # Sort fringes from left to right
    components = sorted(components, key=lambda c: c["mean_col"])         # sorting using mean col ^

    fringe_coords = {}                                                   #dict of fringe coords w/ integer k

    for k, comp in enumerate(components):
        fringe_coords[k] = {"rows": comp["rows"], "cols": comp["cols"]}  #appending to fringe_coords dict

    return fringe_coords


fringe_coords = label_fringe_mask(fringe_mask)
print("# of bright/dark fringes:", len(fringe_coords))


#INTEGER PIX PLOT
plt.figure()
plt.imshow(hyp_int, aspect = 'auto')
for k in fringe_coords:
    rows = fringe_coords[k]["rows"]
    cols = fringe_coords[k]["cols"]
    plt.scatter(cols, rows, s=0.1)

plt.title("Integer-pixel fringe coords")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(0, 50)
plt.ylim(0, 200)
plt.show()


#%% SUB-PIXEL FRINGE LOCALISATION -----------------------------------------------------------------------------------

a_pix_sampled = interfringe_pix / facx

print("Interfringe in real pix:", interfringe_pix)
print("Interfringe in sampled pix:", a_pix_sampled)


def fit_local_sine(signal_1d, col0, a_pix, kind):
    """
    Fits a local sinusoid around an approx fringe position and returns
    the sub-pix column coord of the bright or dark extremum.

    signal_1d  : one row of the intensity image
    col0       : approximate column containing the fringe
    a_pix      : interfringe spacing in pixels
    kind       : "bright" for maxima, "dark" for minima
    """

    n = len(signal_1d)                         # number of cols in the row
    col0 = int(np.round(col0))                 #rounding col mean


    # find max or min in window of data
    #np.ceil -- rounding up
    half_window = int(np.ceil(0.5 * a_pix))

    #LOCAL WINDOW CENTERED ON COL0    
    c_min = max(0, col0 - half_window)
    c_max = min(n, col0 + half_window + 1)          #grid coords
    
    
    #COL0 LOCATION INTEGER FOR BRIGHT OR DARK--------------------------

    local_search = signal_1d[c_min:c_max]           #intensity array from c_min to c_max

    if kind == "bright":
        col0 = c_min + np.argmax(local_search)      #index where largest value occurs
    elif kind == "dark":
        col0 = c_min + np.argmin(local_search)      #index where smallest value occurs
                                                    #grid coord + min or max index within local search coords ie index w/ min or max intensity 
                                                    #returns bright or dark integer coord

    
    cols = np.arange(c_min, c_max)                  #creates array c_min to c_max, grid coords, creates local window
    q = cols - col0                                 # local window in pix coords realtive to centre col0 eg q = [-3, -2, -1, 0, 1, 2, 3]
    values = signal_1d[c_min:c_max]                 #intensity array for cols from c_min to c_max

    omega = 2 * np.pi / a_pix

    # LINEARISED SIN MODEL----------------------------------
    # I = A sin(wq) + B cos(wq) + C*1
    phi_i = omega * q
    M = np.column_stack([np.sin(phi_i), np.cos(phi_i), np.ones_like(q)])
    
    #LEAST SQUARES 
    A, B,C = np.linalg.lstsq(M, values, rcond=None)[0]  #[0] returns A, B, C values
    
    dphi = np.arctan2(B, A)
    
    if kind == "bright":
        target_phase = np.pi / 2      #sin(pi/2) = -1
    else:
        target_phase = -np.pi / 2     #sin(-pi/2) = -1
    
    deltax = (a_pix / (2 * np.pi)) * (target_phase - dphi)  #(pix/rad) * rad = pix
    
    col_sub = col0 + deltax

    
    return col_sub


def build_subpix_fringe_coords(fringe_coords, I, a_pix, kind):
    """
    Converts your labelled integer-pixel fringe coordinates into sub-pixel coordinates.

    For each fringe k and each row, it collapses the several mask pixels into one
    sub-pixel column position.
    """

    fringe_coords_subpix = {}

    for k, coords in fringe_coords.items():     #loop over k

        rows = np.asarray(coords["rows"])
        cols = np.asarray(coords["cols"])

        rows_sub = []
        cols_sub = []

        for row in rows:                   #loop over each row
            col0 = np.mean(cols)           # Approx. fringe column from mask

            col_sub = fit_local_sine( I[row, :], col0, a_pix,  kind )  

            rows_sub.append(row)
            cols_sub.append(col_sub)

     #storing one complete fringe
        fringe_coords_subpix[k] = {
            "rows": np.asarray(rows_sub),
            "cols": np.asarray(cols_sub)
            }

    return fringe_coords_subpix


#%%SUB PIX PLOT-------------------------------------------------------------------------------------------
I = hyp_int

fringe_coords_subpix = build_subpix_fringe_coords(fringe_coords, I, a_pix_sampled,kind)

plt.figure()
plt.imshow(hyp_int, cmap="viridis", origin="lower", aspect = 'auto')

for k, coords in fringe_coords_subpix.items():
    rows = coords["rows"]
    cols = coords["cols"]
    plt.scatter(cols, rows, s=0.2)

plt.title("Sub-pixel fringe coordinates")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


#%% OUTER FRINGES PLOT-------------------------------------------------------------------------------------------

n_edge = 10
n_fringes = len(fringe_coords_subpix)

outer_fringes =  np.arange(0, n_edge)

plt.figure()
plt.imshow(hyp_int, aspect = 'auto')

for k in outer_fringes:
    rows = fringe_coords_subpix[k]["rows"]
    cols = fringe_coords_subpix[k]["cols"]
    plt.scatter(cols, rows, s=3, label=f"k={k}")

plt.title("Outer fringes")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(0, 50)
plt.ylim(200,500)
plt.legend()
plt.show()


plt.figure()
plt.imshow(hyp_int, aspect = 'auto')

for k in outer_fringes:
    rows = fringe_coords_subpix[k]["rows"]
    cols = fringe_coords_subpix[k]["cols"]
    plt.scatter(cols, rows, s=3, label=f"k={k}")

plt.title("Outer fringes")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(0, 100)
plt.ylim(100,500)
plt.legend()
plt.show()


