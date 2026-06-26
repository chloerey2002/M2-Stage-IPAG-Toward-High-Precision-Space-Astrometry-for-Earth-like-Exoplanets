import numpy as np
import matplotlib.pyplot as plt
from functions import grid, hyp, straight, interfringe, alpha

# %%PARAMETERS (all in meters)

lmd = 632e-9              #wavelength of laser
D = 0.4                   #distance between fibres and detector
B = 6e-3                  #basline: distance between fibres
pix_size = 4.4e-6         #pixel size
delta_D = 0               #Set to value if one fibre has different D than the other


#%%PIXEL GRID

npixx = 8000                                        #number of pixels in x
Nx = 800                                            #number of sampling points in x
npixy =5000                                         #number pixels in y
Ny = 500                                            #number of sampling points in y

x, pitch = grid(npixx, Nx, pix_size)                #x coords
y, pitch = grid(npixy, Ny, pix_size)                #y coords

xx, yy = np.meshgrid(x, y)  

extent = [0, npixx, 0, npixy]

fac_x = npixx / Nx                                   #factor the pitch will be increased by to account for fewer sampling points than pixels
pitch_x = pix_size * fac_x                           #effective pitch

interfringe_m = lmd * D / B                          #interfringe in meters
interfringe_pix = interfringe_m / pix_size           #interfringe in pixels
interfringe_pix_scaled = interfringe_m / pitch_x     #effective interfringe 

print(interfringe_m)
print(interfringe_pix)
print(interfringe_pix_scaled)

# %%OFFSETS: offset values if fibres are not in ideal positions

#fibre 1
x_offset1 = 0
y_offset1 = 0

#fibre 2
x_offset2 = 0
y_offset2 = 0

# Source positions
#fibre 1
s1x = B/2 + x_offset1
s1y = 0 + y_offset1

#fibre 2
s2x = -B/2 + x_offset2
s2y = 0 + y_offset2


#%%ROTATION ANGLE: Set to value to include tilt of camera or fibres

theta_deg = 0                         #tilt in degrees
theta_rad= np.deg2rad(theta_deg)      #tilt in radians

#new x,y coordinates with tilt
xprime = xx*np.cos(theta_rad) + yy*np.sin(theta_rad)
yprime = -xx*np.sin(theta_rad) + yy*np.cos(theta_rad)


# %%STRAIGHT FRINGES MODEL

dL_st, I_st = straight(xprime, B, D, lmd)

#2d map of straight fringes
plt.title('Straight')
plt.imshow(I_st, cmap='viridis', origin='lower', extent = extent)
plt.colorbar(label='Intensity')


# %%HYPERBOLIC FRINGES MODEL

dL_hyp, phi, I_hyp = hyp(s1x, s2x, s1y, s2y, xprime, yprime, B, D, lmd)

#2d map of hyperbolic fringes
plt.figure()
plt.title('Hyperbolic')
plt.imshow(I_hyp, cmap='viridis', origin='lower', extent = extent)
plt.colorbar(label='Intensity')


#%% THEORETICAL STRAIGHT INTERFRINGE

err = dL_st - dL_hyp

i_st = lmd * D / B  
i_stpix = i_st / pix_size

spacing_st = np.full_like(dL_st, i_st)
interfrange_st = np.full_like(xx, i_stpix)

plt.figure()
plt.imshow(interfrange_st, origin='lower', cmap='viridis', extent = extent)
plt.title('Straight Interfringe')
plt.colorbar(label = 'pixel')


#%% 2D MAPS

#INTERFRINGE: Spacing between fringes

#Straight 
plt.figure()
plt.imshow(interfringe(dL_st, lmd, x, y, pix_size), origin='lower', cmap='viridis', extent = extent)
plt.title('Straight Interfringe', pad=20)
plt.colorbar(label='Number of pixels')


#Hyperbolic 
plt.figure()
plt.imshow(interfringe(dL_hyp, lmd, x, y, pix_size), origin='lower', cmap='viridis', extent = extent)
plt.title('Hyperbolic Interfringe')
plt.colorbar(label='Number of pixels')

#-----------------------------------------------------------------------------------------------------------

# ALPHA: Local angle of fringes

#Straight
alpha_st = np.zeros_like(alpha(dL_st, lmd, x, y))
plt.figure()
plt.imshow(alpha_st, origin='lower', cmap='viridis', extent = extent, vmin=0, vmax=1)
plt.title('Alpha Straight')
plt.colorbar(label = 'local angle(deg)')


#Hyperbolic
plt.figure()
plt.imshow(alpha(dL_hyp, lmd, x, y), origin='lower', cmap='viridis', extent = extent)
plt.title('Alpha Hyperbolic')
plt.colorbar(label = 'local angle(deg)')

#-----------------------------------------------------------------------------------------------------------

# DIFFERENCE BETWEEN HYPERBOLIC AND STRAIGHT FRINGES


def error():
    err_if = interfringe(dL_hyp, lmd, x, y, pix_size) - interfringe(dL_st, lmd, x, y, pix_size)
    err_alpha = alpha(dL_hyp, lmd, x, y) - alpha(dL_st, lmd, x, y)
    return err_if, err_alpha

err_if, err_alpha = error()

err_if = interfringe(dL_hyp, lmd, x, y, pix_size) - interfringe(dL_st, lmd, x, y, pix_size)   #Difference of interfringe between the two models

plt.figure()
plt.title('Interfringe Error')
plt.imshow(err_if, origin='lower', cmap='viridis', extent = extent)
plt.colorbar(label='Number of pixels')


err_alpha = alpha(dL_hyp, lmd, x, y) - alpha(dL_st, lmd, x, y)                                #Difference of local angle between the two models

plt.figure()
plt.title('Alpha Error')
plt.imshow(err_alpha, origin='lower', cmap='viridis', extent = extent)
plt.colorbar(label='local angle (deg)')


avg = np.mean(err_if)
res = err_if - avg

plt.figure()
plt.title('Interfringe Error avg')
plt.imshow(np.abs(res), origin='lower', cmap='viridis', extent = extent)
plt.colorbar(label='Number of pixels')




# %%LOOP OF FRINGES AT DIFFERENT DISTANCES AND WITH DIFFERENT NUMBER OF PIXELS
#aim is to see how far away we have to be for the straight fringe approximation to be true

#max and min number of pixels
npix_min = 1000
npix_max = 10000
n = 10
npix_loop = np.linspace(npix_min, npix_max, n)   # array of pixels example : ([ 5000.,  8750., 12500., 16250., 20000.])

#max and min distances in meters
D_max = 3
D_min = 0.3
D_loop = np.linspace(0.3, 3, 5) # array of distances example: ([0.3  , 0.975, 1.65 , 2.325, 3. ])


#initialising total interfringe and local angle error lists
if_error_all = []
a_error_all = []

for i, pix in enumerate(npix_loop):
    
    #initialising interfringe and local angle error list for each loop of pixel array 
    if_error = []
    a_error = []

    for D in D_loop:
        
        # GRID
        N = 1000
        pitch = pix_size * (pix/N)
        npix = np.linspace(-pix/2, pix/2, N)
        
        x = y = npix * pitch
        xx_loop, yy_loop = np.meshgrid(x, y)
        extent = [0, pix, 0, pix]


        dL_st, st = straight(xx_loop, B, D, lmd)
      
        dL_hyp,phi, I_hyp = hyp(s1x, s2x, s1y, s2y, xx_loop, yy_loop, B, D, lmd)
        
        err_if = interfringe(dL_hyp, lmd, x, y, pix_size) - interfringe(dL_st, lmd, x, y, pix_size)   #Difference of interfringe between the two models
        err_alpha = alpha(dL_hyp, lmd, x, y) - alpha(dL_st, lmd, x, y)  
        
        #Uncomment if you wish to see the maps of each loop
        '''
        # Interfringe maps
        plt.figure()
        plt.imshow(interfringe(dL_st, lmd, x, y , pix_size), origin='lower', cmap='viridis', extent = extent)
        plt.title(f'Straight Interfringe\nD = {D:.2f} m, {pix}px', pad=20)
        plt.colorbar(label='Number of pixels')

        plt.figure()
        plt.imshow(interfringe(dL_hyp, lmd, x, y , pix_size), origin='lower', cmap='viridis', extent = extent)
        plt.title(f'Hyperbolic Interfringe\nD = {D:.2f} m, {pix}px')
        plt.colorbar(label='Number of pixels')

        # Alpha maps
        plt.figure()
        plt.imshow(alpha(dL_st, lmd, x, y), origin='lower', cmap='viridis', extent = extent)
        plt.title(f'Alpha Straight\nD = {D:.2f} m, {pix}px')
        plt.colorbar(label='local angle')

        plt.figure()
        plt.imshow(alpha(dL_hyp, lmd, x, y), origin='lower', cmap='viridis', extent = extent)
        plt.title(f'Alpha Hyperbolic\nD = {D:.2f} m, {pix}px')
        plt.colorbar(label='local angle')
        
        # Interfringe error
        plt.figure()
        plt.title(f'Interfringe Error\nD = {D:.2f} m, {pix}px')
        plt.imshow(err_if, origin='lower', cmap='viridis', extent = extent)
        plt.colorbar(label='Number of pixels')


        # Alpha error
        plt.figure()
        plt.title(f'Alpha Error\nD = {D:.2f} m, {pix}px')
        plt.imshow(err_alpha, origin='lower', cmap='viridis', extent = extent)
        plt.colorbar(label='local angle')
        
        '''
        if_error.append(err_if.max())
        a_error.append(err_alpha.max())
        
    if_error_all.append(if_error)
    a_error_all.append(a_error)
    
#----------------------------------------------------------------------------------------------------------------

# MAX ERROR PLOTS:visulising results from loop

plt.figure()
for i in range(len(npix_loop)):
    plt.semilogy(D_loop, if_error_all[i], label=f'{int(npix_loop[i])} px')
plt.xlabel('Distance (m)')
plt.ylabel('Log Max interfringe error (px)')
plt.legend(title="Number of pixels" , loc='upper right')
plt.title('Max interfringe error vs Distance')
plt.show()

plt.figure()
for i in range(len(npix_loop)):
    plt.semilogy(D_loop, a_error_all[i], label=f'{int(npix_loop[i])} px')
plt.xlabel('Distance (m)')
plt.ylabel(' Log Max alpha error (deg) ')
plt.legend(title="Local angle(deg)", loc='upper right')
plt.title('Max alpha error vs Distance')
plt.show()



# %% LOOP OVER NUMBER OF PIXELS AT FIXED DISTANCE


D = 0.4  #fixed distance

# Range of detector sizes in pixels
npix_min = 1000
npix_max = 10000
n = 10
npix_loop = np.linspace(npix_min, npix_max, n) #pixel array 

# Number of samples 
N = 1000

#initialising total interfringe and local angle error lists
if_error_all = []
a_error_all = []

for pix in npix_loop:

    # GRID
    # pix is the detector size in pixels

    npix_arr = np.linspace(-pix/2, pix/2, N)

    # Physical detector coordinates in metres
    x = npix_arr * pix_size
    y = npix_arr * pix_size

    xx, yy = np.meshgrid(x, y)
    extent = [0, pix, 0, pix]

    # STRAIGHT MODEL
    dL_st = xx * B / D
    st = np.cos((2 * np.pi / lmd) * dL_st)

    # HYPERBOLIC MODEL
    L1 = np.sqrt((xx + B/2)**2 + yy**2 + D**2)
    L2 = np.sqrt((xx - B/2)**2 + yy**2 + D**2)
    dL_hyp = L2 - L1
    hyp = np.cos((2 * np.pi / lmd) * dL_hyp)

    # ERROR MAPS
    err_if = interfringe(dL_hyp, lmd, x, y, pix_size) - interfringe(dL_st, lmd, x, y, pix_size)   #Difference of interfringe between the two models
    err_alpha = alpha(dL_hyp, lmd, x, y) - alpha(dL_st, lmd, x, y)  

    # Store maximum absolute errors
    if_error_all.append(np.nanmax(np.abs(err_if)))
    a_error_all.append(np.nanmax(np.abs(err_alpha)))

#----------------------------------------------------------------------------------------------------------------

# MAX ERROR PLOTS:visulising results from loop for a fixed D

plt.figure()
plt.semilogy(npix_loop, if_error_all, marker='o')
plt.xlabel('Number of pixels')
plt.ylabel('Max interfringe error (pix)')
plt.title('Maximum interfringe error vs detector size\nD = 0.4 m')
plt.show()

plt.figure()
plt.semilogy(npix_loop, a_error_all, marker='o')
plt.xlabel('Number of pixels')
plt.ylabel('Max local angle error (deg)')
plt.title('Maximum local angle error vs detector size\nD = 0.4 m')
plt.show()


