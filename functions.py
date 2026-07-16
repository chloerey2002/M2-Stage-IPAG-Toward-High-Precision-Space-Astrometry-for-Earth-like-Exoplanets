import numpy as np

def grid(npix, N, pix_size):
    npix_min = -npix / 2
    npix_max = npix / 2

    fac = npix / N

    npix_min_fac = npix_min / fac
    npix_max_fac = npix_max / fac

    npix_arr = np.linspace(npix_min_fac, npix_max_fac, N)

    pitch = pix_size * fac
    c = npix_arr * pitch

    return c, pitch, fac


def straight(x, B, D, lmd):
    dL_st = (x) * B / D
    I_st = np.cos((2 * np.pi)/lmd * dL_st)
    return dL_st, I_st


def hyp(s1x, s2x, s1y, s2y, xx, yy, B, D, lmd):

    L1 = np.sqrt((xx + s1x)**2 + (yy + s1y)**2 + D**2)
    L2 = np.sqrt((xx + s2x)**2 + (yy + s2y)**2 + D**2)

    dL_hyp = L2 - L1
    phi = (2 * np.pi / lmd) * dL_hyp
    hyp_int = np.cos(phi)

    return dL_hyp, phi, hyp_int


def hyp_gaussian(xx, yy, B, D, lmd, w0):
    s1x = B / 2
    s2x = -B / 2

    L1 = np.sqrt((xx - s1x)**2 + yy**2 + D**2)
    L2 = np.sqrt((xx - s2x)**2 + yy**2 + D**2)

    dL = L2 - L1
    phi = (2 * np.pi / lmd) * dL

    I1 = np.exp(-2*((xx - s1x)**2 + yy**2) / w0**2)
    I2 = np.exp(-2*((xx - s2x)**2 + yy**2) / w0**2)

    I = I1 + I2 + 2*np.sqrt(I1 * I2) * np.cos(phi)

    return I


def interfringe(dL, lmd, x, y , pix_size):
    phi = (2 * np.pi / lmd) * dL

    dphi_dy, dphi_dx = np.gradient(phi, y, x)
    grad_phi = np.sqrt(dphi_dx**2 + dphi_dy**2)

    interfringe = 2 * np.pi / grad_phi
    interfringe_pix = interfringe / pix_size
    return interfringe_pix

def alpha(dL, lmd, x, y):
    phi = (2 * np.pi / lmd) * dL
    dphi_dy, dphi_dx = np.gradient(phi, y, x)

    alpha_rad = np.abs(np.arctan(dphi_dy/dphi_dx))
    alpha_deg = np.rad2deg(alpha_rad)
    return alpha_deg






def phi_model(s1x, rows, cols, xx, yy, B, D, lmd): 
    rx = xx[rows, cols] 
    ry = yy[rows, cols] 
    
    s1y = 0 
    s2y = 0  
    s2x = -B/2 
    
    L1 = np.sqrt((rx - s1x)**2 + (ry - s1y)**2 + D**2) 
    L2 = np.sqrt((rx - s2x)**2 + (ry - s2y)**2 + D**2) 
    
    dL = L2 - L1 
    phi = (2*np.pi/lmd) * dL 
    
    return phi # % (2*np.pi) 


#two params
def phi_model2(s1x, s2x, rows, cols, xx, yy, B, D, lmd): 
    rx = xx[rows, cols] 
    ry = yy[rows, cols] 
    
    s1y = 0 
    s2y = 0  
    
    L1 = np.sqrt((rx - s1x)**2 + (ry - s1y)**2 + D**2) 
    L2 = np.sqrt((rx - s2x)**2 + (ry - s2y)**2 + D**2) 
    
    dL = L2 - L1 
    phi = (2*np.pi/lmd) * dL 
    
    return phi # % (2*np.pi) 


def phi_full(s1x, xx, yy, B, D, lmd): 
    
    s1y = 0 
    s2y = 0 
    s2x = -B/2 
    
    L1 = np.sqrt((xx - s1x)**2 + (yy - s1y)**2 + D**2) 
    L2 = np.sqrt((xx - s2x)**2 + (yy - s2y)**2 + D**2) 
    
    dL = L2 - L1 
    phi = (2*np.pi/lmd) * dL 
    
    return (phi % (2*np.pi) )





