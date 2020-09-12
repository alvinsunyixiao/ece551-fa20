import numpy as np 
from scipy.interpolate import splev, splrep

def gen_wave(UIN,n0 = 0):
    T = np.arange(100)
    Ts = 1 + 3*np.arange(len(UIN))
    sp = splrep(Ts, UIN, k=3)
    return splev(T-n0,sp,ext=1)

def estimate_delay_and_gain(x1,x2):
    # Your algorithm Here
    return delta, rho

UIN = np.array([ Fill In your UIN here, comma separated ])
n1,n2 = randint(1,40,size = 2)
alpha1,alpha2 = randn(2)
x1, x2= genwave(alpha1*UIN, n1), genwave(alpha2*UIN, n2)

delta, rho = estimate_delay_and_gain(x1,x2)
# Then compare with n1-n2, alpha1/alpha2