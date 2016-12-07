"""
Calculate Chebyshev Pseudospectral differentiation matrix
"""

from numpy import *
from numpy.matlib import repmat

def chebdiff(N):
    """
    x,D=chebdiff(N)
    N - Number of collocation points needs to be even (should not be much greater than 100)
    x - location of collocation points returned as array of size N+1
    D - Chebyshev differentiation matrix (returned as array of size N+1
    
    Usage Example
    N = 100
    x,D = chebdiff(N)
    """
    if N==0:
        D=0
        x=1
    else:
        n = arange(0,N+1)
        x=cos(pi*n/N)
        #c = [2; ones((N-1,1)); 2].*(-1).^(0:N)';
        c=hstack((2,ones((N-1,)),2))*(-1)**n
        X = repmat(x,N+1,1).transpose()
        dX = X-X.transpose()
        D  = outer(c,(1.0/c).transpose())/(dX+(eye(N+1)))
        D  = D - diag(sum(D,axis=1))
        
    return x,D

def chebint(x,f):
    """
    Computes Chebyshev spectral integral of function f
    evaluated at Chebyshev collolcation points x
    f = chebint(f)
    
    """
    N = len(x)
    wi = pi/N
    return sum(sqrt(1-x**2)*f*pi*wi)