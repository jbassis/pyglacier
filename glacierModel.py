"""
Classes to solve shallow ice approximation and shelfy stream approximation to flowline glacier dynamics using the spectral method

"""

from numpy import *
from pylab import *

import scipy.sparse as sparse
import scipy.sparse.linalg
from scipy.sparse.linalg import spsolve as spsolve
from scipy.sparse import spdiags
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.optimize import minimize
from scipy.optimize import fsolve
from scipy.integrate import quadrature
from scipy.integrate import odeint
from scipy.integrate import quad
from scipy.integrate import dblquad
from numpy import interp
from scipy.signal import fftconvolve
import sys
from decimal import Decimal
from scipy.interpolate import UnivariateSpline as spline
#from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline as interp2d

from cheb import chebdiff


def update_progress(progress):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

# Calculate height of cliff, assuming it is at the yield strength
def calving_cliff_height(D,tau_c,mu=0.0,rho=910.0,rho_w=1028.0):
    """
    Calculate height of calving cliff based on yield strength tau_c,
    friction coefficient mu and water depth D.  For now, we don't include
    friction in these calculation.  If the bed is above sea level (z=0),
    the water depth is zero. 
    """
    #rho=910.0 # Density of ice kg/m^3
    #rho_w = 1028.0
    g = 9.8
    Hmax = 2*tau_c/(rho*g*(1-2*mu)) + sqrt((2*tau_c/(rho*g*(1-2*mu)))**2+rho_w/rho*D**2)
    #Hmax = 2*tau_c/rho/g + sqrt((2*tau_c/rho/g)**2-(4*tau_c/rho/g*D-rho_w/rho*D**2))
    return maximum(Hmax,rho_w/rho*D)



    

class Bed(object):
    def __init__(self,bed_function=None,max_length=4e6):
        self.rho_m = 3300.0
        self.rho = 900.0
        self.rho_w = 1028.0
        self.g = 9.81
        self.F = 5.0e24 # Flexural rigidity
        self.secpera = 31556926.0
        self.eta = 1.0e21 # Viscosity of the mantle (Pa yr)
        self.tau = 1e3 # Relaxation time in yrs
        self.viscosity_ratio = 0.01
        self.Tc = 100e3
        self.L0 = max_length
        self.N = 2048
        self.xx = linspace(-self.L0,self.L0,self.N)
        self.k = fftfreq(len(self.xx), d=mean(abs(diff(self.xx))))
        if bed_function==None:
            def bed_function(x):
                return x*0
        self.bed = bed_function(self.xx)
        self.bed_interp = bed_function
        self.bed_init = bed_function(self.xx)
        try:
            self.I = load("I_3000_"+str(self.N)+".npy")
        except:  
            self.__elastic_greens_function_init__()
        
            
        self.displace=None
        self.H_previous = None
        self.sea_level = 0.0
        self.layers = 2
        self.bed_old = self.bed
        
    def viscous_plate(self,x,H,dt,sym=True):
        """
        Compute viscoelastic relaxation based on Bueler, Lingle and Brown
        """
        
        xx =self.xx
        k = 2*pi*abs(self.k) 
        dt = dt*self.secpera
        
        if sym==True:
            xi=hstack([x,-x[-1:0:-1]])
            Hi=hstack([H,H[-1:0:-1]])
            H_interp=interp1d(xi,Hi,'linear',bounds_error=False, fill_value=0.0)(xx)
        else:
            H_interp=interp1d(xi,Hi,'linear',bounds_error=False, fill_value=x[0])(xx)
        

        if self.displace == None:
            w=self.elastic_plate(x,H)
            self.displace=w
         
        if self.layers==3:
            Tc = self.Tc # Thickness of channel in m
            kappa = abs(k)
            c = cosh(Tc*kappa)
            s = sinh(Tc*kappa)
            r = self.viscosity_ratio # Ratio of viscosities
            num = (2*r*c*s+(1-r**2)*Tc**2*kappa**2+r**2*s**2+c**2)
            denom = (r+1.0/r)*c*s +(r-1.0/r)*Tc*kappa + s**2+c**2
            R = num/denom
        else:
            R = 1.0
            
        #t1 = 2*R*self.eta*(1+self.rho_m*self.g/1e11/2/(k+1e-32))*sqrt(k**2)/(dt*self.rho_m*self.g)
        t1 = 2*R*self.eta*sqrt(k**2)/(dt*self.rho_m*self.g)
        t2 = 1 + self.F*k**4/(self.rho_m*self.g)
        bed_new = 0.5*(self.__call__(xx) + self.bed_old)
        bed_init = self.bed_init
        water_load_new = bed_new*(bed_new<0.0)*(H_interp<=0)
        water_load_init = bed_init*(bed_init<0.0)
        
        #max_width = 180e3
        water_load_init = water_load_init#*max_width/width(xx)
        water_load_new = water_load_new#*max_width/width(xx)
        H_interp = H_interp#*max_width/width(xx)
        sigma_zz = -self.rho_w/self.rho_m*fft(water_load_new) -  self.rho/self.rho_m*fft(H_interp) + self.rho_w/self.rho_m*fft(water_load_init)#- 0*self.rho_w/self.rho_m*fft(disp)
    
        frhs =  (t1*fft(self.displace) + sigma_zz)/(t1 + t2)
        
        w=real(ifft( frhs )) 
    
        return w
        
    
        
    def update(self,x,H,dt):
        self.sea_level = self.sea_level
        wv=self.viscous_plate(x,H,dt)
        we =  0.0
        #we=self.elastic(x,H)
        self.displace = wv
        self.bed_interp=spline(self.xx,self.bed+wv+we+self.sea_level,s=0.0)
        self.bed_old = self.bed+wv+self.sea_level
       
    
        
    def __elastic_greens_function_init__(self):
        rm=array([ 0.0,0.011, 0.111, 1.112,  2.224,  3.336,  4.448,  6.672,  8.896,  11.12, 17.79,
                22.24,  27.80,  33.36,  44.48,  55.60,  66.72,  88.96,  111.2,  133.4, 177.9,
                222.4,  278.0,  333.6,  444.8,  556.0,  667.2,  778.4,  889.6, 1001.0, 1112.0,
                1334.0, 1779.0, 2224.0, 2780.0, 3336.0, 4448.0, 5560.0, 6672.0, 7784.0, 8896.0,
                10008.0])*1e3  # converted to meters
        # GE /(10^12 rm) is vertical displacement in meters
        GE=array([-33.6488, -33.64, -33.56, -32.75, -31.86, -30.98, -30.12, -28.44, -26.87, -25.41,
                -21.80, -20.02, -18.36, -17.18, -15.71, -14.91, -14.41, -13.69, -13.01,
                -12.31, -10.95, -9.757, -8.519, -7.533, -6.131, -5.237, -4.660, -4.272,
                -3.999, -3.798, -3.640, -3.392, -2.999, -2.619, -2.103, -1.530, -0.292,
                 0.848,  1.676,  2.083,  2.057,  1.643])
        
        #GE_interp = spline(rm,GE)
        GE_interp = interp1d(rm,GE,'linear',bounds_error=False,fill_value=0.0)
        
        # Replace normalization constant with rm[1] to avoid singularity at the origin
        rd = rm.copy()
        rd[0]=rm[1]
        
        
        # Compute integral approximation of Greens function for each radius point
        xx = self.xx
        dx = mean(abs(diff(xx)))
        N = len(xx)
        I = zeros(size(xx))
        
        print "Initializaing Elastic Green's function . . . this will take a while"
        for i in xrange(N):
            #print i,float(i)/N,N
            #if mod(i,8)==0:
            #    sys.stdout.write("-")
            #    sys.stdout.flush()
            update_progress(round(float(i)/N,2))
            xi = xx[i]
            
            def integrand(eta,nu):
                
                r=(sqrt((xi-eta)**2+(xi-nu)**2))
                if r>=rm[-1]:
                    z=0.0
                elif r==0.0:
                    z=GE[0]/(rm[1]*1e12)
                else:
                    z=GE_interp(r)/(r*1e12)
                #for jj in xrange(len(z)):
                #    print jj,r[(jj)]
                #    if r[(jj)]>=rm[-1]:
                #        z[(jj)]=0
                #    elif r[(jj)]==0.0:
                #        z[(jj)]=GE[0]/(rm[1]*1e12)
                #    else:    
                #        z=GE_interp(r)/(r*1e12).reshape((size(r),))
                #print r,z
                
                return z
            #print dx
            I[i],err=dblquad(integrand,-0.5*dx,0.5*dx,lambda nu: -25e3, lambda nu: 25e3)
        self.I = I
        fname = "I_3000_"+str(self.N)+".npy"
        save(fname,I)
        self.I_interp = spline(self.xx,self.I)
            
            
    def elastic(self,x,H,sym=True):
        xx=self.xx
        if sym==True:
            xi=hstack([x,-x[-1:0:-1]])
            Hi=Hi=hstack([H,H[-1:0:-1]])
            H_interp=interp1d(xi,Hi,'linear',bounds_error=False, fill_value=0.0)(xx)
        else:
            H_interp=interp1d(xi,Hi,'linear',bounds_error=False, fill_value=x[0])(xx)
        
        #sigma_zz = self.rho*self.g*H_interp
        
        bed_new = 0.5*(self.__call__(xx) + self.bed_old)
        bed_init = self.bed_init
        water_load_new = bed_new*(bed_new<0.0)*(H_interp<=0)
        water_load_init = bed_init*(bed_init<0.0)
        sigma_zz = -self.rho_w*self.g*(water_load_new) -  self.rho*self.g*(H_interp) + self.rho_w*self.g*(water_load_init)#- 0*self.rho_w/self.rho_m*fft(disp)
        ue = fftconvolve(-sigma_zz,self.I,'same')
        return ue
        
    
    #def elastic(self,x,H,sym=True):
    #    xx =self.xx
    #    k = 2*pi*abs(self.k)
    #   
    #    if sym==True:
    #        xi=hstack([x,-x[-1:0:-1]])
    #        Hi=hstack([H,H[-1:0:-1]])
    #        H_interp=interp1d(xi,Hi,'linear',bounds_error=False, fill_value=0.0)(xx)
    #    else:
    #        H_interp=interp1d(xi,Hi,'linear',bounds_error=False, fill_value=x[0])(xx)
    #    
    #    bed_new = 0.5*(self.__call__(xx) + self.bed_old)
    #    bed_init = self.bed_init
    #    water_load_new = bed_new*(bed_new<0.0)*(H_interp<=0)
    #    water_load_init = bed_init*(bed_init<0.0)
    #    sigma_zz = -self.rho_w*self.g*fft(water_load_new) -  self.rho*self.g*fft(H_interp) + self.rho_w*self.g*fft(water_load_init)#- 0*self.rho_w/self.rho_m*fft(disp)
    #    #ue = real(ifft((1/self.mu + 1/(self.mu+self.lam))*fft(sigma_zz)/(2*(k+1e-16)/self.mu)))
    #    rho_b = 10000
    #    t2 = (1 + self.F*k**4/(rho_b*self.g))
    #    #ue = real(ifft((1/self.mu + 1/(self.mu+self.lam))/2/(k+alpha)*sigma_zz))
    #    #G= (1/self.mu + 1/(self.lam+self.mu))**(-1)
    #    #S=sinh(k*self.Te)
    #    #C=cosh(k*self.Te)
    #    #t2 = (2*k*self.mu/(self.rho*self.g)*((self.lam+self.mu)/(self.lam+2*self.mu))*(S**2-k**2*self.Te**2)+(C*S+k*self.Te))/(S+(k+1e-16)*self.Te*C)
    #    #t2[0]=1.0
    #    G = self.mu
    #    ue = real(ifft(sigma_zz/(2*G*k/(rho_b*self.g) + t2)))/(rho_b*self.g)
    #    #ue = real(ifft(((1/self.mu + 1/(self.mu+self.lam))/self.mu/(k+1e-16)/2)*sigma_zz))
    #    return ue
        
    def topo(self):
        return self.xx,self.bed  
        
    def __call__(self,x):
        #print 'Shape x',shape(x)
        #print 'Shape xx',shape(self.xx),max(self.xx)
        return self.bed_interp(x)

    
    def elastic_plate(self,x,H,sym=True):
        """
        Calculate elastic plate deflection based on ice load w
        """
        
        F = self.F
        rho_m = self.rho_m
        xx=self.xx
        k = 2*pi*abs(self.k)
        if sym==True:
            xi=hstack([x,-x[-1:0:-1]])
            Hi=Hi=hstack([H,H[-1:0:-1]])
            H_interp=interp1d(xi,Hi,'linear',bounds_error=False, fill_value=0.0)(xx)
        else:
            H_interp=interp1d(xi,Hi,'linear',bounds_error=False, fill_value=x[0])(xx)
        
        #bed = self.bed 
        #water_depth = -bed*(bed<0.0)*(H_interp>0)
        
        #bed_fft = fft(water_depth)
        #k=fftfreq(len(xx), d=mean(abs(diff(xx))))
        
        #bed_new = 0.5*(self.__call__(xx) + self.bed_old)
        #bed = (bed_new - self.bed)
        #bed = self.bed 
        #water_depth = -bed*(bed_new<0.0)
        #water_load = water_depth*(H_interp<=0)
        #disp = self.displace*((self.bed + self.displace)<0.0)*H_interp==0
        #sigma_zz = self.rho_w/self.rho_m*fft(water_depth) - self.rho/self.rho_m*fft(H_interp) #- 0*self.rho_w/self.rho_m*fft(disp)
        #sigma_zz = -self.rho_w/self.rho_m*fft(water_load) -  self.rho/self.rho_m*fft(H_interp) #- 0*self.rho_w/self.rho_m*fft(disp)
        
        
        bed_new = 0.5*(self.__call__(xx) + self.bed_old)
        bed_init= self.bed_init
        water_load_new = bed_new*(bed_new<0.0)*(H_interp<=0)
        water_load_init = bed_init*(bed_init<0.0)

        sigma_zz = -self.rho_w/self.rho_m*fft(water_load_new) -  self.rho/self.rho_m*fft(H_interp) + self.rho_w/self.rho_m*fft(water_load_init)#- 0*self.rho_w/self.rho_m*fft(disp)
        
        
        H_fft = fft(H_interp)
        
        #w_fft = (self.rho_w/rho_m*bed_fft-self.rho/rho_m*H_fft)/(1+F*k**4/(rho_m*self.g)) -(self.rho_w/rho_m*bed_fft)/(1+F*k**4/(rho_m*self.g))
        #w_fft = -self.rho/rho_m*H_fft/(1+F*k**4/(rho_m*self.g)) + self.rho_w/rho_m*bed_fft/(1+F*k**4/(rho_m*self.g))
        w_fft = sigma_zz/(1+F*k**4/(rho_m*self.g))
        w = real(ifft(w_fft))
        
        return w
        
class Bed2D_full(Bed):
    def __init__(self,bed_function,max_length=4e6,fill=0.0):
        self.rho_m = 3300.0
        self.rho = 900.0
        self.rho_w = 1028.0
        self.g = 9.81
        self.F = 5.0e24 # Flexural rigidity
        self.secpera = 31556926.0
        self.eta = 1.0e21 # Viscosity of the mantle (Pa yr)
        self.tau = 1e3 # Relaxation time in yrs
        self.viscosity_ratio = 0.01
        self.Tc = 100e3
        self.L0 = max_length
        self.Nx = 512
        self.Ny = 512
        self.Ly = max_length
        self.xx = linspace(-self.L0,self.L0,self.Nx)
        self.yy = linspace(-self.L0,self.L0,self.Nx)
        self.Y,self.X = meshgrid(self.yy,self.xx)
        self.kx = fftfreq(len(self.xx), d=mean(abs(diff(self.xx))))
        self.ky = fftfreq(len(self.yy), d=mean(abs(diff(self.yy))))
        
        self.bed_interp = bed_function
        self.displace = zeros(shape(self.X))
        self.H_previous = None
        self.sea_level = 0.0
        self.layers = 2
        self.bed = bed_function(self.X,self.Y)
        self.bed_init = bed_function(self.X,self.Y)
        self.bed_old = self.bed_init
        self.disp_interp = interp2d(self.xx,self.yy,zeros(shape(self.bed)))
        
    def viscous_plate(self,X,Y,H,dt,sym=True):
        """
        Compute viscoelastic relaxation based on Bueler, Lingle and Brown
        """
        #xx =self.xx
        #yy =self.yy
        xx= self.X
        yy= self.Y
        
        ky,kx = meshgrid(2*pi*self.ky,2*pi*self.kx)
        dt = dt*self.secpera
        
        # 2D interpolated result with fill values???  # Pad array with values for extrapolation !
        yi = vstack((Y[0,:]*0 + amin(yy),Y,Y[-1,:]*0+amax(yy))) # Pad top and bottom of array
        xi = vstack((X[0,:] ,X,X[-1,:]))
        Hi = vstack((H[0,:]*0,H,0*H[-1]))
        
        #yi = Y
        #xi = X
        #Hi = H
    
        # Pad end of array
        #xi = vstack((xi[:,0]+xi[:,0]-xi[:,1],xi.transpose())).transpose()
        #yi = vstack((yi[:,0],yi.transpose())).transpose()
        #Hi = vstack((0*yi[:,0],Hi.transpose())).transpose()
        
        
        from scipy.interpolate import griddata
        points = vstack((xi.ravel(),yi.ravel())).transpose()
        values = Hi.ravel()
        H_interp = griddata(points, values, (abs(xx),yy), method='linear',fill_value=0.0)
        # Need to reverse arrays so x is strictly ascending
        # Check that y is also ascending
        # Then interpolate
        #w = (yi[0,:]-yi[-1,:])
        
        # Reverse array
        #xi=xi[::-1,::-1]
        #yi=yi[::-1,::-1]
        #Hi=Hi[::-1,::-1]
        #yyy = yy/self.width(abs(xx))
        #from scipy.interpolate import RectBivariateSpline as interps
        #H_interp=interps(xi[0,:], yi[:,0]/w[0],Hi.transpose())
        #H_interp = reshape(H_interp(abs(xx.ravel()),yy.ravel(),grid=False),shape(xx))
        #H_interp=H_interp(abs(xx).ravel(),yyy.ravel(),grid=False)
        #yy=y/width(abs(x))
        #H = f(abs(xx),yy,grid=False)
        
        
        kappa = sqrt(kx**2+ky**2)
        if self.layers==3:
            Tc = self.Tc # Thickness of channel in m
            c = cosh(Tc*kappa)
            s = sinh(Tc*kappa)
            r = self.viscosity_ratio # Ratio of viscosities
            num = (2*r*c*s+(1-r**2)*Tc**2*kappa**2+r**2*s**2+c**2)
            denom = (r+1.0/r)*c*s +(r-1.0/r)*Tc*kappa + s**2+c**2
            R = num/denom
        else:
            R = 1.0
            
        t1 = 2*R*self.eta*kappa/(dt*self.rho_m*self.g)
        t2 = 1 + self.F*kappa**4/(self.rho_m*self.g)
        
        # Make 2D???
        bed_new = 0.5*(self.__call__(xx,yy) + self.bed_old)
        
        
        bed_init = self.bed_init
        water_load_new = bed_new*(bed_new<0.0)*(H_interp<=0)
        water_load_init = bed_init*(bed_init<0.0)
        
    
        sigma_zz = -self.rho_w/self.rho_m*fft2(water_load_new) -  self.rho/self.rho_m*fft2(H_interp) + self.rho_w/self.rho_m*fft2(water_load_init)
        
        frhs =  (t1*fft2(self.displace) + sigma_zz)/(t1 + t2)
        
        w=real(ifft2( frhs ))
    
        return w
        
    def update(self,x,y,H,dt):
        wv=self.viscous_plate(x,y,H,dt,sym=True)
        self.displace = wv
        self.disp_interp= interp2d(self.xx,self.yy,wv)
        self.bed_old = self.bed_init+wv
        
    def __call__(self,x,y):
        bed1 = self.bed_interp(x,y)
        if size(x)>1:
            xx=x.flatten()
        else:
            xx=x
        if size(y)>1:
            yy=y.flatten()
        else:
            yy=y
        bed2 = reshape(self.disp_interp(xx,yy,grid=False),shape(x))

        return bed1 + bed2
    

        
    
    
class Bed2D(Bed):
    def __init__(self,bed_function=None,max_length=4e6):
        self.rho_m = 3300.0
        self.rho = 900.0
        self.rho_w = 1028.0
        self.g = 9.81
        self.F = 5.0e24 # Flexural rigidity
        self.secpera = 31556926.0
        self.eta = 1.0e21 # Viscosity of the mantle (Pa yr)
        self.tau = 1e3 # Relaxation time in yrs
        self.viscosity_ratio = 0.01
        self.Tc = 100e3
        self.L0 = max_length
        self.N = 2048
        self.Ny = 3
        self.Ly = 2500e3
        self.xx = linspace(-self.L0,self.L0,self.N)
        self.yy = linspace(-self.Ly/2,self.Ly/2,self.Ny)
        self.X,self.Y = meshgrid(self.xx,self.yy)
        self.k = fftfreq(len(self.xx), d=mean(abs(diff(self.xx))))
        if bed_function==None:
            def bed_function(x):
                return x*0
        #self.bed = bed_function(self.xx)
        self.bed_interp = bed_function
        #self.bed_init = bed_function(self.xx)
        
        self.displace=None
        self.H_previous = None
        self.sea_level = 0.0
        self.layers = 2
        
        self.bed = bed_function(self.X,self.Y)
        #self.bed_init = trapz(self.bed,axis=0)/2
        self.bed_init = bed_function(self.xx,0.0)
        self.bed_old = bed_function(self.xx,0.0)
        self.disp_interp=spline(self.xx,self.xx*0,s=0.0)

        
        
    def elastic_plate(self,x,H,sym=True):
        """
        Calculate elastic plate deflection based on ice load w
        """
        
        F = self.F
        rho_m = self.rho_m
        xx=self.xx
        k = 2*pi*abs(self.k)
        if sym==True:
            xi=hstack([x,-x[-1:0:-1]])
            Hi=Hi=hstack([H,H[-1:0:-1]])
            H_interp=interp1d(xi,Hi,'linear',bounds_error=False, fill_value=0.0)(xx)
        else:
            H_interp=interp1d(xi,Hi,'linear',bounds_error=False, fill_value=x[0])(xx)
        
        #bed = self.bed 
        #water_depth = -bed*(bed<0.0)*(H_interp>0)
        
        #bed_fft = fft(water_depth)
        #k=fftfreq(len(xx), d=mean(abs(diff(xx))))
        
        #bed_new = 0.5*(self.__call__(xx) + self.bed_old)
        #bed = (bed_new - self.bed)
        #bed = self.bed 
        #water_depth = -bed*(bed_new<0.0)
        #water_load = water_depth*(H_interp<=0)
        #disp = self.displace*((self.bed + self.displace)<0.0)*H_interp==0
        #sigma_zz = self.rho_w/self.rho_m*fft(water_depth) - self.rho/self.rho_m*fft(H_interp) #- 0*self.rho_w/self.rho_m*fft(disp)
        #sigma_zz = -self.rho_w/self.rho_m*fft(water_load) -  self.rho/self.rho_m*fft(H_interp) #- 0*self.rho_w/self.rho_m*fft(disp)
        
        
        bed_new = 0.5*(self.__call__(xx,0.0) + self.bed_old)
        bed_init= self.bed_init
        water_load_new = bed_new*(bed_new<0.0)*(H_interp<=0)
        water_load_init = bed_init*(bed_init<0.0)

        sigma_zz = -self.rho_w/self.rho_m*fft(water_load_new) -  self.rho/self.rho_m*fft(H_interp) + self.rho_w/self.rho_m*fft(water_load_init)#- 0*self.rho_w/self.rho_m*fft(disp)
        
        
        H_fft = fft(H_interp)
        
        #w_fft = (self.rho_w/rho_m*bed_fft-self.rho/rho_m*H_fft)/(1+F*k**4/(rho_m*self.g)) -(self.rho_w/rho_m*bed_fft)/(1+F*k**4/(rho_m*self.g))
        #w_fft = -self.rho/rho_m*H_fft/(1+F*k**4/(rho_m*self.g)) + self.rho_w/rho_m*bed_fft/(1+F*k**4/(rho_m*self.g))
        w_fft = sigma_zz/(1+F*k**4/(rho_m*self.g))
        w = real(ifft(w_fft))
        
        return w

    def viscous_plate(self,X,Y,H,dt,sym=True):
        """
        Compute viscoelastic relaxation based on Bueler, Lingle and Brown
        """
        x=X[0,:]
        #H0 =-trapz(H,self.sigma_y,axis=0)/2
        H =-trapz(H,Y,axis=0)/self.width(x)
        #print H0
        #print H1
        xx =self.xx
        k = 2*pi*abs(self.k) 
        dt = dt*self.secpera
        if sym==True:
            xi=hstack([x,-x[-1:0:-1]])
            Hi=hstack([H,H[-1:0:-1]])
            H_interp=interp1d(xi,Hi,'linear',bounds_error=False, fill_value=0.0)(xx)
        else:
            H_interp=interp1d(xi,Hi,'linear',bounds_error=False, fill_value=x[0])(xx)
        

        if self.displace == None:
            w=self.elastic_plate(x,H)
            self.displace=w
         
        if self.layers==3:
            Tc = self.Tc # Thickness of channel in m
            kappa = abs(k)
            c = cosh(Tc*kappa)
            s = sinh(Tc*kappa)
            r = self.viscosity_ratio # Ratio of viscosities
            num = (2*r*c*s+(1-r**2)*Tc**2*kappa**2+r**2*s**2+c**2)
            denom = (r+1.0/r)*c*s +(r-1.0/r)*Tc*kappa + s**2+c**2
            R = num/denom
        else:
            R = 1.0
            
        #t1 = 2*R*self.eta*(1+self.rho_m*self.g/1e11/2/(k+1e-32))*sqrt(k**2)/(dt*self.rho_m*self.g)
        t1 = 2*R*self.eta*sqrt(k**2)/(dt*self.rho_m*self.g)
        t2 = 1 + self.F*k**4/(self.rho_m*self.g)
        bed_new = 0.5*(self.__call__(xx,0.0) + self.bed_old)
        bed_init = self.bed_init
        water_load_new = bed_new*(bed_new<0.0)*(H_interp<=0)
        water_load_init = bed_init*(bed_init<0.0)
        
        #max_width = 180e3
        water_load_init = water_load_init#*max_width/width(xx)
        water_load_new = water_load_new#*max_width/width(xx)
        H_interp = H_interp#*max_width/width(xx)
        sigma_zz = -self.rho_w/self.rho_m*fft(water_load_new) -  self.rho/self.rho_m*fft(H_interp) + self.rho_w/self.rho_m*fft(water_load_init)#- 0*self.rho_w/self.rho_m*fft(disp)
        
        frhs =  (t1*fft(self.displace) + sigma_zz)/(t1 + t2)
        
        w=real(ifft( frhs ))

        return w
        
    def update(self,X,Y,H,dt):
        #x=X[0,:]
        #H =-trapz(H,Y,axis=0)/self.width(x)
        
        wv=self.viscous_plate(X,Y,H,dt,sym=True)
        #print shape(wv)
        self.displace = wv
        #Nx,Ny=shape(self.X)
        #print shape(wv)
        #wv = tile(wv,(self.Ny,1)) # Tile the displacement so it is the same across y
        #print shape(self.xx),shape(self.yy),shape(wv),shape(self.bed)
        #self.bed_interp=interp(self.xx,wv,s=0)
        self.disp_interp=spline(self.xx,wv,s=0.0)
        self.bed_old = self.bed_init+wv+self.sea_level
        
    def __call__(self,x,y):
        S=shape(x)
        if len(S)<=1:
            w = self.disp_interp(x)
        else:
            w = self.disp_interp(x[0,:])
            w = tile(w,(S[0],1))
        return self.bed_interp(x,y)+w
    
    
class Bed_ELRA2D(Bed2D):
    def __init__(self,bed_function=None,max_length=4e6):
        self.rho_m = 3300.0
        self.rho = 900.0
        self.rho_w = 1028.0
        self.g = 9.81
        self.F = 5.0e24 # Flexural rigidity
        self.secpera = 31556926.0
        self.eta = 1.0e21 # Viscosity of the mantle (Pa yr)
        self.tau = 1e3 # Relaxation time in yrs
        self.viscosity_ratio = 0.01
        self.Tc = 100e3
        self.L0 = max_length
        self.N = 2048
        self.Ny = 3
        self.Ly = 2500e3
        self.xx = linspace(-self.L0,self.L0,self.N)
        self.yy = linspace(-self.Ly/2,self.Ly/2,self.Ny)
        self.X,self.Y = meshgrid(self.xx,self.yy)
        self.k = fftfreq(len(self.xx), d=mean(abs(diff(self.xx))))
        if bed_function==None:
            def bed_function(x):
                return x*0
        #self.bed = bed_function(self.xx)
        self.bed_interp = bed_function
        #self.bed_init = bed_function(self.xx)
        
        self.displace=None
        self.H_previous = None
        self.sea_level = 0.0
        self.layers = 2
        
        self.bed = bed_function(self.X,self.Y)
        #self.bed_init = trapz(self.bed,axis=0)/2
        self.bed_init = bed_function(self.xx,0.0)
        self.bed_old = bed_function(self.xx,0.0)
        self.disp_interp=spline(self.xx,self.xx*0,s=0.0)
        
    def update(self,x,H,dt):
        wv=self.viscous_plate(x,H,dt,sym=True)
        #print shape(wv)
        self.displace = wv
        #Nx,Ny=shape(self.X)
        #print shape(wv)
        #wv = tile(wv,(self.Ny,1)) # Tile the displacement so it is the same across y
        #print shape(self.xx),shape(self.yy),shape(wv),shape(self.bed)
        #self.bed_interp=interp(self.xx,wv,s=0)
        self.disp_interp=spline(self.xx,wv,s=0.0)
        self.bed_old = self.bed_init+wv+self.sea_level
        
    def __call__(self,x,y):
        S=shape(x)
        if len(S)<=1:
            w = self.disp_interp(x)
        else:
            w = self.disp_interp(x[0,:])
            w = tile(w,(S[0],1))
        return self.bed_interp(x,y)+w
    
    def viscous_plate(self,x,H,dt,sym=True):
        """
        Compute viscoelastic relaxation based on Bueler, Lingle and Brown
        """
        
        xx =self.xx
        #k = self.k
        #k = maximum(kk,2*pi/3000e3)
        dt = dt*self.secpera
        
        if sym==True:
            xi=hstack([x,-x[-1:0:-1]])
            Hi=hstack([H,H[-1:0:-1]])
            H_interp=interp1d(xi,Hi,'linear',bounds_error=False, fill_value=0.0)(xx)
        else:
            H_interp=interp1d(xi,Hi,'linear',bounds_error=False, fill_value=x[0])(xx)
        
        
        if self.displace == None:
            w=self.elastic_plate(x,H)
            self.displace=w
            #self.displace = zeros(shape(xx))
         
        w_ref=self.elastic_plate(x,H)
        w = self.displace - (self.displace-w_ref)/self.tau*(dt/self.secpera) 
    
        return w
    

class Bed_ELRA(Bed):
    def __init__(self,bed_function=None,max_length=4e6):
        self.rho_m = 3300.0
        self.rho = 900.0
        self.rho_w = 1028.0
        self.g = 9.81        
        self.F = 5.0e24 # Flexural rigidity
        self.secpera = 31556926.0
        self.tau = 1e3 # Relaxation time in yrs
        self.L0 = max_length
        self.N = 1024
        self.xx = linspace(-self.L0,self.L0,self.N)
        self.k = fftfreq(len(self.xx), d=mean(abs(diff(self.xx))))
        self.bed = bed_function(self.xx)
        self.bed_interp = bed_function
        self.bed_init = bed_function(self.xx)
        self.displace=None
        self.H_previous = None
        self.sea_level = 0.0
        #try:
        #    self.I = load("I_3000_"+str(self.N)+".npy")
        #except:  
        #    self.__elastic_greens_function_init__()
        self.sea_level = 0.0
        self.bed_old = self.bed
       
    def viscous_plate(self,x,H,dt,sym=True):
        """
        Compute viscoelastic relaxation based on Bueler, Lingle and Brown
        """
        
        xx =self.xx
        #k = self.k
        #k = maximum(kk,2*pi/3000e3)
        dt = dt*self.secpera
        
        if sym==True:
            xi=hstack([x,-x[-1:0:-1]])
            Hi=hstack([H,H[-1:0:-1]])
            H_interp=interp1d(xi,Hi,'linear',bounds_error=False, fill_value=0.0)(xx)
        else:
            H_interp=interp1d(xi,Hi,'linear',bounds_error=False, fill_value=x[0])(xx)
        
        
        if self.displace == None:
            w=self.elastic_plate(x,H)
            self.displace=w
            #self.displace = zeros(shape(xx))
         
        w_ref=self.elastic_plate(x,H)
        w = self.displace - (self.displace-w_ref)/self.tau*(dt/self.secpera) 
    
        return w

class Glacier(object):
    g = 9.81 # Acceleration due to gravity (m,/s^2)
    secpera = 31556926.0  # Number of seconds per annum 
    def __init__(self,B,tau_c,mu=0.0,p=1.0/3.0,C=7.624e6,n=3.0,rho=900.0,rho_w=1028.0,plastic=True):
        self.B=B    # Rate constant for ice (Pa s^{1/n)})
        self.n=n    # Flow law exponent for ice rheology
        self.tau_c = tau_c # Yield strength of ice (Pa)
        self.mu = mu # Coefficient of friction (dimensionless)
        self.p = p # Exponent in sliding law (dimensionless)
        self.C=C  # Constant coefficient in sliding law (Pa (m/s)^{1/p})
        self.rho=rho# Density of ice kg/m^3
        self.rho_w = rho_w# Density of water kg/m^3
        self.plastic = plastic
        self.time = []
        self.time.append(0.0)
   

    def get_time(self):
        return self.time[-1]




        
    
    
class Model(Glacier):
    def __init__(self, N,*args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)    
        self.N = N
        [sigma,D]=chebdiff(N)
        self.sigma = (sigma+1)/2 # Definse sigma so that it goes from [0,1]
        self.D = 2*D 
        self.bed_topo=None
        self.width=None
        self.tau_b = 0.0
        self.u = array([0.0])
        self.Vc = 0.0
        
    def accum(self,x,hsurf,time):
        return 0.1*ones(size(x))
        
    def set_geometry(self,function_list):
        self.bed_topo = function_list[0]
        self.width = function_list[1]
        
    def set_accumulation(self,accum):
        self.accum = accum
    
    def get_sigma(self,L):
        """Returns horizontal coordinate"""
        return self.sigma*L
    
    def water_depth(self,x):
        """
        Calculate water depth based on bed topography
        water depth is positive if the bed is beneath sea level
        and zero otherwise
        """
        bed=self.bed_topo(x)
        D = -bed*(bed<0.0)
        return D
    
    def set_initial_cond(self,hsurf, H, L):
        """
        Initial condition
            hsurf = surface elevation
            H = ice thickness
            L = length of the glacier from ice divide to terminus
        """
        # Check to make sure each array has the right dimensions
        #if (len(hsurf))!=self.N:
        #    print "hsurf doesn't have the right number of grid points"
        #if (len(H))!=self.N:
        #    print "H doesn't have the right number of grid points"
        
        # Initialize surface elevation and ice thickness  
        self.hsurf = hsurf
        self.H = H
        self.L = L
        self.H_old = H
    
    
    def check_initialization(self):
        flag = 0
        if (self.N == None):
            print 'Need to initialize grid first'
            flag=1
        
        if (self.accum == None):
            print 'Need to initialize accumulation rate first'
            flag=2
        
        if (self.bed_topo==None):
            print 'Need to initialize bed topography first'
            flag=3
        
        if (self.width == None):
            print "Need to initialize glacier width"
            flag=4
        return flag


    def get_flux(self,x,hsurf,time):
        
        if size(hsurf)>1:
            h = interp1d(x,hsurf,kind='cubic')
            L=max(x)
        else:
            L=x

        def integrand(x):
            if size(hsurf)>1:
                accum_rate,front=self.accum(x,h(x),time)
            else:
                accum_rate,front=self.accum(x,hsurf,time)
            #print x/1e3,h(x),accum_rate
            q=accum_rate*self.width(x)
            return q
        Q,tol=quad(integrand,0,L,epsrel=0.1)
        #a,f = self.accum(x,0.0,0.0)
        #Q = x*a
        return Q
        """
        xx=linspace(0,L,1001)
        dx = xx[1]-xx[0]
        w=self.width(xx)
        Q = trapz(integrand(xx),xx)
        #accum_rate,front=self.accum(xx,h(xx),time)
        #Q = sum(accum_rate*w)*dx
        #a,f = self.accum(x,0.0,0.0)
        #Q = x*a
        return Q
        """
    def get_area(self,x):
        Area,tol=quad(width,0,L)
        return Area
    
    # Calculate height of cliff, assuming it is at the yield strength
    def calving_cliff_height(self,D): # Can eliminate all arguments??
        """
        Calculate height of calving cliff based on yield strength tau_c,
        friction coefficient mu and water depth D.  For now, we don't include
        friction in these calculation.  If the bed is above sea level (z=0),
        the water depth is zero. 
        """
        
        g = self.g
        rho_w = self.rho_w
        rho = self.rho
        mu = self.mu
        tau_c = self.tau_c
        #D = self.water_depth(self.L)
        Hmax = 2*tau_c/(rho*g*(1-2*mu)) + sqrt((2*tau_c/(rho*g*(1-2*mu)))**2+rho_w/rho*D**2)
        Hf = rho_w/rho*D
        Hmax = maximum(Hf,Hmax)
        #Hmax = Hf
        #Hmax = 0.5*(Hf+Hmax)
        return Hmax


class Model2D(Glacier):
    def __init__(self, Nx,Ny, *args, **kwargs):
        super(Model2D, self).__init__(*args, **kwargs)    
        self.Nx = Nx
        self.Ny = Ny 
        [sigma,Dx]=chebdiff(Nx)
        self.sigma_x = (sigma+1)/2 # Definse sigma so that it goes from [0,1]
        self.Dx = 2*Dx
        [sigma,Dy]=chebdiff(Ny)
        self.Dy = Dy
        self.sigma_y = sigma
        self.bed_topo=None
        self.width=None
        self.tau_b = 0.0
        self.u = array([0.0])
        self.Vc = 0.0
        Di = inv(self.Dy[0:-1,0:-1])
        self.quad_weights = hstack((Di[0,:],0.0))
        self.ux = zeros((Ny+1,Nx+1))
        self.uy = zeros((Ny+1,Nx+1))
        #self.quad = inv(self.Dy)
        
    def set_geometry(self,function_list):
        self.bed_topo = function_list[0]
        self.width = function_list[1]
        
    def set_accumulation(self,accum):
        self.accum = accum
    
    def get_sigma_x(self,L):
        """Returns horizontal coordinate"""
        return self.sigma_x*L
    
    def get_sigma_y(self,W):
        """Returns horizontal coordinate"""
        return self.sigma_y*W/2
    
    def water_depth(self,x,y):
        """
        Calculate water depth based on bed topography
        water depth is positive if the bed is beneath sea level
        and zero otherwise
        """
        bed=self.bed_topo(x,y)
        D = -bed*(bed<0.0)
        
        return D
    
    def set_initial_cond(self,hsurf, H, L, W):
        """
        Initial condition
            hsurf = surface elevation
            H = ice thickness
            L = length of the glacier from ice divide to terminus
        """
        # Check to make sure each array has the right dimensions
        #if (len(hsurf))!=self.N:
        #    print "hsurf doesn't have the right number of grid points"
        #if (len(H))!=self.N:
        #    print "H doesn't have the right number of grid points"
        
        # Initialize surface elevation and ice thickness  
        self.hsurf = hsurf
        self.H = H
        self.L = L
        self.dLdt_max = 0.0
        self.dLdt_min = 0.0
        self.dLdt_mean = 0.0
        self.W = W
        self.H_old = H
    
    # Calculate height of cliff, assuming it is at the yield strength
    def calving_cliff_height(self,D): # Can eliminate all arguments??
        """
        Calculate height of calving cliff based on yield strength tau_c,
        friction coefficient mu and water depth D.  For now, we don't include
        friction in these calculation.  If the bed is above sea level (z=0),
        the water depth is zero. 
        """
        
        g = self.g
        rho_w = self.rho_w
        rho = self.rho
        mu = self.mu
        tau_c = self.tau_c
        #D = self.water_depth(self.L)
        Hmax = 2*tau_c/(rho*g*(1-2*mu)) + sqrt((2*tau_c/(rho*g*(1-2*mu)))**2+rho_w/rho*D**2)
        Hf = rho_w/rho*D
        Hmax = maximum(Hf,Hmax)
        #Hmax = Hf
        #Hmax = 0.5*(Hf+Hmax)
        return Hmax

    
 
class PerfectPlastic(Model):
    def __init__(self, N,*args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)    
        self.N = N
        #self.sigma=linspace(0,1,N)
        n = arange(0,N)
        #self.sigma = sqrt(1-linspace(0,1,N)**2)[::-1]
        sigma = cos(pi*n/(N-1))
        self.sigma = 0.5*(sigma+1)[::-1]
        self.Hbc = None  # Inflow ice thickness at x=0
        self.Ubc = None  # Inflow velocity at x=0 in units of m/yr
        self. u = [0.0]
        self.Vc = 0.0
        
        
    def get_term_vel(self):
        return self.u[0]

    
    def steady_state_prof(self,x):
        """
        Calculate steady-state profile for a constant accumulation rate
        and constant width glacier using quadrature
        
        Input: x -- horizontal coordinates where surface elevation is desired
               hsurf_c -- surface elevation at calving front
               accum_rate -- constant accumulation rate in m/a
               
        Returns: h_surf -- steady state surface elevation
                 qc -- steady state flux at calving front
                 
        Notes:  Assumes that x[0] is the position of the calving front
        
        Usage:
        >>hsurf,qc = sia_model.steady_state_prof(x,hc,accum_rate)
        
        """
        bed_topo = self.bed_topo
        rho = self.rho
        g = self.g
        tau_c = self.C
    
        L = x[-1]
        
        Hc =self.calving_cliff_height(self.water_depth(L))
        hsurf_c = bed_topo(L)+Hc
        def _dhdx(h,x):
            H = h-bed_topo(L-x)
            dhdx = tau_c/H/rho/g
            return dhdx
        
       
        
        h = odeint(_dhdx,hsurf_c,x).ravel()
        H = h - bed_topo(L-x)
        return h[::-1],H[::-1]
        
    
        
    def balance_vel(self,x,hsurf,time):
        
        H = hsurf - self.bed_topo(x)
        a,f = self.accum(x,hsurf,time)
        Vb = trapz(a*self.width(x),x)/H/self.width(x)
        return Vb
 
 
    #def balance_vel(self,x,hsurf,time):
    #    # Compute total flux
    #    q=self.get_flux(x,hsurf,time)
    #    L = x[-1]
    #    Hc =self.calving_cliff_height(self.water_depth(L))
    #    Vb = q/self.width(x[-1])/Hc
    #    return Vb
    
    def integrate(self,dt=1.0,time_interval=100e3,advance = False):
        L = self.L
        x=self.get_sigma(L)
        bed_topo  = self.bed_topo
        n = self.n
        Delta   = (self.rho *self.g/self.B/4)**n*self.secpera # Dimensional rheology parameter in annum
    
    
    
        
        time = self.get_time()
        start_time = self.get_time()
        end_time = time + time_interval -dt
        Nt = int(round((time_interval)/(dt),0))
    
        # Calculate average value over interval
        dLdt_vals = []
        Vt_vals = []
        
        for i in xrange(Nt):
            
            
            x=self.get_sigma(L)
            hsurf,H = self.steady_state_prof(x)
            
            bed = bed_topo(x)
            accum_rate,front_melt_rate = self.accum(x,hsurf,time) # Accumulation rate
            front_melt_fun = 2*(1-(L-x)/H[-1])*self.water_depth(L)/H[-1]*front_melt_rate
            front_melt_fun = front_melt_fun*(front_melt_fun>0.0)
            accum_rate = accum_rate - front_melt_fun
            front_melt_rate = 0.0
            
            
            u = self.balance_vel(x,hsurf,time)
            Vt = u[-1]
            if advance == True:
                dx = x[-1]-x[-2]
                W  = self.width(L)
                Wx = (W -self.width(L-dx))/dx
                Hc = self.calving_cliff_height(self.water_depth(L))
                Ux = Delta*((1-self.rho_w/self.rho*self.water_depth(L)**2/Hc**2)*Hc)**(n)
                Hx = (H[-1]-H[-2])/dx
                Hcx =(Hc-self.calving_cliff_height(self.water_depth(L-dx)))/dx
                dLdt = (accum_rate[-1] - Vt*Hx - Ux*Hc - Vt*Hc*Wx/W)/(Hcx-Hx)
                L = L + dt*dLdt
                Lmax = 1.0
                L = max(L,Lmax)
                #if L<=Lmax:
                #    dLdt = 0.0
                
            x=self.get_sigma(L)
            hsurf,H = self.steady_state_prof(x)
            
            dLdt_vals.append(dLdt)
            Vt_vals.append(Vt)
        
        
        time = start_time+(i+1)*dt    
        dLdt_ave = mean(dLdt_vals)
        Vt_ave = mean(Vt)
        
        self.time.append(time)   
        self.hsurf = hsurf
        self.H = H
        self.L = L
        self.u = u
        self.Vc = u[-1]-dLdt
            
        return H,hsurf,L,dLdt_ave,Vt_ave
            
        
        
    
        
        
class ModelFE(Model):
    def __init__(self, N,a=0.0,*args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)    
        self.N = N
        self.a=a
        #self.sigma=linspace(0,1,N)
        n = arange(0,N)
        #self.sigma = sqrt(1-linspace(0,1,N)**2)[::-1]
        sigma = cos(pi*n/(N-1))
        self.sigma = 0.5*(sigma+1)[::-1]
        self.Hbc = None  # Inflow ice thickness at x=0
        self.Ubc = None  # Inflow velocity at x=0 in units of m/yr
        self. u = [0.0]
        self.Vc = 0.0
        
        
    def get_term_vel(self):
        return self.u[-1]
        
class Ssa(Model):
    
   
    def prog_solve(self,dt,H,hsurf,u,dLdt,xc,accum_rate,fixed_calving_front=True,Hc=None,eps=5e3):
        """
        Inputs:
            H - ice thickness
            hsurf - surface elevation
            u - glacier velocity
            dLdt - rate of terminus advance/retreawt
            xc - Terminus position
        Optional inputs:
            fixed_calving_front - True or False if True then fix calving front thickness to be Hc
            Hc - Fixed calving front thickness
            
        Returns Updated ice thickness
        
        Example:
        H = prog_solve(H,hsurf,u,dLdt,xc,fixed_calving_front=False,Hc=None):
        
        NOTE: Implicit time step is only semi-implicit - we use the velocity at t instead of t+1 to advect the ice
        """
        
        
        
        Dxc = self.D/xc
        N = self.N
        x=self.get_sigma(xc)
        I = eye(N+1)
         
        accum_rate,front_melt_rate = self.accum(x,hsurf,self.time[-1])
        
        # Fully implicit time step
        omega = 1.0
        if fixed_calving_front==True:
            eps = eps # We add some diffusion for stabilization if the calving front thickness is fixed
        else:
            eps =0.0

        # If the calving front thickness is not provided, we assume it is given by the yield strength    
        if Hc==None:
            Hc = self.calving_cliff_height(self.water_depth(xc)) 
        H_new = H.copy()
        
        # Set soem
        W = self.width(x)
        rhs = H - (1-omega)*dt*dot((dot(diag(W*u),Dxc) + diag(dot(Dxc,W*u))),H)/W + (1-omega)*dt*dot(dot(diag(x/xc*dLdt),Dxc),H)  + accum_rate*dt + (1-omega)*eps*dt*dot(dot(Dxc,Dxc),H)        
        A = I + omega*(dt*(dot(diag(u),Dxc) + dot(diag(1.0/W),diag(dot(Dxc,u*W)))) - dt*dot(diag(x/xc*dLdt),Dxc)) - dt*omega*eps*dot(Dxc,Dxc)
        
        #rhs = H + dt*accum_rate
        #A = I+dt*diag((dot(Dxc,u*W)/W)) + dt*dot(diag(u),Dxc)
        
        
        
        # Account for zero slope surface elevation boundary condition at x = 0
        A[-1,:] = Dxc[-1,:]; rhs[-1] = -dot(Dxc,self.bed_topo(x))[-1]
        #A[-1,:] = 0; A[-1,-1]=1;rhs[-1] = 202.10094776663891
        
        # If the calving front thickness is fixed, we apply this is a constraint to the problem
        #if fixed_calving_front==True:
        #    A[0,:]=0;A[0,0]=1;rhs[0]=Hc
            
        # Finally solve the system
        H_new = array(linalg.solve(A,rhs))
        
        return H_new
    
    
    
    def diag_solve(self,H,hsurf,xc,plastic=False,tau_b=0.0):
        """
        function solves diagnostic equation for ice sheet velocity
        Inputs : H - ice thickness
                 hsurf - surface elevation
                 plastic - optional flag if set set true solves for a plastic bed with plastic yield stress
                  tau_b
                 
        Returns velocity in units of m/a         
        """
        rho = self.rho
        rho_w = self.rho_w
        g = self.g
        n = self.n
        p = self.p
        C = self.C
        B = self.B
        
        tau_c = self.tau_c
        Dxc = self.D/xc
        L0 = 100e3
        #u = self.accum_rate.cumsum()[::-1]/H/secpera
        u = ones(size(hsurf))
        # Calculate driving stress
        rhs = rho*g*H*dot(Dxc,hsurf)
        rhs[-1] = 0.0
        
        
        bed = hsurf - H
        water_depth = -bed*(bed<0.0)
        
        itnum = 0.0
        tol =  1e-10
        err=9999.9
        delta = 1.0/self.secpera
        gamma = sqrt((dot(Dxc,u))**2+(delta/L0)**2)
        visc = 0.5*B*gamma**(1.0/n-1)
        #visc = 0.5*B*(abs(dot(Dxc,u))+1e-14)**(1.0/n-1) # Calculate viscosity based on guess of viscosity
        while err>tol:
            
            uold = u.copy()
            tau_xx=dot(diag(visc),dot(Dxc,u))
            
            # Estimate viscosity of ice based on previous estimate of velocity
            
            # Second invariant of strain rate
            gamma = sqrt((dot(Dxc,u))**2+(delta/L0)**2) # Strain rate invariant
            
            #visc = 0.5*B*gamma**(1.0/n-1)*(abs(tau_xx)<=tau_c) + 0.5*(2*(1-0.1)*tau_c/gamma + B/10*gamma**(1.0/n-1))*(abs(tau_xx)>tau_c)
            visc = 0.5*B*gamma**(1.0/n-1)
            
            # Calculate estimate of fric coefficient based on estimate of velocity
            if plastic==True:
                fric_coeff = tau_b/sqrt(u**2+delta**2)
            else:
                fric_coeff = C*(sqrt(u**2+delta**2))**(p-1) 
            

            # Create operators for matrix solve
            K = 4*dot(Dxc,dot(diag(H*visc),Dxc)) - diag(fric_coeff)
            
            # Apply fixed velocity at inflow boundary
            K[-1,:] = 0; K[-1,-1]=1.0;
            
            # Apply dynamic boundary condition at calving front
            #K[0,:] = 4*H[0]*visc[0]*Dxc[0,:];rhs[0]=0.5*(rho*g*H[0]**2-rho_w*g*water_depth(xc)**2)
            K[0,:] = Dxc[0,:];
            rhs[0] =  (rho*g*H[0]/4/B*(1-rho_w/rho*water_depth[0]**2/H[0]**2))**n
            
            #rhs[0] =  (rho*g*H[0]/4/B*(1-rho/rho_w))**n
            
            # Solve for velocity
            u = array(linalg.solve(K,rhs))
            
            # Calculate difference between new and old velocity
            err = max(abs(u-uold))
            
            # Break out of iteration if number of iterations is too large
            itnum = itnum +1
            if itnum>1000:
                disp('Diagnostic solve failed to converge')
                break
            
            
        return u*self.secpera
    
    
    def integrate(self,h_initial,xc,dt=1.0,time_interval=100e3,advance = False,advance_factor = 0.0,eps=5e3):
        # First check to make sure we are appropriately initialized
        flag = self.check_initialization()
        if flag!=0:
            return None
    
        # Get grid parameters
        D = self.D.copy()
        sigma = self.sigma
        
        # Create physical coordinate
        x = sigma*xc
        
        # Some short hand notation to avoid having to type self.???
        water_depth = self.water_depth
        bed_topo = self.bed_topo
        tau_c = self.tau_c
        mu = self.mu
        p = self.p
        n = self.n
        N = self.N
        rho = self.rho
        rho_w = self.rho_w
        
        
        # Initial surface elevation and ice thickness
        #H = H_initial.copy()
        #hsurf = bed_topo(x) + H
        H_initial = h_initial-bed_topo(x)
        hsurf = h_initial.copy()
        H = H_initial.copy()
        
        
        time = self.get_time()
        start_time = self.get_time()
        end_time = time + time_interval -dt
        
        Nt = int(round((time_interval)/(dt),1))
        
        dLdt_vals = []
        Vt_vals = []
        for i in xrange(Nt):
            
            # Calculate accumulation rate (m/a)
            accum_rate,front_melt_rate = self.accum(x,hsurf,self.time[-1])
            
            # Scale derivative operators to length of glaciers
            Dxc = D/xc
            
            # Calculate width
            W = self.width(x)
            
            u = self.diag_solve(H,hsurf,xc)
            #u = 10*ones(size(x))
            
            # Define terminus velocity
            Vt = u[0]
          
            if advance == False:
                dLdt = 0.0
            else:
                dLdt = Vt*advance_factor
                
            dLdt = 0.0
            # Calving front yield strength
            Hc = self.calving_cliff_height(water_depth(xc+dLdt*dt))
            
            # Floatation thickness
            Hf = rho_w/rho*water_depth(xc+dLdt*dt)
            H_new = self.prog_solve(dt,H,hsurf,u,dLdt,xc,fixed_calving_front=False)
            #H= H_new
            # Take a tentative step forward and then check to see if the thickness exceeds the yield thickness or is less than floatation
            #if advance==True:
            #    H_new = self.prog_solve(dt,H,hsurf,u,dLdt,xc,fixed_calving_front=False)
            #else:
            #    H_new = self.prog_solve(dt,H,hsurf,u,dLdt,xc,fixed_calving_front=False)
                #u = self.diag_solve(H_new,H_new + self.bed_topo(x),xc)
                #u = 0.5*(u+unew)
                #H_new = self.prog_solve(dt,H,hsurf,u,dLdt,xc,fixed_calving_front=False)
                #u = self.diag_solve(H_new,H_new + self.bed_topo(x),xc)
                #H_new = self.prog_solve(dt,H,hsurf,u,dLdt,xc,fixed_calving_front=False)
                #u = self.diag_solve(H_new,H_new + self.bed_topo(x),xc)
                #H_new = self.prog_solve(dt,H,hsurf,u,dLdt,xc,fixed_calving_front=False)
            """    
            # If we exceed the yield strength, then solve for the dLdt  
            if (H_new[0]>=Hc) & (advance==True):
                try:
                    dLdt = brentq(is_yielded, a=-u[0], b=u[0], args=(H,hsurf,xc,u), xtol=0.1, rtol=4.4408920985006262e-16, maxiter=100, full_output=False, disp=False)
                except:
                    dLdt=fsolve(is_yielded,0.0,args=(H,hsurf,xc,u),xtol=0.01)
                Hc = self.calving_cliff_height(water_depth(xc+dLdt*dt),tau_c,mu,self.rho,self.rho_w)
                H_new = self.prog_solve(dt,H,hsurf,u,dLdt*0,xc,fixed_calving_front=False)
            
            
            # If we are not above the yield strength, but are less than flotation then we find the dLdt that allows us to remain at flotation
            elif (H_new[0]<Hf) & (Hf>0) & (advance==True):
                try:
                    dLdt = brentq(is_floating, a=-u[0], b=u[0], args=(H,hsurf,xc,u), xtol=0.1, rtol=4.4408920985006262e-16, maxiter=100, full_output=False, disp=False)
                except:
                    dLdt=fsolve(is_floating,0.0,args=(H,hsurf,xc,u),xtol=0.01)
                
                H_new = self.prog_solve(dt,H,hsurf,u,dLdt,xc,fixed_calving_front=False)
            """
            # Update ice thickness   
            H=H_new
            
            # Make sure ice thickness is never less than zero (should never happen)
            #H[H<0.1]=0.1
            
            
            # Update terminus position
            xc = xc + dLdt*dt
            x = sigma*xc
            
            # Calculate surface elevation assuming no flotation
            hsurf = H + maximum(self.bed_topo(x),self.rho/self.rho_w*H)
            
            #bed = hsurf-H
            #water_depth = -bed*(bed<0.0)
        
            # Make sure glacier is floating
            #filter = (self.rho/self.rho_w*H<water_depth)
            #hsurf = hsurf*(1-filter) + (1-self.rho/self.rho_w)*H*filter
            
            
            # Update time and number of time steps
            time = start_time+(i+1)*dt
        
        
        self.time.append(time)
          
            
        return hsurf, xc, dLdt,u[0]
    
class Sia2D(Model2D):
    def integrate(self,dt=1.0,time_interval=100e3,advance = False,omega=2.0,creep=False,boundary='no-slip'):
        
        # Convenient short hand for some variables
        water_depth = self.water_depth
        bed_topo = self.bed_topo
        width = self.width
        tau_c = self.tau_c
        mu = self.mu
        p = self.p
        n = self.n
        Nx = self.Nx
        Ny = self.Ny
        xc = self.L
        #W = self.W
        hsurf = self.hsurf
        
        # Get grid parameters
        Dx = self.Dx.copy()
        Dy = self.Dy.copy()
        sigma_x = self.sigma_x.copy()
        sigma_y = self.sigma_y.copy()
        [S1,S2]=meshgrid(sigma_x,sigma_y)
        Ix = eye(self.Nx+1)
        Iy = eye(self.Ny+1)
        Ds1 = kron(Dy,Ix)
        self.Ds1 = Ds1
        self.Ix = Ix
        self.Iy = Iy
        
        # Create physical coordinates
        x = sigma_x*xc
        X = S1*xc
        W = self.width(x)
        
        # Create mesh in physical coordinates
        #[X,Y]=meshgrid(x,y)
        Y = self.width(X)*S2/2.0
        
        
        bed = bed_topo(X,Y)
        
        # Initial surface elevation and ice thickness
        H = hsurf - bed_topo(X,Y)
        hsurf=maximum(H+self.bed_topo(X,Y),(1-self.rho/self.rho_w)*H) # Possible problem because 
        H = hsurf - self.bed_topo(X,Y)
        H_old = self.H_old
        
        # Define some dimensional constants and convert time unit to annum
        Gamma   = (self.rho*self.g/self.C)**(1.0/p)*self.secpera # Dimensional constant in annum
        Beta    = 2/(self.n+2)*(self.rho*self.g/self.B)**self.n*self.secpera
        Beta_failed    = 2/(self.n+2)*(self.rho*self.g/(self.B/4.0))**self.n*self.secpera
        Delta   = (self.rho *self.g/self.B/4)**n*self.secpera # Dimensional rheology paratmer in annum
        Ux_max  = (self.tau_c/self.B)**self.n*self.secpera
        # Normalize yield strength by rho*g to convert into units of thickness (m)
        Tau_b = self.tau_b/(self.rho*self.g)
        Tau_c = self.tau_c/(self.rho*self.g) 
        
        time = self.get_time()
        start_time = self.get_time()
        end_time = time + time_interval -dt
        
        Nt = int(round((time_interval)/(dt),0))
        
        dLdt_max = []
        dLdt_min = []
        dLdt_max_vals = []
        dLdt_min_vals = []
        dLdt_mean_vals = []
        dLdt_vals = []
        Vt_vals = []
        for i in xrange(Nt):
            
            W = self.width(x)
            
            # Scale differential operator to scale of the grid     
            Dxc = Dx/xc
            #self.Dxc = Dxc
            
            # And y-direction requires width
            #Dyc = dot(diag(2/W),Dy)
            Dyc = Dy
            
            # Gradient of width
            dWdx = dot(Dxc,W)
            
            # Extra term associated with variable width of flowline
            #Dxi = dot(diag(2*y/W**2*dWdx),Dxc)
            
            # And mesh in transformed coordinates
            [S1,S2]=meshgrid(sigma_x,sigma_y)
            #Y = tile(W/2,(Ny+1,1))*S2
            Y = self.width(X)*S2/2.0
            
            
            # Tensor product grid
            Ix = eye(self.Nx+1)
            Iy = eye(self.Ny+1)
            Dx1 = kron(Iy,Dxc)
            #Dy1 = kron(2*Dyc/self.W,Ix)
            Dy1 = kron(Dy,diag(2/W))
            self.Dy1 = Dy1
            Dxi1 = dot(diag(Y.ravel()),kron(Iy,dot(diag(dWdx/W),Dxc)))
            self.Dxi1 = Dxi1
            self.Dx1 =Dx1
            #Dy1 = Dy1 + Dxi1
            W = width(X).flatten()
            dWdx = dot(Dx1,W)
            Dx2 = dot(diag(-2*Y.ravel()*dWdx/W**2),Ds1)
            Dx1 = Dx1 + Dx2
            self.Dx2 = Dx2
            self.Dx1 = Dx1
            self.Dy1 = Dy1
            #Dxi1 = dot(diag(Y.ravel()*dWdx/W**2),kron(2*Iy,Dxc))
            
            # Create physical coordinate
            #x = sigma_x*xc 
            #y = sigma_y*W
            #[X,Y]=meshgrid(x,y)
            
            
            # Get water depth
            accum_rate,front_melt_rate = self.accum(X,hsurf,time) # Accumulation rate
            
            #hsurf.ravel()
            #H.ravel()
            ### Change for mean versus max 
            # Mean thickness of calving front
            #Hm = mean(H,axis=0)
            Hm =-trapz(H,self.sigma_y,axis=0)/2
            #Hm =sqrt(trapz(-H**2,self.sigma_y,axis=0)/2)

            #Hm = sum(H[:,0]*self.quad_weights)/2
            #Dm = mean(self.water_depth(X,Y),axis=0)
            D  = self.water_depth(X,Y)
            Dm =sqrt(-trapz(D**2,self.sigma_y,axis=0)/2)
            
            #Dm = sum(self.water_depth(X[:,0],Y[:,0])*self.quad_weights)/2

            Hc = self.calving_cliff_height(Dm) # Calving front thickness
            Dm =-trapz(D,self.sigma_y,axis=0)/2
            #Hc = -trapz(self.calving_cliff_height(D),sigma_y,axis=0)/2
            Hcm = Hc
            #print Hc0[0],Hcm[0]

            ### Calculate melt rate function
            front_melt_fun = 2*(1-(xc-x)/Hm)*Dm[0]/Hm[0]*front_melt_rate
            front_melt_fun = front_melt_fun*(front_melt_fun>0.0)
            front_melt_fun = front_melt_rate*Dm[0]/Hm[0]*(X>=(xc-Hm[0]))
            
            
            #front_water_ratio=tile(D[:,0]/H[:,0],(self.Nx+1,1)).transpose()
            #front_melt_fun = 2*(1-(xc-X)/Hm)*front_water_ratio*front_melt_rate
            #front_melt_fun = front_melt_fun*(front_melt_fun>0.0)
            #front_melt_fun = front_melt_rate*front_water_ratio*(X>=(xc-Hm[0]))
           
            accum_rate = accum_rate - front_melt_fun
            # Set absolute version to zero just in case we accidentally try to use it later
            front_melt_rate = 0.0
            
            # Average accumulation rate across the front
            am = dot(self.quad_weights,accum_rate[:,0])/2 
            
            # Stretch accumulation rate into a long vector
            acc_term = accum_rate[:,0]
            accum_rate = accum_rate.ravel()
            
            # Need to find rate of terminus advance/retreat for all grid cells near front
            # Then take max, min or mean
            #hsurf_c = Hc + bed_topo(xc,0.0) # Surface elevation at calving front
            
            #W=self.width(x)
            #Hx = dot(Dxc,H)[0]
            
            #Neff = 1.0 + 2*H*self.B/(W*self.C)*(5/W)**(self.p)
            #u = -Gamma*(H*abs(hx)-Tau_b)**(1.0/p)*hx/(abs(hx)+1e-16)/Neff**(1.0/p)
            hx =  dot(Dx1,hsurf.ravel())
            hy =  dot(Dy1,hsurf.ravel()) #+  dot(Dxi1,hsurf.ravel())
            
            #print 'max hx',abs(hx).max(),'max hy',abs(hy).max()
            
            #diffusivity = diffusivity.ravel()
            
            
            hx = reshape(hx,(Ny+1,Nx+1))
            hy = reshape(hy,(Ny+1,Nx+1))
            #print hy[-1,:]  - hx[-1,:]*dWdx
            #hy[-1,:] = -hx[-1,:]*dWdx
            #if boundary=='stick':
            #    hx[0,:]=0.0;hy[0,:]=0.0
            #    hx[-1,:]=0.0;hy[-1,:]=0.0
            """
            if boundary=='slip':
                hy[0,:]  = hx[0,:]*dWdx
                hy[-1,:] = -hx[-1,:]*dWdx
            """
            # Diffusivity associated with sliding   
            diffusivity = Gamma*H**(1.0/p+1.0)*sqrt(hx**2+hy**2)**(1.0/p-1.0)
            diffusivity = diffusivity.ravel()
            
            ### Change for mean versus max 
            #Vt = -mean(Gamma*(H[:,0]*sqrt(hx[:,0]**2+hy[:,0]**2))**(1.0/p)*hx[:,0])
            ux = -Gamma*H**(1.0/p)*sqrt(hx**2+hy**2)**(1.0/p-1.0)*hx
            uy = -Gamma*H**(1.0/p)*sqrt(hx**2+hy**2)**(1.0/p-1.0)*hy
            
            if creep==True:
                temp = -Beta*H**(n+1)*sqrt(hx**2+hy**2)**(n-1)
                ux_creep =  temp*hx
                uy_creep =  temp*hy
                ux = ux + ux_creep
                uy = uy + uy_creep
                diffusivity_flow = (Beta*H**(n+2)*sqrt(hx**2+hy**2)**(n-1)).ravel()
                diffusivity = diffusivity + diffusivity_flow
            
            
            
            self.ux = ux
            self.uy = uy
            Vt = sqrt(ux[:,0]**2+uy[:,0]**2)
            #Vt = ux[:,0]
            # Entire terminus advances or retreats
            dLdt_max = 0.0
            dLdt_min = 0.0
            dLdt_mean = 0.0
            
            
            #D  = self.water_depth(X,Y)
            #Dm =-trapz(D,self.sigma_y,axis=0)/2
            #Hcc = self.calving_cliff_height(Dm)
            #Hcc = tile(Hcc,(Ny+1,1))
            
            Hcc = self.calving_cliff_height(D)
            # Testing consistent calving law
            #Hc = (self.calving_cliff_height(self.water_depth((xc+dLdt*dt)*ones(shape(Y[:,0])),Y[:,0])))
            #hsurf_new = bed_topo(X,Y) + Hcc
            #hsurf_max = -trapz(hsurf_new,sigma_y,axis=0)/2
            #hsurf_max = maximum(hsurf_max,(1-self.rho/self.rho_w)*H[:,0])
            #hsurf_max = maximum(tile(hsurf_max,(Ny+1,1)),rho_w
            
            
            hsurf_c = Hcc + bed_topo(X,Y)
            idx = argmax(hsurf_c,axis=0)
            
            #self.idx = idx
            #hsurf_c_max = hsurf_c[idx,:]
            idx2 = arange(0,Nx+1)
            hsurf_c_max = hsurf_c[idx,idx2]
            #Hmax = H[idx,idx2]
            #Dmax = D[idx,idx2]
            #Hcmax = Hcc[idx,idx2]
            #hsurf_c_max = amax(hsurf_c,axis=0)
            #hsurf_c_max = tile(hsurf_c_max,(Ny+1,1))

            #Hcc = (hsurf_c_max-bed_topo(X,Y))
            self.dLdt = 0.0
            if advance==True:
                # We start by calculating advance and retreat associated with average thickness and water depth
                #     We can also try to compute the rate of advance and retreat for each node along the terminus
            
                # Compute max and min and mean based on local points
                Dt = self.water_depth(X,Y)
                Ht = maximum(H[:,0],self.rho_w/self.rho*Dt[:,0])
                Ht = H[:,0]
                Hcc = self.calving_cliff_height(Dt)
                Hcc = maximum(Hcc,self.rho_w/self.rho*Dt)
                Ux = Delta*((1-self.rho_w/self.rho*Dt[:,0]**2/Ht**2)*Ht)**(n)
                Hx = reshape(dot(Dx1,H.ravel()),(Ny+1,Nx+1))[:,0]
                Hcx = reshape(dot(Dx1,Hcc.ravel()),(Ny+1,Nx+1))[:,0]
                dLdt = (acc_term -Vt*Hx - Ux*Hcc[:,0])/(Hcx-Hx)
                #dLdt = minimum(dLdt,Vt)
                dLdt_max = max(dLdt)
                dLdt_min = min(dLdt)
                dLdt_mean = sum(dLdt*self.quad_weights)/2
                #kernel = exp(-sigma_y[1:-1]**2)
                #kernel = kernel/sum(kernel)
                self.dLdt = dLdt
                
                #dLdt = mean(dLdt[Ny/2-1:Ny/2+2])
                #dLdt = dLdt[idx[0]]
                #dLdt =  sum(dLdt*self.quad_weights)/2
                #dLdt = dLdt[Ny/2]
                #dLdt =  sum(dLdt[1:-1]*kernel)
                #self.dLdt = dLdt
                #print Ux
                #print Ht
                #print Hx
                #print Hcx
                #f=abs(Y[:,0])<100e3
                dLdt = maximum(dLdt,-20e3)
                dLdt = median(dLdt)
                #print i,dLdt
                #Hterm = H[:,0] + dt*(accum_rate[:,0]-Ux*H[:,0]-Vt*Hx+dLdt*Hx)
                
                #Hterm = H[:,0] + accum_
                #print '********'
                #print dLdt
                #print idx[0],dLdt[idx[0]]
                #print '********'
                #dLdt = dLdt[idx[0]]
                
                # Compute average based on width averaged properties
                # Ht = Hm[0]
                # Dt = Dm[0]
                # 
                # Ux = Delta*((1-self.rho_w/self.rho*Dt**2/Ht**2)*Ht)**(n)
                # Hx = dot(Dxc,Hm)[0]
                # Hcx = dot(Dxc,Hc)[0]
                # v = dot(self.quad_weights,Vt)/2
                # #dLdt = (am -v*Hx - Ux*Ht)/(Hcx-Hx)
                # #print 'Terminus thick',Ht,'Water depth',Dt,'Calving front thick',Hc[0]
                # #Dtt =self.water_depth((xc)*ones(shape(Y[:,0])),Y[:,0])
                # Hm2 =sqrt(-trapz(H**2,self.sigma_y,axis=0)/2)[0]
                # Dm2 =sqrt(-trapz(D**2,self.sigma_y,axis=0)/2)[0]
                # Ux = Delta*((1-self.rho_w/self.rho*Dm2**2/Hm2**2)*Hm[0])**(n)
                # dLdt = (am -v*Hx - Ux*Hm[0])/(Hcx-Hx)
                # self.dLdt = dLdt
                
                
                
                dLdt = maximum(dLdt,-20e3)
                dLdt = minimum(dLdt, max(Vt))
            else:
                dLdt = 0.0
            
            
            dLdt_vals.append(dLdt)
            dLdt_mean_vals.append(dLdt_mean)
            dLdt_max_vals.append(dLdt_max)
            dLdt_min_vals.append(dLdt_min)
            Vt_vals.append(sum(Vt*self.quad_weights)/2)
            # Calculate non-linear diffusivity
            #W = width(x)
            
            
    
            # Create differential operator
            #diffusivity=diffusivity.ravel()
            L1 = dot(Dx1,dot(diag(diffusivity),Dx1)) + dot(Dy1,dot(diag(diffusivity),Dy1)) #+ dot(Dy1,dot(diag(diffusivity),Dxi1)) + dot(Dxi1,dot(diag(diffusivity),Dy1)) + dot(Dxi1,dot(diag(diffusivity),Dxi1)) #dot(Dy1,Dxi1) + dot(Dxi1,Dxi1)
            L2 = dot(diag(S1.ravel()*dLdt),Dx1)
            #L2 = 0.0
            K =  eye((Nx+1)*(Ny+1)) - omega*(L1+L2)*dt
            rhs = H.ravel() + (accum_rate)*dt + (1-omega)*dot(L1+L2,H.ravel())*dt + dt*dot(L1,bed_topo(X,Y).ravel())
            
            
            #Hc = self.calving_cliff_height(self.water_depth(xc+dLdt*dt,0.0))
            
  
            
            
            
            """
            if boundary == 'slip':
                W = self.width(X).ravel()
                dWdx = dot(Dx1,W/2)
                DD=dot(diag(dWdx),Dx1)
                # Set slope at left and right walls to zero    
                for j in xrange(0,Nx+1):
                    # Account for zero slope along y=W
                    idx1 = j
                    
                    # dh/dy = 0.0
                    #K[idx,:] = (Dy1[idx,:] - Dxi1[idx,:])
                    #dWdx * hx - hy = 0
                    K[idx1,:] = -(Dy1[idx1,:] - Dxi1[idx1,:]) + DD[idx1,:]
                    rhs[idx1] = 0.0#dot(Dy1,bed_topo(X,Y).ravel())[idx]-dot(Dxi1,bed_topo(X,Y).ravel())[idx]
                    #ww1=self.width(X.ravel()[idx1])/1e3/2
                    #print idx1,X.ravel()[idx1]/1e3,Y.ravel()[idx1]/1e3,ww1,ux.ravel()[idx1],uy.ravel()[idx1],-hx.ravel()[idx1]*dWdx[idx1]+hy.ravel()[idx1]

                    
                    
                    #print idx,X.ravel()[idx]/1e3,Y.ravel()[idx]/1e3
    
                    # Account for zero slope along y=W
                    idx = (Nx+1)*(Ny+1)-1-j*1
                    # -dWdx * hx + hy = 0.0
                    K[idx,:] = (Dy1[idx,:] - Dxi1[idx,:]) + DD[idx,:]
                    # hx = 0.0
                    #K[idx,:] = Dy1[idx,:]- Dxi1[idx,:]
                    rhs[idx] = 0.0#-dot(Dy1,bed_topo(X,Y).ravel())[idx]-dot(Dxi1,bed_topo(X,Y).ravel())[idx]
                    #print idx,X.ravel()[idx]/1e3,Y.ravel()[idx]/1e3,ux.ravel()[idx],uy.ravel()[idx],-hx.ravel()[idx]*dWdx[idx]+hy.ravel()[idx]
                    ww=self.width(X.ravel()[idx])/1e3/2
                    #print X.ravel()[idx]/1e3,Y.ravel()[idx]/1e3,ww,ux.ravel()[idx],uy.ravel()[idx],hx.ravel()[idx]*dWdx[idx]+hy.ravel()[idx]
                    #print -hx.ravel()[idx1]*dWdx[idx1]+hy.ravel()[idx1],-hx.ravel()[idx]*dWdx[idx]+hy.ravel()[idx]
                    #print ux.ravel()[idx1],uy.ravel()[idx1],ux.ravel()[idx],uy.ravel()[idx],-hx.ravel()[idx1]*dWdx[idx1]+hy.ravel()[idx1],-hx.ravel()[idx]*dWdx[idx]+hy.ravel()[idx]
            #print '***********'
            """
            W = self.width(X).ravel()
            dWdx = dot(Dx1-Dx2,W/2)
            DD=dot(diag(dWdx),Dx1)
            
            bed_y=dot(Dy1,bed_topo(X,Y).ravel())
            Dr = dot(DD,bed_topo(X,Y).ravel())
            for j in xrange(0,Nx+1):
                    # Account for zero slope along y=W
                    idx1 = j
                    idx2 = (Nx+1)*(Ny+1)-1-j*1
                    K[idx1,:] =  Dy1[idx1,:]  - DD[idx1]
                    rhs[idx1] =  -bed_y[idx1]  + Dr[idx1]
                    K[idx2,:] =  Dy1[idx2,:]  + DD[idx2]
                    rhs[idx2] =  -bed_y[idx2]  - Dr[idx2]
                    #print -hx.ravel()[idx1]*dWdx[idx1]+hy.ravel()[idx1], -hx.ravel()[idx2]*dWdx[idx2]+hy.ravel()[idx2]
                    #print  hy.ravel()[idx1]-hx.ravel()[idx1]*dWdx[idx1]
            
            
            # Account for ice thickness boundary condition at calving front and zero slope BC at ice divide
            bed_x=dot(Dx1,bed_topo(X,Y).ravel())
            #print shape(xc),shape(dLdt),shape(Y[:,0])
            #D=self.water_depth(xc+dLdt*dt,Y[:,0])
            Hcm = Hc[0]
            D =self.water_depth((xc+dLdt*dt)*ones(shape(Y[:,0])),Y[:,0])
            Hc = (self.calving_cliff_height(self.water_depth((xc+dLdt*dt)*ones(shape(Y[:,0])),Y[:,0])))
            #D  = self.water_depth(S1*(xc+dLdt*dt),Y)
            #Dm =-trapz(D,self.sigma_y,axis=0)/2
            #Hc = self.calving_cliff_height(Dm)[0] 
            hsurf_new = bed_topo((xc+dLdt*dt)*ones(shape(Y[:,0])),Y[:,0]) + Hc
            hsurf_max = max(hsurf_new)
            #hsurf_max = median(hsurf_new)
            #hsurf_max = trapz(hsurf_new,sigma_y)/2.0
            #hsurf_max = maximum(dot(hsurf_new,self.quad_weights)/2.0,D*(self.rho_w/self.rho-1))
            #hsurf_max = maximum(max(hsurf_new),D*(self.rho_w/self.rho-1))
            #print hsurf_new-hsurf_max
            #print hsurf_new
            bed =  bed_topo((xc+dLdt*dt)*ones(shape(Y[:,0])),Y[:,0])
            Hc_max = hsurf_max - bed

            #print Hcm,mean(Hc),max(Hc),max(Hc_max),mean(Hc_max)

            #print Hc_max
            #Hc_max = hsurf_max 
            #Hc = min(Hc)
            
            for j in xrange(Ny+1):
                # Account for fixed ice thickness boundary condition at x=L
                idx = (Nx+1)*j
                
                Hc = self.calving_cliff_height(self.water_depth(xc+dLdt*dt,Y.ravel()[idx]))
                Hc = max(Hc,self.water_depth(xc+dLdt*dt,Y.ravel()[idx])*self.rho_w/self.rho)
                #yy = Y.ravel()[idx]
                #ll = xc+dLdt*dt
                #Hc = self.calving_cliff_height(self.water_depth(ll,yy))
                #print ll,yy,shape(Hc)
                K[idx,:] = 0; K[idx,idx]=1.0; rhs[idx] = Hc#_max[j]
                
                # Account for zero slope boundary condition at x = 0
                idx = -1-(Nx+1)*j
                K[idx,:] = Dx1[idx,:]
                #rhs[idx] = -dot(Dx1,bed_topo(X,Y).ravel())[idx]
                rhs[idx] = -bed_x[idx]

            
            
            
            
            # Solve the dang thing
            self.H_old = H
            H_old = H
            K = sparse.csr_matrix(K)
            H = reshape(array(spsolve(K,rhs)),(Ny+1,Nx+1))
            #H = reshape(array(linalg.solve(K,rhs)),(Ny+1,Nx+1))
            #H = reshape(array(linalg.solve(K,rhs)),(Ny+1,Nx+1))
            
            # Adjust calving front position and calculate new horizontal positions
            xc = xc + dt*dLdt
            
            
            # Create physical coordinates
            x = sigma_x*xc 
            X = S1*xc
            #Y = tile(W/2,(Ny+1,1))*S2
            Y = self.width(X)*S2/2.0
            #x = sigma_x*xc 
            #y = sigma_y*(mean(W)/2)
        
            # Create mesh in physical coordinates
            #[X,Y]=meshgrid(x,y)
            
            
            # Make sure ice thickness is positive definite -- thickness will only
            # be negative if something goes horribly wrong with the time stepping
            #filter = H<10.0
            #H[filter]=10.0
            
            
            # Update bed topography
            #bed_topo.update(x,0.5*(H+H_old),dt)
            #Hin = 0.5*sum(dot(self.quad,(H+H_old)),axis=0)
            #print Hin
            #Q=tile(self.quad_weights,(len(x),1)).transpose()/2
            #Hin = 0.5*(H[4,:]+H_old[4,:])
            #Hin = 0.5*sum(Q*(H+H_old),axis=0)
            #Hin = -trapz(H+H_old,x,axis=0)
            #Hin =-trapz(0.5*(H+H_old),self.sigma_y,axis=0)/2
            
            Hin = 0.5*(H+H_old)
            bed_topo.update(X,Y,Hin,dt)

            
            # Make sure ice never thins beneath flotation
            hsurf = H + self.bed_topo(X,Y)
            #hsurf=maximum(H+self.bed_topo(X,Y),(1-self.rho/self.rho_w)*H) 
            #H = hsurf - self.bed_topo(X,Y)
            
            
           
            
            
            # Update time and number of time steps
            time = start_time+(i+1)*dt
            
            #print time,start_time+(i+1)*dt,i,Nt

        Vt_ave = mean(Vt_vals)
        dLdt_ave = mean(dLdt_vals)
        self.dLdt_max =mean(dLdt_max_vals)
        self.dLdt_min = mean(dLdt_min_vals)
        self.dLdt_mean = mean(dLdt_mean_vals)
        self.time.append(time)   
        self.hsurf = hsurf
        self.H = H
        self.L = xc
        
        return H,hsurf,xc,dLdt_ave,Vt_ave
        
        return
        
    
class Sia(Model):
    
    def get_term_vel(self):
        return self.u[0]
    
    def steady_state_prof(self,x,hsurf_c,accum_rate):
        """
        Calculate steady-state profile for a constant accumulation rate
        and constant width glacier using quadrature
        
        Input: x -- horizontal coordinates where surface elevation is desired
               hsurf_c -- surface elevation at calving front
               accum_rate -- constant accumulation rate in m/a
               
        Returns: h_surf -- steady state surface elevation
                 qc -- steady state flux at calving front
                 
        Notes:  Assumes that x[0] is the position of the calving front
        
        Usage:
        >>hsurf,qc = sia_model.steady_state_prof(x,hc,accum_rate)
        
        """
        bed_topo = self.bed_topo
        p = self.p
        Gamma = self.C/(self.rho*self.g)
        Delta = (self.rho*self.g/(4*self.B))
        accum_rate = accum_rate/self.secpera
        n = self.n
        def _dhdx(h,x):
            H = h-bed_topo(x)
            W = self.width(x)
            #q = accum_rate*x
            q = self.get_flux(x,1600,0.0)/self.secpera
            dhdx = -Gamma*abs(q)**(p-1)*q/H**(p+1)/W**(p)
            #print x,h,q,dhdx,H
            return dhdx
        
       
            
        h = odeint(_dhdx,hsurf_c,x).ravel()
        H = h - bed_topo(x)
        D = self.water_depth(x[0])
        Hc = H[0]
        Hx = dot(self.D/x[0],H)[0]
        qc = -(Hc**(n+1)*(Delta*(1-self.rho_w/self.rho*D**2/Hc**2))**n -accum_rate)*Hc/Hx
        return h,qc*self.secpera
    
    
            
    
    def integrate(self,dt=1.0,time_interval=100e3,advance = False,omega=2.0,creep=False):
        """
        Time step diffusion equation
        Here we solve the equation H_t = (D*h_x)_x + a where D is diffusivity and a is accum rate
        
        
        Input:
        hsurf -- initial condition for ice sheet surface elevation
        xc -- initial calving front position
        dt -- time step (years)
        time_interval -- length of time of numerical integration (years)
        advance -- Boolean flag True means the terminus will advance/retreat and False means it will stay fixed
        omega -- Numerical parameter for time integration setting it higher will result in more stable but less accurate solutions
        
        Returns:
        hsurf -- updated surface elevation
        xc -- location of the terminus
        dLdt -- rate of terminus advance/retreat (negative for retreat)
        
        Usage:
        hsurf,L,dLdt=sia.integrate(dt=dt,time_interval=time_interval,advance=True)
        """
        # First check to make sure we are appropriately initialized
        flag = self.check_initialization()
        if flag!=0:
            return None
        
        
        xc = self.L
        hsurf = self.hsurf
        
        # Get grid parameters
        D = self.D.copy()
        sigma = self.sigma
        
        # Create physical coordinate
        x = sigma*xc 
        
        # Convenient short hand for some variables
        water_depth = self.water_depth
        bed_topo = self.bed_topo
        width = self.width
        tau_c = self.tau_c
        mu = self.mu
        p = self.p
        n = self.n
        N = self.N
        
        bed = bed_topo(x)
        
        # Initial surface elevation and ice thickness
        H = hsurf - bed_topo(x)
        H_old = self.H_old

        # Define some dimensional constants and convert time unit to annum
        Gamma   = (self.rho*self.g/self.C)**(1.0/p)*self.secpera # Dimensional constant in annum
        Beta    = 2/(self.n+2)*(self.rho*self.g/self.B)**self.n*self.secpera
        Delta   = (self.rho *self.g/self.B/4)**n*self.secpera # Dimensional rheology paratmer in annum
        Ux_max  = (self.tau_c/self.B)**self.n*self.secpera
        # Normalize yield strength by rho*g to convert into units of thickness (m)
        Tau_b = self.tau_b/(self.rho*self.g)
        Tau_c = self.tau_c/(self.rho*self.g) 
        
        time = self.get_time()
        start_time = self.get_time()
        end_time = time + time_interval -dt
        
        Nt = int(round((time_interval)/(dt),0))
        dLdt_vals = []
        Vt_vals = []
        for i in xrange(Nt):
            
            
            
            # Scale differential operator to scale of the grid     
            Dxc = D/xc
            
            # Gradient in surface elevation, ice thickness and calving front yield thickness
            hx = dot(Dxc,hsurf)
            
            
            # Get water depth
            accum_rate,front_melt_rate = self.accum(x,hsurf,time) # Accumulation rate
            front_melt_fun = 2*(1-(xc-x)/H[0])*self.water_depth(xc)/H[0]*front_melt_rate
            front_melt_fun = front_melt_fun*(front_melt_fun>0.0)
            #front_melt_fun = front_melt_rate*self.water_depth(xc)/H[0]*(x>=(xc-H[0]))
            accum_rate = accum_rate - front_melt_fun
            front_melt_rate = 0.0
            
            Hc = self.calving_cliff_height(self.water_depth(xc)) # Calving front thickness
            hsurf_c = Hc + bed_topo(xc) # Surface elevation at calving front
            #factor = 0.0
            W=self.width(x)
            Hx = dot(Dxc,H)[0]
            #Pi = self.rho*self.g*H
            #Pw = minimum(self.rho_w*self.g*water_depth(x),Pi)
            #f = 1.0 - Pw/Pi*exp(-((self.L-x)/(10*H[0])))
            #Neff = (1-0.8*Pw/Pi) + 2*H*self.B/(W*self.C)*(5/W)**(self.p)
             
            
            #Neff = 1.0 + 3.5/2.0*0.5*(tanh((bed_topo(x)-200.0)/200.0)+1) + 2*H*self.B/(W*self.C)*(5/W)**(self.p)
            Neff = 1.0 + 2*H*self.B/(W*self.C)*(5/W)**(self.p)
            #Neff = 1.0 + 1.5/2.0*0.5*(tanh((bed_topo(x)-200.0)/200.0)+1) + 2*H*self.B/(W*self.C)*(5/W)**(self.p)
            #Neff = (2.0 + 3.5*0.5*(tanh((bed_topo(x)-200.0)/200.0)+1))*f + 2*H*self.B/(W*self.C)*(5/W)**(self.p)
            
            #Neff = (4.0 + 3.5*0.5*(tanh((bed_topo(x)-200.0)/200.0)+1))*(1-Pw/Pi)+ 2*H*self.B/(W*self.C)*(5/W)**(self.p)
            
            #Neff = 2.0 + 2*H*self.B/(W*self.C)*(5/W)**(self.p) + 0*2.5*0.5*(tanh((bed_topo(x)-200.0)/200.0)+1)+ 2*H*self.B/(W*self.C)*(5/W)**(self.p)
            #Neff = (2.0+3.5) + 2*H*self.B/(W*self.C)*(5/W)**(self.p)  - 0.5*(tanh((bed_topo(x)-1000.0)/1000.0)+1)
            #Neff = 5.0 + 2*H*self.B/(W*self.C)*(5/W)**(self.p)
            #Neff = 1.5
            #Neff = 1.0
            #u = -Gamma*(H*abs(hx)-Tau_b)**(1.0/p)*hx/(abs(hx)+1e-16)/Neff**(1.0/p)
            
            #diffusivity = W*Gamma*H/sqrt(hx**2+1e-16)*(H*abs(hx)-Tau_b)**(1.0/p)/Neff**(1.0/p)
            diffusivity = W*Gamma*H**(1.0/p+1.0)*sqrt(hx**2)**(1.0/p-1.0)/Neff**(1.0/p)
            u=-Gamma*H**(1.0/p)*abs(hx)**(1.0/p-1.0)*hx/Neff**(1.0/p)

            if creep==True:
                u_creep =  -Beta*H**(n+1)*abs(hx)**(n-1)*hx
                u = u + u_creep
                diffusivity_flow = W*Beta*H**(n+2)*abs(hx)**(n-1)
                diffusivity = diffusivity + diffusivity_flow
            Vt = u[0]

            
            self.Hx = 0
            self.Ux = 0
            self.Wx = 0
            self.a = accum_rate[0]
            self.Hcx = 0
            self.Hc = Hc
            self.Vt = Vt
            self.W = W[0]
            self.dLdt = 0.0
        
            if advance==True:
                #dHc_dD = self.rho_w/self.rho*self.water_depth(xc)/(4*self.tau_c**2*(1-2*self.mu)**2/self.rho*2/self.g**2+self.rho_w/self.rho*self.water_depth(xc)**2)
                #w1=self.bed_topo.viscous_plate(x,H,dt)+self.bed_topo.bed
                #bed_new=spline(self.bed_topo.xx,w1,s=0.0)(xc)
                #w2=self.bed_topo.viscous_plate(x,self.H_old,dt)+self.bed_topo.bed
                #bed_old=spline(self.bed_topo.xx,w2,s=0.0)(xc)
                #dDdt = (bed_new-bed_old)/dt
                #dHcdt = dHc_dD*dDdt
                dHcdt = 0.0
                Ux = Delta*((1-self.rho_w/self.rho*self.water_depth(xc)**2/Hc**2)*Hc)**(n)
                #Ux = minimum(Ux,Ux_max)
                Hx = dot(Dxc,H)[0]
                #Hx = dot(Dxc,hsurf)[0]
                Hcx = dot(Dxc,self.calving_cliff_height(self.water_depth(x)))[0]
                # Calculate stretching rate based on force balance at terminus at rheology of ice (approximate)
                Ux = Delta*((1-self.rho_w/self.rho*self.water_depth(xc)**2/Hc**2)*Hc)**(n) # Need to account for stretching of yielded ice
                dx = x[0]-x[1]
                #Wx = (width(xc+dx)-width(xc-dx))/(2*dx)
                Wx = dot(Dxc,width(x))[0]
                #Vt = max(u)
                # Calculate width at terminus
                W = width(xc)
                dLdt = (accum_rate[0] -Vt*Hx - Ux*Hc - Vt*Hc*Wx/W)/(Hcx+dHcdt-Hx)
                #if dLdt<-1e3:
                #    dLdt = (accum_rate[0]+front_melt_fun[0]/3 -Vt*Hx - Ux*Hc-Vt*Hc*Wx/W - 0*front_melt_rate*water_depth(xc)/Hc)/(Hcx-Hx)
               
                dLdt = min(dLdt,Vt)
                dLdt = maximum(dLdt,-20e3)
                dLdt = minimum(dLdt,20e3)
                self.dLdt = dLdt
                self.Hx = Hx
                self.Ux = Ux
                self.Wx = Wx
                self.a = accum_rate[0]
                self.Hcx = Hcx
                self.Hc = Hc
                self.Vt = Vt
                self.W = W
                self.dLdt = dLdt
                
            else:
                dLdt = 0.0
            
            dLdt_vals.append(dLdt)
            Vt_vals.append(Vt)
            
            # Calculate non-linear diffusivity
            W = width(x)
            
            
            # Calculate diffusivity
            #diffusivity_sliding = W*Gamma*H/sqrt(hx**2+1e-16)*(H*abs(hx)-Tau_b)**(1.0/p)/Neff**(1.0/p)
            #diffusivity_flow = W*Beta*H**(n+2)*abs(hx)**(n-1)
            #diffusivity = diffusivity_sliding+ diffusivity_flow
            # Create differential operator
            L1 = dot(diag(1.0/W),dot(Dxc,dot(diag(diffusivity),Dxc))) 
            L2 = dot(diag(sigma*dLdt),Dxc)
            K =  eye(N+1) - omega*(L1+L2)*dt
            rhs = H + (accum_rate)*dt + (1-omega)*dot(L1+L2,H)*dt + dt*dot(L1,bed_topo(x))
            
            # Account for fixed ice thickness boundary condition at sigma = 1
            Hc = self.calving_cliff_height(self.water_depth(xc+dLdt*dt))
            K[0,:] = 0; K[0,0]=1.0; rhs[0] = Hc
            
            # Account for zero slope boundary condition at x = 0
            K[-1,:] = Dxc[-1,:]
            rhs[-1] = -dot(Dxc,bed_topo(x))[-1]
            
            #K[-1,:] = 0.0;K[-1,-1]=1.0
            #rhs[-1] = 0.0
            
            # Solve the dang thing
            self.H_old = H
            H_old = H
            H = reshape(array(linalg.solve(K,rhs)),(N+1))
            
            # Adjust calving front position and calculate new horizontal positions
            xc = xc + dt*dLdt
            x = sigma*xc 
            
            # Make sure ice thickness is positive definite -- thickness will only
            # be negative if something goes horribly wrong with the time stepping
            filter = H<10.0
            H[filter]=10.0
            
            
            # Update bed topography
            bed_topo.update(x,0.5*(H+H_old),dt)
            
            # Make sure ice never thins beneath flotation
            #hsurf=maximum(H+self.bed_topo(x),(1-self.rho/self.rho_w)*H) # Possible problem because 
            #H = hsurf - self.bed_topo(x)
            hsurf = bed_topo(x)+H
            #try:
            #    self.accum.__update__(Vc)
            #except:
            #    try:
            #        self.smb.__update__(Vc)
            #    except:
            #        pass
            
            # Update time and number of time steps
            time = start_time+(i+1)*dt
            
            #print time,start_time+(i+1)*dt,i,Nt
        Vt_ave = mean(Vt_vals)
        dLdt_ave = mean(dLdt_vals)    
        self.time.append(time)   
        self.hsurf = hsurf
        self.H = H
        self.L = xc
        self.u = u
        self.Vc = u[0]-dLdt
        return H,hsurf,xc,dLdt_ave,Vt_ave
    
class SsaFE(ModelFE):
    def get_term_vel(self):
        return self.u[-1]
    
    def get_term_thick(self):
        return self.H[-1]
    
    def get_term_hsurf(self):
        return self.hsurf[-1]
    
    
    
    def diag_solve(self,H,hsurf,xc,tau_b=0.0,front_melt_rate=0.0):
        """
        function solves diagnostic equation for ice sheet velocity
        Inputs : H - ice thickness
                 hsurf - surface elevation
                 plastic - optional flag if set set true solves for a plastic bed with plastic yield stress
                  tau_b
                 
        Returns velocity in units of m/a         
        """
        rho = self.rho
        rho_w = self.rho_w
        g = self.g
        n = self.n
        p = self.p
        C = self.C
        B = self.B
        plastic = self.plastic
        x=self.get_sigma(xc)
        
        N = self.N
        
        W = self.width(x)
        Wmid = 0.5*(W[0:-1]+W[1::])
        
        # Bed location
        bed = hsurf - H
                
        # Vector of grid spacing
        dx = diff(x)
        
        # Ice thickness at mid point of elements
        Hmid = 0.5*(H[0:-1]+H[1::])
        xmid = 0.5*(x[0:-1]+x[1::])
        # Regularization constants
        L0 = 100e3
        delta = 1.0/self.secpera 
        
        u = ones(size(hsurf))
        
       
        # Convert frontal melt rate to units of m/s from m/a
        front_melt_rate = front_melt_rate/self.secpera
        
        #water_depth = self.water_depth(x)
        water_depth = - bed*(bed<0)
        water_depth_mid = 0.5*(water_depth[0:-1]+water_depth[1::])
        Neff  = 1.0
    
        
        Pi = self.rho*self.g*Hmid
        Pw = minimum(self.rho_w*self.g*water_depth_mid,Pi)
        
        bed_mid = 0.5*(bed[1::]+bed[0:-1])
        #Neff = 1.0 + 5*0.5*(tanh((bed_mid-150.0)/150.0)+1)
        Neff = 1.0# + 3.5/2.0*0.5*(tanh((bed_mid-200.0)/200.0)+1)
        
        #Neff = 2.0
        
        #Neff = Pi-Pw
        #Cmax = 0.07
        #Cmin = 0.07
        #mu = ((Cmax-Cmin)*0.5*(tanh((bed_mid-1000.0)/500.0)+1) + Cmin)*Neff
        #mu = maximum(minimum(mu,150e3),50e3)
        #print max(mu)/1e3,min(mu)/1e3
        #mu = self.C*ones(shape(Wmid))
        #mu[xmid<50e3]=75e3
        #mu = 200e3
        mu = self.C  #+ self.mu*(Pi-Pw)
        if plastic==False:
            def bed_fric(xi):
                #f = 1.0 - Pw/Pi*exp(-((self.L-xmid)/(10*H[-1])))
                fric_coeff=C*((0.5*u[0:-1]*(1-xi)+0.5*u[1::]*(1+xi))**2 + delta**2)**(0.5*p-0.5)*(Pi>Pw)
                lat_drag = 2*Hmid/Wmid*B*(5/Wmid)**(1.0/n)*((0.5*u[0:-1]*(1-xi)+0.5*u[1::]*(1+xi))**2 + delta**2)**(0.5/n-0.5)
                #lat_drag = C*((0.5*u[0:-1]*(1-xi)+0.5*u[1::]*(1+xi))**2 + delta**2)**(0.5*p-0.5)*Hmid/Wmid
                um = 0.5*(u[0:-1]+u[1::])
                fric_coeff_plastic = self.tau_c/sqrt(um**2+delta**2)*(Pi>Pw)
                fric_coeff = minimum(fric_coeff_plastic,fric_coeff)
                #above_yield = fric_coeff*abs(umid)>self.tau_c
                #fric_coeff[above_yield] = self.tau_c/umid[above_yield]
                #above_yield=fric_coeff*abs(umid)>self.tau_c
                #if sum(above_yield)>0:
                #    print 'Basal layer is yielded',sum(above_yield)
                #if sum(above_yield)==0:
                #    print 'Basal layer NOT yielded',sum(above_yield)
                #fric_coeff[above_yield] = self.tau_c/sqrt(umid[above_yield]**2+delta**2)
                return fric_coeff+lat_drag
        elif plastic=='Plastic_Mix':
            def bed_fric(xi):
                um = 0.5*(u[0:-1]+u[1::])
                fric_coeff1 = C*((0.5*u[0:-1]*(1-xi)+0.5*u[1::]*(1+xi))**2 + delta**2)**(0.5*p-0.5)*Neff
                fric_coeff2 = mu/sqrt(um**2+delta**2)
                tau_bx1 = fric_coeff1*sqrt(um**2)
                tau_bx2 = fric_coeff2*sqrt(um**2)
                fric_coeff = minimum(tau_bx1,tau_bx2)/sqrt(um**2+delta**2)
                #fric_coeff = fric_coeff1*sqrt(um**2+delta**2)/sqrt(um**2+delta**2)
                #lat_drag = 2*Hmid/Wmid*B*(5/Wmid)**(1.0/n)*((0.5*u[0:-1]*(1-xi)+0.5*u[1::]*(1+xi))**2 + delta**2)**(0.5/n-0.5)
                #lat_drag = 2*Hmid/Wmid
                lat_drag1 = 2*C*((0.5*u[0:-1]*(1-xi)+0.5*u[1::]*(1+xi))**2 + delta**2)**(0.5*p-0.5)*Hmid/Wmid
                lat_drag2 = 2*(Hmid/Wmid)*mu/sqrt(um**2+delta**2)
                lat_drag  = minimum(lat_drag1*sqrt(um**2),lat_drag2*sqrt(um**2))/sqrt(um**2+delta**2)
                #fric_coeff = fric_coeff1
                return fric_coeff+lat_drag
        elif plastic=='All':
            # tau_b at bed and walls
            def bed_fric(xi):
                um = 0.5*(u[0:-1]+u[1::])
                fric_coeff = mu/sqrt(um**2+delta**2)*(Pi>Pw)
                #lat_drag = 2*Hmid/Wmid*B*(5/Wmid)**(1.0/n)*((0.5*u[0:-1]*(1-xi)+0.5*u[1::]*(1+xi))**2 + delta**2)**(0.5/n-0.5)
                lat_drag = 2*Hmid/Wmid*mu/sqrt(um**2+delta**2)
                return fric_coeff+lat_drag
        else:
            # tau_b at bed and tau_c at walls
            def bed_fric(xi):
                um = 0.5*(u[0:-1]+u[1::])
                fric_coeff = mu/sqrt(um**2+delta**2)*(Pi>Pw)
                #lat_drag = 2*Hmid/Wmid*B*(5/Wmid)**(1.0/n)*((0.5*u[0:-1]*(1-xi)+0.5*u[1::]*(1+xi))**2 + delta**2)**(0.5/n-0.5)
                lat_drag = 2*Hmid/Wmid*self.tau_c/sqrt(um**2+delta**2)
                return fric_coeff+lat_drag
            
        
       
        def N1(xi):
            N1 = 0.5*(1-xi)  # 1 at xi=-1, 0 at xi = 1
            return N1
        
        def N2(xi):
            N2 = 0.5*(1+xi)  # 0 at xi=-1, 1 at xi
            return N2
        
        
        if self.Ubc ==None:
            Ubc = 0.0
        else:
            Ubc = self.Ubc/self.secpera
        
        # Calculate rhs forcing (constant for loop)
        b1 = hstack((rho*g/6.0*(H[0:-1]**2 + H[0:-1]*H[1::] + H[1::]**2),0.0))
        b2 = hstack((0.0,rho*g/6.0*(H[0:-1]**2 + H[0:-1]*H[1::] + H[1::]**2)))
        
        v1 = hstack((rho*g/6.0*(2*H[0:-1] + H[1::])*(bed[1::]-bed[0:-1]),0.0))
        v2 = hstack((0.0,rho*g/6.0*(2*H[0:-1] + H[1::])*(bed[1::]-bed[0:-1])))
        
        rhs = (b1 -b2) + v1 + v2
        rhs = rhs[1::]
        # Apply essential boundary condition at x=1
        rhs[-1] = rhs[-1] + 0.5*rho_w*g*bed[-1]**2*(bed[-1]<0.0)
        
        # This is where we will stick the effect of frontal melting on the near terminus strain rate field
        z = zeros(size(rhs))
        
        itnum = 0.0
        tol =  1.0/self.secpera
        err=9999.9
        while err>tol:
            
            uold = u.copy()
            
       
            gamma = sqrt((diff(u)/dx)**2+(delta/L0)**2)

            
            cj = 2*B*sqrt((diff(u)/dx)**2+(delta/L0)**2)**(1.0/n-1.0)*Hmid
            e = (cj/dx)
            e1 = hstack((0,e))
            e2 = hstack((e,0))
            
            # Stiffness matrix            
            K = -spdiags([e1[1::]+e2[1::]],0,N-1,N-1) + spdiags(e[1::],-1,N-1,N-1) + spdiags(e,1,N-1,N-1) 

            # Mass matrix using two point quadrature
            pt1 = -1.0/sqrt(3)
            pt2 =  1.0/sqrt(3)
            
            a1 = hstack((0.5*dx*(N1(pt1)**2*bed_fric(pt1) + N1(pt2)**2*bed_fric(pt2)),0.0))
            a2 = hstack((0.0,0.5*dx*(N2(pt1)**2*bed_fric(pt1) + N2(pt2)**2*bed_fric(pt2))))
            a3 = 0.5*dx*(N1(pt1)*N2(pt1)*bed_fric(pt1)) + 0.5*dx*(N1(pt2)*N2(pt2)*bed_fric(pt2))
            M = spdiags([a1[1::]+a2[1::]],[0],N-1,N-1) + spdiags(a3[1::],-1,N-1,N-1) + spdiags(a3,1,N-1,N-1)
    
            # Finite element operator
            L = K - M
            
            #z[-1] = -0.5*front_melt_rate*cj[-1]/H[-1]
            # Not sure why the factor of two was there before.
            # This should be cj*du/dx + cj*(dudx_melt) = right hand side
            # du_dx_melt  = front_melt/H so
            # cj*du_dx = right hand side - cj*front_melt/H
            # Think this is correct, but had a factor of two before
            z[-1] = -front_melt_rate*cj[-1]/H[-1]#*(H[-1]-H[-2]) /dx[-1]
            u = array(spsolve(L,rhs+z))
            

            # Add boundary condition back in as rigid body motion
            
            u = hstack((0.0,u)) + Ubc

            # Calculate difference between new and old velocity
            err = max(abs(u-uold))
            
            # Break out of iteration if number of iterations is too large
            itnum = itnum +1
            if itnum>1000:
                disp('Diagnostic solve failed to converge')
                break
            
        return u*self.secpera
    
    def prog_solve_testing(self,dt,H,hsurf,u,dLdt,xc,fixed_calving_front=True,Hc=None):
        """
        Inputs:
            H - ice thickness
            hsurf - surface elevation
            u - glacier velocity
            dLdt - rate of terminus advance/retreawt
            xc - Terminus position
        Optional inputs:
            fixed_calving_front - True or False if True then fix calving front thickness to be Hc
            Hc - Fixed calving front thickness
            
        Returns Updated ice thickness
        
        Example:
        H = prog_solve(H,hsurf,u,dLdt,xc,fixed_calving_front=False,Hc=None):
        
        NOTE: Implicit time step is only semi-implicit - we use the velocity at t instead of t+1 to advect the ice
        """
        
        Beta    = 2/(self.n+2)*(self.rho*self.g/self.B)**self.n*self.secpera
        bed = hsurf-H
        water_depth = -bed*(bed<0.0)
        
        
        # x-coordinate
        x=self.get_sigma(xc)
        
        W = self.width(x)

        N = self.N
         
        # Grid spacing
        dx = diff(x)

        # Calculate accumulation rate and frontal melt rate    
        accum_rate, front_melt = self.accum(x,hsurf,self.get_time())
            
        front_melt_fun = 2*(1-(xc-x)/H[-1])*self.water_depth(xc)/H[-1]*front_melt
        front_melt_fun = front_melt_fun*(front_melt_fun>0.0)
        accum_rate = accum_rate - front_melt_fun
        front_melt = 0.0

        # If the calving front thickness is not provided, we assume it is given by the yield strength
        water_depth = -bed*(bed<0.0)
        
        #if Hc==None:
        Hc = self.calving_cliff_height(water_depth[-1]) 
        H_new = H.copy()
        
        # Create mass matrix for ice thickness (stencil [1/6*dx0, 1/3*dx0+1/3*dx1, 1/6*dx1])
        e1 = hstack((dx,0.0))
        e2 = hstack((0,dx))
        M = spdiags((e1+e2)/3,0,N,N) + spdiags(e2/6,1,N,N) + spdiags(e1/6,-1,N,N)
        
        # Create mass matrix for diffusion term (stencil )
        hx = (hsurf[1::]-hsurf[0:-1])/(x[1::]-x[0:-1])
        Hmid = 0.5*(H[0:-1]+H[1::])
        Wmid = 0.5*(W[0:-1]+W[1::])
        diffusivity = Beta*Hmid**(self.n+2)*abs(hx)**(self.n-1)
        e1 = hstack((diffusivity/dx,0.0))
        e2 = hstack((0.0,diffusivity/dx))
        K = spdiags(e1,-1,N,N) -spdiags(e1+e1,0,N,N) + spdiags(e2,1,N,N)
        lam=1e12
        
        
        L = M-K
        L[-1,-1]=L[-1,-1]+lam
        # Rhs forcing vector
        rhs = H + dt*(e1*accum_rate + e2*accum_rate)/3
        rhs[0:-1] = rhs[0:-1] + dx*accum_rate[1::]/6
        rhs[1::] = rhs[1::] + dx*accum_rate[0:-1]/6
        rhs[-1] = Hc*lam
        #rhs[0]=-(bed[1]-bed[0])
        
        rhs = rhs + K.dot(bed)
        
        
        H_new = array(spsolve(L,rhs))
       
        H_new[H_new<0.0]=1.0
        
        return H_new
    
    def prog_solve(self,dt,H,hsurf,u,dLdt,xc,accum_rate,fixed_calving_front=True,Hc=None):
        """
        Inputs:
            H - ice thickness
            hsurf - surface elevation
            u - glacier velocity
            dLdt - rate of terminus advance/retreawt
            xc - Terminus position
        Optional inputs:
            fixed_calving_front - True or False if True then fix calving front thickness to be Hc
            Hc - Fixed calving front thickness
            
        Returns Updated ice thickness
        
        Example:
        H = prog_solve(H,hsurf,u,dLdt,xc,fixed_calving_front=False,Hc=None):
        
        NOTE: Implicit time step is only semi-implicit - we use the velocity at t instead of t+1 to advect the ice
        """
        
        bed = hsurf-H
        water_depth = -bed*(bed<0.0)
        
        
        # x-coordinate
        x=self.get_sigma(xc)
        
        W = self.width(x)

        N = self.N
        # Grid spacing
        dx = diff(x)

        # Calculate accumulation rate and frontal melt rate    
        #accum_rate,front_melt_rate = self.accum(x,hsurf,self.time[-1])
        
        #accum_rate,front_melt_rate = self.accum(x,hsurf,self.time[-1]) # Accumulation rate
        #front_melt_fun = 2*(1-(xc-x)/H[-1])*self.water_depth(xc)/H[-1]*front_melt_rate
        #front_melt_fun = front_melt_fun*(front_melt_fun>0.0)
        #accum_rate = accum_rate - front_melt_fun
        #front_melt_rate = 0.0

        # If the calving front thickness is not provided, we assume it is given by the yield strength
        water_depth = -bed*(bed<0.0)
        #if Hc==None:
        Hc = self.calving_cliff_height(water_depth[-1]) 
        H_new = H.copy()
        
        # Take fully implicit time step
        
        # Mass matrix for terminus advance term
        b1 = 0.5*dt*x/xc*dLdt*hstack((0,1/dx))
        b1[-1] = b1[-1]*2
        b2 = 0.5*dt*x/xc*dLdt*hstack((1/dx,0))
        b2[0] = b2[0]*0
        b3 = hstack((0,0.5*dt*x[0:-1]/xc*dLdt/dx))
        b3[-1]=b3[-1]
        M = spdiags(b1-b2,0,N,N) - spdiags(b1[1::],-1,N,N) + spdiags(b3,1,N,N)
        
        
        # Mass matrix for upwind advection
        ddx = hstack((dx[0],dx))
        e0 = ones((N,));e0[0]=0.0
        e1 = dt/ddx*u
        e1[0] = -1.0
        e2=dt/dx*u[0:-1]*W[0:-1]/W[1::]
        e3 = zeros(size(dx))
        #e3 = -dt/dx*x[1::]/xc*dLdt
        e3[0] = 1.0
        e3[1]=1.0
        A = spdiags(e1,0,N,N) -spdiags(e2,-1,N,N) + spdiags(e3,1,N,N)
        I = spdiags(e0,0,N,N)
        omega = 1.0
        L = I + omega*(A - M)
        
        # Rhs forcing
        rhs = H + dt*accum_rate - (1-omega)*(A-M).dot(H)
        
        rhs[0]=-(bed[1]-bed[0])
        #rhs[0] = 0.0
        
        H_new = array(spsolve(L,rhs))
        #hsurf = array(spsolve(L,rhs))
        #H_new = hsurf - bed
        H_new[H_new<0.0]=1.0
        
        return H_new
     
    def sia_diffusion(self,dt,H,hsurf,xc,accum_rate,Hc,f,creep=True):
        
        # Flow law exponent
        n = self.n
        p = self.p
        
        # Number of points
        N = self.N
        
        # Horizontal coordinate
        x=self.get_sigma(xc)
        
        # Define some dimensional constants and convert time unit to annum
        Beta    = 2/(self.n+2)*(self.rho*self.g/self.B)**self.n*self.secpera
        Gamma   = (self.rho*self.g/self.C)**(1.0/p)*self.secpera # Dimensional constant in annum

        # Width of glacier
        W = self.width(x)
        
        # Bed topography
        bed = hsurf- H
        bed_mid = 0.5*(bed[1::]+bed[0:-1])
        
        # Grid spacing
        dx = x[1::]-x[0:-1]
    
              
        # Surface slope, midpoint ice thickness, width and nodal diffusivities
        hx = (hsurf[1::]-hsurf[0:-1])/dx
        Hmid = 0.5*(H[0:-1]+H[1::])
        Wmid = 0.5*(W[0:-1]+W[1::])
        fmid = 0.5*(f[0:-1]+f[1::])
        
        water_depth = self.water_depth(x)
        water_depth_mid = 0.5*(water_depth[0:-1]+water_depth[1::])        
        Pi = self.rho*self.g*Hmid
        Pw = self.rho_w*self.g*water_depth_mid
        #Neff = 1.0 + 5*0.5*(tanh((bed_mid-150.0)/150.0)+1)
        Neff = 1.0 + 5*0.5*(tanh((bed_mid-150.0)/150.0)+1)+ 2*Hmid*self.B/(Wmid*self.C)*(5/Wmid)**(self.p)
        diffusivity = fmid*Wmid*Gamma*Hmid/sqrt(hx**2+1e-12)*(Hmid*abs(hx))**(1.0/p)/Neff**(1.0/p)
        Vt = -fmid*Gamma/sqrt(hx**2+1e-12)*(Hmid*abs(hx))**(1.0/p)/Neff**(1.0/p)*hx
        
        if creep==True:
            diffusivity_creep = fmid*Wmid*Beta*Hmid**(self.n+2)*abs(hx)**(self.n-1)
            Vt_creep  = -fmid*Beta*Hmid**(n+1)*abs(hx)**(self.n-1)*hx
            #diffusivity_sliding = fmid*Wmid*Gamma*Hmid/sqrt(hx**2+1e-12)*(Hmid*abs(hx))**(1.0/p)/Neff**(1.0/p)
            diffusivity = diffusivity + diffusivity_creep
            #Vt_slide = -fmid*Gamma/sqrt(hx**2+1e-12)*(Hmid*abs(hx))**(1.0/p)/Neff**(1.0/p)*hx
            Vt = Vt + Vt_creep
      
        
        # Create stiffness matrix (stencil [-D0/dx0, (D0/dx0 +D1/dx1), -D1/dx1 )
        e1 = hstack((diffusivity/dx,0.0))
        e2 = hstack((0.0,diffusivity/dx))
        K = -spdiags(e1,-1,N,N) +spdiags(e1+e2,0,N,N) - spdiags(e2,1,N,N) # no boundary conditions yet
        
        
        # Create mass matrix for ice thickness (stencil 1/6[dx0, 2*dx0+2*dx1, 1*dx1])
        e1 = hstack((dx*Wmid,0.0))
        e2 = hstack((0,dx*Wmid))
        M = spdiags((e1+e2)/3,0,N,N) + spdiags(e2/6,1,N,N) + spdiags(e1/6,-1,N,N)
        
        omega = 2.0
        L = (M+omega*dt*K)
        
        # Rhs forcing vector
        rhs = M.dot(H) - dt*K.dot(bed) + dt*M.dot(accum_rate) - (1-omega)*dt*K.dot(H)
        f = L[:,-1]*Hc # Column that multiples H[-1]
        
        # Strip these columnes out
        L2 = L[0:-1,0:-1]
        rhs2 = rhs[0:-1] - f[0:-1].transpose()
        
        # Solve system
        H_new = hstack((array(spsolve(L2,rhs2.transpose())),Hc))
  
        
        H_new[H_new<0.0]=1.0
        Vt = hstack((0.0,0.5*(Vt[0:-1]+Vt[1::]),Vt[-1]))
        return H_new,Vt
    
    def integrateSIA(self,dt=1.0,time_interval=100e3,advance = False,fixed_calving_front=False,creep=False):
        
        H = self.H
        hsurf = self.hsurf
        xc = self.L
        # First check to make sure we are appropriately initialized
        flag = self.check_initialization()
        if flag!=0:
            return None
    
        sigma = self.sigma
        
        # Create physical coordinate
        x = sigma*xc
        
        Delta   = (self.rho *self.g/self.B/4)**self.n*self.secpera # Dimensional rheology paratmer in annum
        
        hsurf=maximum(H+self.bed_topo(x),(1-self.rho/self.rho_w)*H)
        H = hsurf - self.bed_topo(x)
        
        time = self.get_time()
        start_time = self.get_time()
        
        Nt = int(round((time_interval)/(dt),1))
        
        
        for i in xrange(Nt):
            
            
            accum_rate,front_melt_rate = self.accum(x,hsurf,time) # Accumulation rate
            front_melt_rate = front_melt_rate*self.water_depth(xc)/H[-1]
            Hc = self.calving_cliff_height(self.water_depth(xc))
            H_new,uc = self.sia_diffusion(dt,H,hsurf,xc,accum_rate,Hc,f=ones(size(x)),creep=creep)
            
            dLdt=0.0
            # Update terminus position
            self.bed_topo.update(x,0.5*(H+H_new),dt)
            H = H_new
            #hsurf = H + self.bed_topo(x)
            hsurf=maximum(H+self.bed_topo(x),(1-self.rho/self.rho_w)*H)
            H = hsurf - self.bed_topo(x)
            
            # Update time and number of time steps
            time = start_time+(i+1)*dt
        
        self.u=uc
        self.H = H
        self.hsurf = hsurf
        self.L = xc
        self.u = uc
        self.time.append(time)
          
        return H,hsurf, xc, dLdt, uc[-1]
    
    def integrate(self,dt=1.0,time_interval=100e3,advance = False,fixed_calving_front=False,creep=False):
        # First check to make sure we are appropriately initialized
        flag = self.check_initialization()
        if flag!=0:
            return None
        H = self.H
        xc = self.L
        sigma = self.sigma
        
        bed_topo = self.bed_topo
        
        # Create physical coordinate
        x = sigma*xc
        
        Delta   = (self.rho *self.g/self.B/4)**self.n*self.secpera # Dimensional rheology paratmer in annum
        
        #hsurf=maximum(H+self.bed_topo(x),(1-self.rho/self.rho_w)*H)
        #H = hsurf - self.bed_topo(x)
        
        # Testing
        #hsurf = H + self.bed_topo(x)
        #hsurf=maximum(H+self.bed_topo(x),(1-self.rho/self.rho_w)*H)
        #H = hsurf-self.bed_topo(x)
        hsurf=maximum(H+self.bed_topo(x),(1-self.rho/self.rho_w)*H) # Possible problem because 
        H = hsurf - self.bed_topo(x)
        # Testing
        
        time = self.get_time()
        start_time = self.get_time()
        
        Nt = int(round((time_interval)/(dt),1))
        
        dLdt_vals = []
        Vt_vals =  []
        for i in xrange(Nt):
            
            
            accum_rate,front_melt_rate = self.accum(x,hsurf,time) # Accumulation rate
            front_melt_rate = front_melt_rate*self.water_depth(xc)/H[-1]
            u = self.diag_solve(H,hsurf,xc,front_melt_rate=front_melt_rate)
        
            
            # Define terminus velocity
            Vt = u[-1]
            
            
            # Rate of terminus advance
            dx = x[-1]-x[-2]
            Hx = (H[-1]-H[-2])/dx
            Hc = self.calving_cliff_height(self.water_depth(xc))
            Ux = Delta*((1-self.rho_w/self.rho*self.water_depth(xc)**2/H[-1]**2)*H[-1])**(self.n)
            #Ux = (u[-1]-u[-2])/dx
            W = self.width(xc)
            Wx = (self.width(xc+dx)-self.width(xc-dx))/(2*dx)
            Hcx = (self.calving_cliff_height(self.water_depth(xc+dx)) - self.calving_cliff_height(self.water_depth(xc-dx)))/2/dx
            dLdt_new = (H[-1]-Hc)/dt + (accum_rate[-1] -Vt*Hx - Ux*H[-1]-Vt*H[-1]*Wx/W- front_melt_rate*0)/(Hcx-Hx)
   
            if advance == False:
                dLdt = 0.0
            else:
                dLdt = dLdt_new
                err = 9999.9
                kk=0
                itnum = 0
                while (err>0.1):
                    H_n = self.prog_solve(dt,H,hsurf,u,dLdt,xc,accum_rate,fixed_calving_front=fixed_calving_front)
                    H_new = H_n[-1]
                    Hc = self.calving_cliff_height(self.water_depth(xc+dLdt*dt))
                    Hcx = (self.calving_cliff_height(self.water_depth(xc+dLdt*dt)) - self.calving_cliff_height(self.water_depth(xc+dLdt*dt-dx)))/dx
                    dLdt = dLdt + ((H_new-Hc)/dt)/(Hcx-Hx)
                    err = abs(Hc-H_new)
                    kk=kk+1
                    
                    if itnum>25:
                        print 'Failed to converge'
                        break
                    itnum = itnum+1
                #dx_max = 10*(x[-1]-x[-2])/dt
                #dLdt = minimum(maximum(dLdt,-dx_max),Vt)
                    
            # Calving front yield strength
            #Hc = self.calving_cliff_height(self.water_depth(xc+dLdt*dt))
            H_old = H
            H = self.prog_solve(dt,H,hsurf,u,dLdt,xc,accum_rate,fixed_calving_front=fixed_calving_front)
            #H = H_n
            #print Hc-H[-1],err,fixed_calving_front
            # Update terminus position
            xc = xc + dLdt*dt
            x = sigma*xc
            
            # Update bed topography
            #self.bed_topo.update(x,0.5*(H+H_old),dt)
            Vc = u[-1]-dLdt
            try:
                self.accum.__update__(Vc)
            except:
                try:
                    self.smb.__update__(Vc)
                except:
                    pass
           
            # Make sure ice never thins beneath flotation
            hsurf=maximum(H+self.bed_topo(x),(1-self.rho/self.rho_w)*H) # Possible problem because 
            H = hsurf - self.bed_topo(x)
            # Testing
            #hsurf = H + self.bed_topo(x)
            #hsurf=maximum(H+self.bed_topo(x),(1-self.rho/self.rho_w)*H)
            #H = hsurf-self.bed_topo(x)
            # Testing
            
            # Update bed topography
            bed_topo.update(x,0.5*(H+H_old),dt)
            
            # Make sure ice never thins beneath flotation
            #hsurf=maximum(H+self.bed_topo(x),(1-self.rho/self.rho_w)*H) # Possible problem because 
            #H = hsurf - self.bed_topo(x)
            #hsurf = bed_topo(x)+H
            
           
            # Update time and number of time steps
            time = start_time+(i+1)*dt
            dLdt_vals.append(dLdt)
            Vt_vals.append(u[-1])
            self.H = H
            self.hsurf = hsurf
            self.L = xc
            self.u = u
            self.dLdt = dLdt
            self.Vc = u[-1]-dLdt
        Vt_ave = mean(Vt_vals)
        dLdt_ave = mean(dLdt_vals)
                
        self.time.append(time)
       
        return H,hsurf,xc,dLdt_ave, Vt_ave


        
class Tributary(object):
        """
        Class for tributaries
        
          SHOULD THIS BE A SUB-CLASE OF SIA CLASS WITH EXTRA STUFF???  I THINK IT SHOULD
        """
        def __init__(self,master,slave,x1,x2,ID):
            # Define flowline (shoudl by Sia or SsaFE)
            self.flowline=slave
            self.xmin = x1 # Upstream intersection of tributary with main flowline
            self.xmax = x2 # Downstream intersection of tributary with main flowline
            self.intersect_width = x2-x1 # Width of intersection
            self.intersect = True # Default is that tributary is assumed to intersect with master flowline 
            self.master = master
            self.flowline.calving_cliff_height = self.calving_cliff_height
            self.ID = ID
            Dx = self.master.D
            nx,ny = shape(Dx)
            from numpy.linalg import inv
            Di = inv(Dx[0:-1,0:-1])
            self.quad_weights = hstack((Di[0,:],0.0))
            
        def smoothing_kernel(self,x):
            """
            Creates smoothing kernel to weight contributions of glacier
            to smb and contribution of master glacier to terminus ice thickness
            based on position relative to the centerline
            """
            
            xmax = self.xmax
            xmin = self.xmin
            
            #xmax = max(x)
            #xmin = min(x)
            
            #L = self.master.L
            w = xmax-xmin
            #xi = (x-xmin)/w
            #kernel = xi*(1-xi)
            #eta = (max(x)-xmin)/w
            #kernel = kernel/sum(kernel)*(-2*eta**3+3*eta**2)
            #kernel = kernel/sum(kernel)*(6*eta**5-15*eta**4+10*eta**3)#(-2*eta**3+3*eta**2)
            xi = 2*(x-xmin)/w-1.0
            eta = max(xi)
            factor = (1.0/16*(3*eta**2-9*eta+8))*(eta+1)**3
            #factor = (xmax-self.xmin)/(self.xmax-self.xmin)
            
            #if max(x)<(xmax-0.1*w):
            #    xi = 2*(x-xmin)/(max(x)-min(x))
            kernel = (1-xi**2)**2
            kernel = kernel/sum(kernel)*factor
            #dx=(max(x)-xmin)/len(xi)
            #kernel = 2*15.0/16.0*(1-xi**2)**2*dx/w
            
            #print sum((kernel-kernel2)**2)
            #factor = (1.0/16*(3*eta**2-9*eta+8))*(eta+1)**3
            #kernel = kernel*factor
            return kernel

        def get_flux(self,x,H):
            xmax = self.xmax
            xmin = self.xmin
            idx = find((x>xmin) & (x<xmax))
            flux_vector = zeros(size(x))
            if len(idx)>0:
                xpts = x[idx]
                kernel=self.smoothing_kernel(xpts)
            #    w = xmax-xmin
                
            #    dx_tmp = abs(diff(xpts))
            #    dx = 0.5*(dx_tmp[0:-1]+dx_tmp[1::])
            #    if idx[0]>0:
            #        dx_new = abs(x[idx[0]]-x[idx[0]-1])
            #        if len(idx)>1:
            #            dx_old = abs(x[idx[1]]-x[idx[0]])
            #        else:
            #            dx_old = abs(x[idx[0]]-x[idx[0]-1])
            #        dx = hstack((0.5*(dx_new+dx_old),dx))                
            #    else:
            #        dx_new = abs(x[idx[0]]-x[idx[0]+1])
            #        dx = hstack((dx_new,dx))    
            #    if idx[-1]<=len(x)-2:
            #        dx_new = abs(x[idx[-1]]-x[idx[-1]+1])
            #        if len(idx)>1:
            #            dx_old = abs(x[idx[-1]]-x[idx[-2]])
            #        else:
            #            dx_old = abs(x[idx[-1]]-x[idx[-1]-1])
            #        dx = hstack((dx,0.5*(dx_new+dx_old)))                
            #    else:
            #        dx_new = abs(x[idx[0]]-x[idx[0]-1])
            #        dx = hstack((dx,dx_new))
               
                # Calculate velocity
                #hsurf = self.master.hsurf
                #hx = (hsurf[idx[0]]-hsurf[idx[-1]])/(x[idx[0]]-x[idx[-1]])
                #bed = mean(self.master.bed_topo(x[idx]))
                #w = mean(self.flowline.width(x[idx]))
                Hc = self.calving_cliff_height(self.flowline.water_depth(self.flowline.L))
                #Neff = 1.0 + 5*0.5*(tanh((bed-150.0)/150.0)+1)+ 2*Hc*self.flowline.B/(w*self.flowline.C)*(5/w)**(self.flowline.p)
                #Gamma   = (self.flowline.rho*self.flowline.g/self.flowline.C)**(1.0/self.flowline.p)*self.flowline.secpera
                #term_vel = -Gamma*(Hc*abs(hx))**(3.0)*hx/(abs(hx)+1e-16)/Neff**(3.0)
                #trib_thick = H[idx]        
                #Hc = dot(kernel,trib_thick)
                
                flux= max(self.flowline.get_term_vel(),0.0)*Hc*self.flowline.width(self.flowline.L)
                #flux= term_vel*Hc*w
                #Dx = self.master.D/self.master.L
                #nx,ny = shape(Dx)
                #from numpy.linalg import inv
                #Di = inv(Dx[0:-1,0:-1])
                #w = hstack((Di[0,:],0.0))
                
                w=maximum(self.quad_weights[idx]*self.master.L,0.1*Hc)
                #w=self.quad_weights[idx]*self.master.L
                
                #dx = (max(xpts)-min(xpts))/len(idx)
                flux_vector[idx] = flux*kernel/w
                # No accumulation from tributaries at first or last points?
                flux_vector[0]=0
                flux_vector[-1]=0
                #w=self.quad_weights*self.master.L
                #flux_vector[0]=0.0
                #flux_vector[-1]=0.0
                #flux_vector[0]=
                #print 'Flux',flux,'Flux vector trap method',(-trapz(flux_vector,x)-flux)/flux*100,'Flux vector spectral method',(sum(dot(flux_vector,w))-flux)/flux*100
           
            return flux_vector
        
        def calving_cliff_height(self,D):
            
            # Find portion of terminus in contact with master flowline
            try:
                # Get default parameters
                x = self.master.get_x()
            except ValueError:
                print "Need to initialize flowline with an initial length before adding tributaries"
            
            #H = self.master.hsurf
            
            # This will fail if the tributary is too small to include any grid
            #    points.  Maybe this is fine?
            filter = find((x> self.xmin) & (x<self.xmax))
            # if some part of the terminus is in contact then find average
            # master flowline ice thickness
            if sum(filter)>0:
                trib_surf = self.master.hsurf[filter]
                xpts = x[filter]
                kernel = self.smoothing_kernel(xpts)
                #dx=(max(xpts)-xmin)/sum(idx)
                # Average thickness of terminus based on master thickness
                #hh = dot(kernel,trib_thick)
                Hc =  (dot(kernel,trib_surf) - self.flowline.bed_topo(self.flowline.L))
                if Hc>self.flowline.H[1]:
                    self.flowline.H[1]=Hc
            
                #Hc = mean(trib_thick)- self.flowline.bed_topo(self.flowline.L)
                #print mean(trib_thick),sum(kernel)
                # If some points intersect, assume that 
                self.intersect = True
            # if the master flowline is no longer in contact with the terminus
            # then calculate yield thickness
            else:
                #print "Tributary",self.ID," doesn't intersect",self.xmin,self.xmax
                g = self.flowline.g
                tau_c = self.flowline.tau_c
                mu = self.flowline.mu
                rho=self.flowline.rho
                rho_w = self.flowline.rho_w
                Hmax = 2*tau_c/(rho*g*(1-2*mu)) + sqrt((2*tau_c/(rho*g*(1-2*mu)))**2+rho_w/rho*D**2)
                Hc = maximum(Hmax,rho_w/rho*D)
                if (self.xmin>max(x)):
                    # Only assume that tributary doesn't intersect if xmin is greater than max of x
                    self.intersect=False
                else:
                    self.intersect=True
            #print 'Water depth', D
            Hc = maximum(Hc,0.0)
            return Hc
        
        
        def integrate(self, dt,time_interval,trib_advance = False):
            ### NEED TO FIGURE OUT HOW TO INTEGRATE CREEEP AND OTHER FLAGS INTO INTEGRATE FUNCTION (**kwargs?)
            # Tributaries can only advance/retreat if trib_advance flag is set to True
            # and they no longer intersect with the master flowline
            advance_flag = ((self.intersect == False))
            #print self.ID,advance_flag,self.intersect
            # Forward integration of tributary flowline 
            H,hsurf,L,dLdt,Vt=self.flowline.integrate(dt=dt,time_interval=time_interval,advance=advance_flag,creep=True)
            return H,hsurf,L,dLdt,Vt
            
                     
class GlacierNetwork(Sia):
    """
    Class that defines glacier network ojbect
    """
    def __init__(self,kind='spectral',L=100e3,*args,**kwargs):
        
        # Initializes all variables based on class Model
        super(GlacierNetwork, self).__init__(*args, **kwargs)
        self.L = L
        #self.accum = self.accum_smb_trib
        
        
        # Vector to store tributaries 
        self.tributaries = []
        self.N_trib = 0
       
    def accum(self,x,hsurf,time):
        
        # First get flux from tributaries
        H = hsurf - self.bed_topo(x)
        flux = zeros(size(x))
        for trib in self.tributaries:
            q=trib.get_flux(x,H)
            flux = flux + q
        
        # Effective accumulation from
        #   tributaries is given by flux/tributary width/flowline width
        trib_accum = flux/self.width(x)
        trib_accum = minimum(trib_accum,1000.0)
        # Total accumulation
        smb,front_melt_rate = self.smb_fun(x,hsurf,time)
        accum_rate = smb + trib_accum
        
        
        return accum_rate,front_melt_rate
    
    def set_accumulation(self,accum):
        self.smb_fun = accum
        
    
    #def set_accumulation(self,accum_fun):
    #    """
    #    Overiding default accumulation function
    #        In this case the accumulation has a component due
    #        to snow accumulation/surface melt and a component
    #        due to the flux from tributaries.  We set the snow accum
    #        function here and overide the accum function with a function
    #        that includes both smb and tributary fluxe
    #    """
    #    self.smb_fun = accum_fun
    
    
    def get_x(self):
        try:
            # Get default parameters
            x = self.get_sigma(self.L)
        except ValueError:
            print "Need to initialize flowline with an initial length before adding tributaries"
        return x
    
    def add_tributary(self,model,xmin,xmax,ID):
        """
        Add tributary to network
            Tributary should be a flowline from class tributary
            model = instance of Sia or Ssa class
        """
        
        x=self.get_x()
            
        # Initialize instance of Tributary class
        trib = Tributary(self,model,xmin,xmax,ID)
        
        # Add tributary to list of tributaries
        self.tributaries.append(trib)
        
        # Increment number of tributaries
        self.N_trib = self.N_trib + 1
        
    def integrate_all(self, dt,time_interval,advance = False, trib_advance = False, creep=True):
        time = self.get_time()
        start_time = self.get_time()
        
        Nt = int(round((time_interval)/(dt),1))
        
        dLdt_vals = []
        Vt_vals =  []
        
        for i in xrange(Nt):
            H,hsurf,L,dLdt,Vt=self.integrate(dt=dt,time_interval=dt,advance=advance,creep=True)
            Vt_vals.append(Vt)
            dLdt_vals.append(dLdt)
            #self.time[-1]=self.time[-1]+time_interval
            for trib in self.tributaries:
                if trib_advance == True:
                    if trib.intersect == True:
                        trib_advance_flag = False
                    else:
                        trib_advance_flag = True
                else:
                    trib_advance_flag = False
                #trib_advance = False
                #print self.ID,trib_advance,trib.intersect
                Ht,hsurft,Lt,dLdtt,Vtt=trib.flowline.integrate_all(dt=dt,time_interval=dt,advance=trib_advance_flag,trib_advance=trib_advance, creep=True)
            
            #dLdt = 0.0
            #Vt = zeros(shape(self.H))[0]
        dLdt = mean(dLdt_vals)
        Vt = mean(Vt_vals)
        return self.H,self.hsurf,self.L,dLdt,Vt
    
    
    def integrate_self(self, dt,time_interval,advance = None,master_advance = False, trib_advance = False, creep=None):
        H,hsurf,L,dLdt,Vt=self.integrate(dt=dt,time_interval=time_interval,advance=advance,creep=True)
        return self.H,self.hsurf,self.L,dLdt,Vt
    
    
    
    
class GlacierNetworkFE(SsaFE):
    """
    Class that defines glacier network ojbect
    """
    def __init__(self,kind='spectral',L=100e3,*args,**kwargs):
        
        # Initializes all variables based on class Model
        super(GlacierNetworkFE, self).__init__(*args, **kwargs)
        self.L = L
        #self.accum = self.accum_smb_trib
        
        [sigma,D]=chebdiff(self.N-1)
        #self.sigma = (sigma+1)/2 # Definse sigma so that it goes from [0,1]
        self.D = 2*D 
        # Vector to store tributaries 
        self.tributaries = []
        self.N_trib = 0
       
    def accum(self,x,hsurf,time):
        
        # First get flux from tributaries
        H = hsurf - self.bed_topo(x)
        flux = zeros(size(x))
        for trib in self.tributaries:
            q=trib.get_flux(x,H)
            flux = flux + q
        
        # Effective accumulation from
        #   tributaries is given by flux/tributary width/flowline width
        trib_accum = flux/self.width(x)
        trib_accum = minimum(trib_accum,1000.0)
        
        # Total accumulation
        smb,front_melt_rate = self.smb_fun(x,hsurf,time)
        accum_rate = smb + trib_accum
        return accum_rate,front_melt_rate
    
    def set_accumulation(self,accum):
        self.smb_fun = accum
        
        
    
    #def set_accumulation(self,accum_fun):
    #    """
    #    Overiding default accumulation function
    #        In this case the accumulation has a component due
    #        to snow accumulation/surface melt and a component
    #        due to the flux from tributaries.  We set the snow accum
    #        function here and overide the accum function with a function
    #        that includes both smb and tributary fluxe
    #    """
    #    self.smb_fun = accum_fun
    
    
    def get_x(self):
        try:
            # Get default parameters
            x = self.get_sigma(self.L)
        except ValueError:
            print "Need to initialize flowline with an initial length before adding tributaries"
        return x
    
    def add_tributary(self,model,xmin,xmax,ID):
        """
        Add tributary to network
            Tributary should be a flowline from class tributary
            model = instance of Sia or Ssa class
        """
        
        x=self.get_x()
            
        # Initialize instance of Tributary class
        trib = Tributary(self,model,xmin,xmax,ID)
        
        # Add tributary to list of tributaries
        self.tributaries.append(trib)
        
        # Increment number of tributaries
        self.N_trib = self.N_trib + 1
        
    def integrate_all(self, dt,time_interval,advance = False,trib_advance = False, creep=True):
        time = self.get_time()
        start_time = self.get_time()
        
        Nt = int(round((time_interval)/(dt),1))
        
        dLdt_vals = []
        Vt_vals =  []
        
        for i in xrange(Nt):
            H,hsurf,L,dLdt,Vt=self.integrate(dt=dt,time_interval=dt,advance=advance,creep=True)
            Vt_vals.append(Vt)
            dLdt_vals.append(dLdt)
            #self.time[-1]=self.time[-1]+time_interval
            for trib in self.tributaries:
                if trib_advance == True:
                    if trib.intersect == True:
                        trib_advance_flag = False
                    else:
                        trib_advance_flag = True
                else:
                    trib_advance_flag = False
                #trib_advance = False
                #print trib.ID,trib.intersect,trib_advance_flag,trib_advance,trib.flowline.L
                Ht,hsurft,Lt,dLdtt,Vtt=trib.flowline.integrate_all(dt=dt,time_interval=dt,advance=trib_advance_flag,trib_advance=trib_advance, creep=True)
                #if trib.ID == 'Trib15':
                #    print float(i)/Nt,trib.ID,'Length',trib.flowline.L,'dLdt',trib.flowline.dLdt,'Advance flag',trib_advance_flag
                #    from pylab import *
                #    figure(1)
                #    clf()
                #    plot(trib.flowline.get_sigma(trib.flowline.L)/1e3,trib.flowline.hsurf,'k')
                #    plot(trib.flowline.get_sigma(trib.flowline.L)/1e3,trib.flowline.hsurf-trib.flowline.H,'k')
                #    draw()
                #    show()
            #dLdt = 0.0
            #Vt = zeros(shape(self.H))[0]
        dLdt = mean(dLdt_vals)
        Vt = mean(Vt_vals)
        return self.H,self.hsurf,self.L,dLdt,Vt
    
    
    def integrate_self(self, dt,time_interval,advance = False,master_advance = False, trib_advance = False, creep=True):
        H,hsurf,L,dLdt,Vt=self.integrate(dt=dt,time_interval=time_interval,advance=advance)
        return self.H,self.hsurf,self.L,dLdt,Vt