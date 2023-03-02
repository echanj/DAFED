#!/usr/bin/python2.7

# coding: utf-8

# ## Langevin Dynamics ##
# 
# first modified by Eric Chan to perfrom Langevin dynamics using a 2D harmonic oscillator  
# 2nd modification is to perfrom setup of AFED algoritham 
#  see M.Tuckerman text book page. 347 
#
# In this notebook you will use a Verlet scheme to simulate the dynamics of a 1D- Harmonic Oscillator and 1-D double well potential using Langevin Dynamics

# also the ABOBA scheme as described in supporting info from Leimkuler and Mathews is in terms of 
# momentum and position updates.
# this means we need to be careful when using volocilites and factor 1/sqrt(Mass) for the random force part.
#
#
# to be honest I am unsure if this implementation is correct
# so if anyone want to branch and make correction then go for it  
#    

# to do: 
#    -  measure velocity autocorelations 
#    -  measure of temperature
#    -  driver interface for alternative inputs 

# In[ ]:


#setup the notebook
# get_ipython().run_line_magic('pylab', 'inline')
import numpy as np
import matplotlib 
from matplotlib import pyplot as plt 
import matplotlib.cm as cm

# ## Part 1, set up the potential and plot it  ##
# This follows from the Molecular Dynamics Homework
# 
# This function has to return the energy and force for a 1d harmonic potential. The potential is
# U(x) = 0.5 k (x - x0)^2 and F = -dU(x)/dx|

# In[ ]:

#this function returns the energy and force on a particle from a harmonic potential
def coupled_oscillator_energy_force(x,y,d0=5.0,a=1.0,k=1.0,lamb=2.878):
    #calculate the energy on force on the right hand side of the equal signs
    energy = d0*(x**2.0-a**2.0)**2.0 + 0.5*k*y**2.0 + lamb*x*y
    force_x = -(2.0*d0*(x**2.0-a**2.0)*2.0*x + lamb*y) 
    force_y = -(k*y + lamb*x)
 
    return energy, force_x, force_y


def Free_energy_analytical(x,d0=5.0,a=1.0,k=1.0,lamb=2.878):
    #calculate the energy on force on the right hand side of the equal signs
    A_energy = d0*(x**2.0-a**2.0)**2.0 - 0.5*(1.0/k)*(lamb**2)*(x**2.0)
 
    return A_energy 



#this function will plot the energy and force
#it is very general since it uses a special python trick of taking arbitrary named arguments (**kwargs) 
#and passes them on to a specified input function
def plot_energy_force(function, xmin=-2,xmax=2,spacing=0.1,**kwargs):
    x_points = np.arange(xmin,xmax+spacing,spacing)
    y_points = np.arange(xmin,xmax+spacing,spacing)
    X, Y = np.meshgrid(x_points, y_points)                   # need 2D-plot
    energies, xforces, yforces = function(X,Y,**kwargs)

    plt.figure()
    label = 'U(x)'
    for arg in kwargs:
        label=label+', %s=%s'%(arg,str(kwargs[arg]))
    levels = np.arange(0.0, 15.0, 0.5)   # contour ranges
    CS = plt.contour(X, Y, energies, levels=levels)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('%s'%label)
    plt.xlabel('x-position')
    plt.ylabel("y-position")
    plt.legend(loc=0)

    plt.figure()
    label = 'Force(x)'
    for arg in kwargs:
        label=label+', %s=%s'%(arg,str(kwargs[arg]))
    levels = np.arange(-80.0, 80.0, 1.0)   # contour ranges
    CS = plt.contour(X, Y, xforces, levels=levels)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('%s'%label)
    plt.xlabel('x-position')
    plt.ylabel("y-position")
    plt.legend(loc=0)

    plt.figure()
    label = 'Force(y)'
    for arg in kwargs:
        label=label+', %s=%s'%(arg,str(kwargs[arg]))
    levels = np.arange(-10.0, 10.0, 1.0)   # contour ranges
    CS = plt.contour(X, Y, yforces, levels=levels)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('%s'%label)
    plt.xlabel('x-position')
    plt.ylabel("y-position")
    plt.legend(loc=0)

#    plt.plot(x_points,xforces,label='forces',color=p[0].get_color(),linestyle='--')
    
##   we can plot the energy (solid) and forces (dashed) to see if it looks right
# plot_energy_force(coupled_oscillator_energy_force,d0=5.0,a=1.0,k=1.0,lamb=2.878)

# plt.show()
# exit()

# plot_energy_force(harmonic_oscillator_energy_force,k=2)


# ## Part 2, code langevin dynamics ##
# Now you will implement the BAOAB scheme of [Leimkuhler and Matthews (JCP, 2013)](https://aip.scitation.org/doi/abs/10.1063/1.4802990)
# 
# The following equations are repeated (Do B,A,O,A,B then repeat) to move forward in time. The A and B steps represent increments by half a time step. 
# 
# B: $v(t) \leftarrow v(t) + \frac{F(t)}{m} (dt/2)$
# 
# A: $x(t) \leftarrow x(t) + v(t) (dt/2)$
# 
# 
# The differential equation for the O process is
# 
# $\frac{d v(t)}{dt} = - \gamma v dt + \sqrt{2 \gamma k_B T/m} d W$
# 
# ($dW$ is a random differential that samples a gaussian)
# 
# Solving this tells us the update rule:
# 
# O: $v(t) \leftarrow e^{-\gamma dt} v(t) + R(t)\sqrt{k_B T/m} \sqrt{1-e^{-2\gamma dt}} $
# 
# where $R(t)$ is a gaussian random number with mean zero and standard-deviation 1.
# 
# ** In the following, I'm setting the mass $m=1$ **

# In[ ]:

#this is step A
def position_update_x(x,vx,dt):
    x_new = x + vx*dt/2.0
    return x_new

def position_update_y(y,vy,dt,ref_M):
    y_new = y + vy*dt/(4.0*ref_M)
    return y_new

#this is step B
# take mass into account when updating velocity 
# for AFED scheme we make this 0.25*dt
def velocity_update_x(vx,Fx,dt,massx,ref_M):
    vx_new = vx + Fx*dt/(ref_M*4.0*massx) 
    return vx_new

def velocity_update_y(vy,Fy,dt,massy,ref_M):
    vy_new = vy + Fy*dt/(ref_M*4.0*massy) 
    return vy_new

def random_velocity_update_x(vx,gammax,kBTx,dt,massx):
    Rx = np.random.normal()
    c1x = np.exp(-gammax*dt)
    c2x = np.sqrt(1.0-c1x*c1x)*np.sqrt(kBTx)
    vx_new = c1x*vx + (Rx*c2x)/np.sqrt(massx)   
    return vx_new

def random_velocity_update_y(vy,gammay,kBTy,dt,ref_M,massy):
    Ry = np.random.normal()
    c1y = np.exp(-(gammay*dt*0.5)/ref_M)
    c2y = np.sqrt(1.0-c1y*c1y)*np.sqrt(kBTy)
    vy_new = c1y*vy + (Ry*c2y)/np.sqrt(massy)
    return vy_new

# this AFED implemetation of langevin dynamics for a 2D-coupled occilator using the baoab scheme 
# its not too tricky and should read baoab(aoa)baoab  where the inner aoa is the position update for the heavy thermostat

# 
# note: AFED is not just about crossing the barrier but obtaining better sampling statistics at the barrier 
#
def AFED_baoab(potential, max_time, dt, gammax, gammay, kBTx, kBTy, 
          initial_position_x, initial_position_y, initial_velocity_x, initial_velocity_y, massx, massy, ref_M,
                                        save_frequency=3, **kwargs ):
    x = initial_position_x
    y = initial_position_y
    vx = initial_velocity_x
    vy = initial_velocity_y
    t = 0
    step_number = 0
    x_positions = []
    y_positions = []
    x_velocities = []
    y_velocities = []
    total_energies = []
    save_times = []
    temps_x = []
    temps_y = []
    
    while(t<max_time):
  

#--------- do the exp(il2) part
      
        for m in range(ref_M): 
         # B
         potential_energy, force_x, force_y = potential(x,y,**kwargs)
         vx = velocity_update_x(vx,force_x,dt,massx,ref_M)
         vy = velocity_update_y(vy,force_y,dt,massy,ref_M)
      #   print "first v %.3f"%v 
         
         #A
         y = position_update_y(y,vy,dt,ref_M)
         
         #O
         vy = random_velocity_update_y(vy,gammay,kBTy,dt,ref_M,massy)
         
         #A
         y = position_update_y(y,vy,dt,ref_M)
         
         # B
         potential_energy, force_x, force_y = potential(x,y,**kwargs)
         vx = velocity_update_x(vx,force_x,dt,massx,ref_M)
         vy = velocity_update_y(vy,force_y,dt,massy,ref_M)

# --------do the exp(iL1ref) part 

        #A
        x = position_update_x(x,vx,dt)

        #O
        vx = random_velocity_update_x(vx,gammax,kBTx,dt,massx)
        
        #A
        x = position_update_x(x,vx,dt)


#--------- do the exp(il2) part again 
      
        for m in range(ref_M): 
        # B
         potential_energy, force_x, force_y = potential(x,y,**kwargs)
         vx = velocity_update_x(vx,force_x,dt,massx,ref_M)
         vy = velocity_update_y(vy,force_y,dt,massy,ref_M)
      #   print "first v %.3f"%v 
         
         #A
         y = position_update_y(y,vy,dt,ref_M)
         
         #O
         vy = random_velocity_update_y(vy,gammay,kBTy,dt,ref_M,massy)
         
         #A
         y = position_update_y(y,vy,dt,ref_M)
         
         # B
         potential_energy, force_x, force_y = potential(x,y,**kwargs)
         vx = velocity_update_x(vx,force_x,dt,massx,ref_M)
         vy = velocity_update_y(vy,force_y,dt,massy,ref_M)

# ------------------


      #  print "second v %.3f"%v 
        
        if step_number%(max_time*10) == 0 : print('time  %.6f' %t)
        if step_number%save_frequency == 0 and step_number>0:
            e_total = .5*massx*vx*vx + .5*massy*vy*vy + potential_energy  # KE + PE
            temp_x = (.5*massx*vx*vx)*(2.0/3.0)  # from T=(2*KE)/(3*N)  
            temp_y = (.5*massy*vy*vy)*(2.0/3.0)  # from T=(2*KE)/(3*N)  

            x_positions.append(x)
            y_positions.append(y)
            x_velocities.append(vx)
            y_velocities.append(vy)
            total_energies.append(e_total)
            save_times.append(t)
            temps_x.append(temp_x)
            temps_y.append(temp_y)
        
        t = t+dt
        step_number = step_number + 1
    
    return save_times, x_positions, y_positions, x_velocities, y_velocities, total_energies,temps_x,temps_y   


# ## Part 3, run  Langevin Dynamics simulation of a harmonic oscillator ##
# 
# 1) Change `my_k` and see how it changes the frequency
# 
# 2) Set `my_k=1`, and change my_gamma. Try lower values like 0.0001, 0.001, and higher values like 0.1, 1, 10. Do you see how underdamped, low $\gamma$, looks more like standard harmonic oscillator, while overdamped, high $\gamma$ looks more like a random walk? 
# 

# In[ ]:

# note: it may be that you need to adjust the gamma 'drag' coefficents differntly for the differnet thermostats
#       since the two systems are actually at differnt temperatures
#        Also, since you know the force constants and masses 
#            you can get an idea of the frequency for each of the two systems 
#                         


my_d0=5.0
my_a=1.0
my_k=1.0
my_lamb= 2.878   # if I set this to zero I will decouple x and y
my_max_time =  1000 # 1000
initial_position_x = 0.0
initial_position_y = 0.0
initial_velocity_x = 0.0
initial_velocity_y = 0.0

massx=300.0 # 300.0
massy=1.0
my_gammax=1.0
my_gammay=1.0
my_kBTx=10.0  #10.0
my_kBTy=1.0
my_dt=0.01
ref_M=1 # 50

times, x_positions, y_positions, x_velocities, y_velocities, total_energies,temps_x,temps_y = AFED_baoab(coupled_oscillator_energy_force,
                                                    my_max_time, my_dt, my_gammax, my_gammay,  my_kBTx, my_kBTy,
                                              initial_position_x, initial_position_y, initial_velocity_x, initial_velocity_y,
                                              massx,massy,ref_M,d0=my_d0,a=my_a,k=my_k,lamb=my_lamb)

# axes[0, 0].plot(A, 'xr')               # this creates plot 
# axes[0, 0].set_ylabel('val')
# axes[0, 0].set_title('field values')
#
# axes[0, 1].hist(A)             
# axes[0, 1].set_title('distribution')
#
# axes[1, 0].plot(np.log(np.abs(A)))             
# axes[1, 0].set_title('log of |values|')
#
# plt.show()  
# ---------------------
#  plot the first part
# 

fig, axes = plt.subplots(2, 2, figsize=(8,6))

axes[0, 0].plot(times,x_positions,marker='.',label='x-position',linestyle='')
# axes[0, 0].plot(times,y_positions,marker='.',label='y-position',linestyle='')
# axes[0, 0].plot(times,x_velocities,marker='',label='x-velocity',linestyle='-')
# axes[0, 0].plot(times,y_velocities,marker='',label='y-velocity',linestyle='-')
axes[0, 0].set_xlabel('time')
axes[0, 0].legend(loc=1)


# draw contour for the energy 
spacing=0.1
xmin=np.min(x_positions)
xmax=np.max(x_positions)
ymin=np.min(y_positions)
ymax=np.max(y_positions)

x_points = np.arange(xmin,xmax+spacing,spacing)
y_points = np.arange(ymin,ymax+spacing,spacing)
X, Y = np.meshgrid(x_points, y_points)                   # need 2D-plot
energies, xforces, yforces = coupled_oscillator_energy_force(X,Y,d0=my_d0,a=my_a,k=my_k,lamb=my_lamb)

label = 'U(x)'
# for arg in kwargs:
#    label=label+', %s=%s'%(arg,str(kwargs[arg]))
# levels = np.arange(0.0, 15.0, 0.5)   # narrow contour ranges
levels = np.arange(0.0, 15.0, 3.0)   # wide contour ranges

axes[0, 1].contour(X, Y, energies, levels=levels)
# axes[0, 1].imshow(energies, interpolation='bilinear', origin='lower', cmap=cm.hot)

# plt.clabel(CS, inline=1, fontsize=10)
# plt.title('%s'%label)
# plt.legend(loc=0)


axes[0, 1].plot(x_positions,y_positions,marker='.',color='b',alpha=0.3,markerfacecolor='none',markeredgewidth=0.05,markersize=2,label='2D-trajectory',linestyle='')
axes[0, 1].set_xlabel('x-position')
axes[0, 1].set_ylabel('y-position')
axes[0, 1].legend(loc=1,prop={'size':6})




# axes[1, 0].plot(times,total_energies,marker='o',linestyle='',label='Simulated E')
axes[1, 0].plot(times,temps_x,marker='',linestyle='-',label='Temp X')
axes[1, 0].plot(times,temps_y,marker='',linestyle='-',label='Temp Y')
axes[1, 0].set_xlabel('time')
axes[1, 0].set_ylabel("Total Energy")
axes[1, 0].legend(loc=1,prop={'size':10})


# ---------------------------------
# plot a bare potential along x with no coupling 
# -----------------


x_points = np.arange(-2.0,2.0+0.1,0.1)
e0,fx0,fy0= coupled_oscillator_energy_force(x_points,0.0,d0=5.0,a=1.0,k=1.0,lamb=0.000)
axes[1, 1].plot(x_points,e0,label='bare potential',color='k',linestyle='--')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('A(x)')
axes[1, 1].legend(loc='upper center',prop={'size':6})


# ### Part 4, Histogram Position and Velocity ###
# 
# What is the probability of seeing a given position or velocity? 
# 
# Now we are supposedly sampling the canonical distribution, so we should have:
# 
# $P(x) = \frac{1}{\sqrt{2 \pi k_B T /k}} e^{-\frac{k (x-x_0)^2}{2 k_B T}}$
# 
# $P(v) = \frac{1}{\sqrt{2 \pi k_B T /m}} e^{-\frac{ m v^2}{2 k_B T}}$
# 
# $P(E) = e^{-E/k_B T}/\int e^{-E/k_B T} dE = \frac{1}{k_B T} e^{-E/k_B T}$
# 
# ** Set gamma to overdamped above and run the following cell **
# 
# The histograms will be compared to the exact formulas.

# In[ ]:


def bin_centers(bin_edges):
    return (bin_edges[1:]+bin_edges[:-1])/2.

# def gaussian_x(x,k,kBT):
#     denominator = np.sqrt(2*np.pi*kBT/k)
#     numerator = np.exp(-k*(x**2)/(2*kBT))
#    return numerator/denominator

# def gaussian_v(v,kBT):
#     denominator = np.sqrt(2*np.pi*kBT)
#     numerator = np.exp(-(v**2)/(2*kBT))
#     return numerator/denominator

#to get a good histogram, we need to run a lot longer than before
# my_max_time = 3000
# my_gamma=0.1
# my_kBT=1.1

# times, positions, velocities, total_energies = baoab(harmonic_oscillator_energy_force, 
#                                                                            my_max_time, my_dt, my_gamma, my_kBT, \
#                                                                            initial_position, initial_velocity,\
#                                                                             k=my_k)
#let's data from the entire trajectory, so we can diagnose 
# for future reference it is better to use np.histogram() 
dist_hist, dist_bin_edges = np.histogram(x_positions,bins=30,density=True)
vel_hist, vel_bin_edges = np.histogram(x_velocities,bins=30,density=True)

#let's only use data from the second half of the trajectory, so it can equilibrate
# for future reference it is better to use np.histogram() 
# updated throw away the first quarter 
# dist_hist, dist_bin_edges = np.histogram(x_positions[-len(x_positions)//4:],bins=30,density=True)
# vel_hist, vel_bin_edges = np.histogram(x_velocities[-len(x_velocities)//4:],bins=30,density=True)
# e_hist, e_bin_edges = np.histogram(total_energies[-len(total_energies)//2:],bins=20,density=True)

FE_calc=-my_kBTx*np.log(dist_hist)
FE_calc=FE_calc-np.min(FE_calc)

#ideal_prediction_x = gaussian_x(x=bin_centers(dist_bin_edges),k=my_k,kBT=my_kBT )
axes[1, 1].plot(bin_centers(dist_bin_edges),FE_calc,marker='o',label='P(x)',linestyle='')
axes[1, 1].set_ylim(-1.0,25.0)
axes[1, 1].set_xlim(-2.5,2.5)
# plt.plot(bin_centers(dist_bin_edges), ideal_prediction_x,linestyle='--',label='', color=p[0].get_color())

# plot the analytical helmholtz free energy
A_helm=Free_energy_analytical(bin_centers(dist_bin_edges),d0=my_d0,a=my_a,k=my_k,lamb=my_lamb)
A_helm=A_helm-np.min(A_helm)

#ideal_prediction_v = gaussian_v(v=bin_centers(vel_bin_edges),kBT=my_kBT )
#p = plt.plot(bin_centers(vel_bin_edges), vel_hist,marker='s',label='P(v)',linestyle='')
axes[1, 1].plot(bin_centers(dist_bin_edges), A_helm,linestyle='-.',marker='v', color='r',label='Analytical Free Energy')
axes[1, 1].legend(loc='upper right',prop={'size':6})

#plt.figure()
#p = plt.plot(bin_centers(e_bin_edges), e_hist,marker='s',label='P(E)',linestyle='')

#compute the energy histogram values to the boltzman factors for the observed energies
# plt.plot(bin_centers(e_bin_edges), np.exp(-bin_centers(e_bin_edges)/my_kBT)/my_kBT,linestyle='--',color=p[0].get_color())
# plt.yscale('log')
# plt.xlabel("E")
# plt.ylabel("P(E)")

savename = 'AFED_coupled_harmonic_graph_outputs.png' 
plt.savefig(savename,dpi=300)   # this is a bit bigger than the default DPI which will give higher resoltion
plt.show()

exit()


# ## Simulate a double well potential ##
# Let's do a simulation in a double well also
# 
# $U(x) = \frac{k}{4} (x-a)^2 (x+a)^2$
# 
# This potential has a minimum at $x=a$ and $x=-a$. It also has a barrier at $x=0$. 

# In[ ]:


#this function returns the energy and force on a particle from a double well
def double_well_energy_force(x,k,a):
    #calculate the energy on force on the right hand side of the equal signs
    energy = 0.25*k*((x-a)**2) * ((x+a)**2)
    force = -k*x*(x-a)*(x+a)
    return energy, force

plot_energy_force(double_well_energy_force, xmin=-4,xmax=+4, k=1, a=2)
plt.axhline(0,linestyle='--',color='black')
plt.axvline(0,linestyle='--',color='black')
plt.ylim(-10,10)


# ### Part 5, run langevin verlet dynamics on the double well ###
# We will see what happens when we change temperature `my_KBT` and barrier height `my_a`.
# 
# 1) Run the simulation as is and see that the particle samples both the left and right sides of the well
# 
# 2) Lower the temperature to 0.1, what happens?
# 
# 3) Keep the temperature at 1.0, and raise $a$ to 3, what happens?
# 
# *when is the sampling ergodic?*

# In[ ]:

my_k = 1

#CHANGE THESE
my_kBT = 1.0
my_a = 2

plot_energy_force(double_well_energy_force, xmin=-4,xmax=+4, k=my_k, a=my_a)
plt.ylim(-10,10)


my_initial_position = my_a
my_initial_velocity = 1

my_gamma = 0.1  # 0.1
my_dt = 0.05
my_max_time = 2000


times, positions, velocities, total_energies = baoab(double_well_energy_force, 
                                                                            my_max_time, my_dt, my_gamma, my_kBT,\
                                                                            my_initial_position, my_initial_velocity,\
                                                                             k=my_k, a=my_a)

plt.figure()
plt.plot(times,positions,marker='',label='position',linestyle='-')
plt.plot(times,velocities,marker='',label='velocity',linestyle='-')

plt.xlabel('time')
plt.legend(loc='upper center')

plt.figure()
initial_energy = total_energies[0]
plt.plot(times,total_energies,marker='',linestyle='-',label='Simulated E')
plt.xlabel('time')
plt.ylabel("Total Energy")
plt.legend()

# histogramming the results
plt.figure()

dist_hist, dist_bin_edges = np.histogram(positions,bins=25,density=True)
vel_hist, vel_bin_edges = np.histogram(velocities,bins=25,density=True)

p = plt.plot(bin_centers(dist_bin_edges), dist_hist,marker='o',label='P(x)',linestyle='')

#test against exact prediction
dd = 0.1
test_bin_positions = np.arange(-10,10,dd)
double_well_energies, double_well_froces = double_well_energy_force(test_bin_positions,my_k,my_a)

plt.plot(test_bin_positions, np.exp(-double_well_energies/my_kBT)/np.sum(dd*np.exp(-double_well_energies/my_kBT)),     linestyle='--',color=p[0].get_color())

p = plt.plot(bin_centers(vel_bin_edges), vel_hist,marker='s',label='P(v)',linestyle='')
ideal_prediction_v = gaussian_v(v=bin_centers(vel_bin_edges),kBT=my_kBT )
plt.plot(bin_centers(vel_bin_edges), ideal_prediction_v,linestyle='--',label='', color=p[0].get_color())


plt.legend(loc='upper left')

plt.show()


############
# #  
# # this is the other way to do it which is neater and has more control
# # optional sub plot
# 
#  fig, axes = plt.subplots(2, 2, figsize=(8,8))
# axes[0, 0].plot(A, 'xr')               # this creates plot 
# axes[0, 0].set_ylabel('val')
# axes[0, 0].set_title('field values')
#
# axes[0, 1].hist(A)             
# axes[0, 1].set_title('distribution')
#
# axes[1, 0].plot(np.log(np.abs(A)))             
# axes[1, 0].set_title('log of |values|')
#
# plt.show()  
#
##################


