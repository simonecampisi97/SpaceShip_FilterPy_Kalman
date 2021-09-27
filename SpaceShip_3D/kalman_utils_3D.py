import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch, HandlerCircleCollection
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

import os
from IPython.display import Image, display

#from model_evaluation_3D import plot_covariance_ellipsoide


from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.common import Saver



DT= 0.01
SIGMA=0.5

class Trajectoy3DGenerattion:

    def __init__(self,sigma=0.5, T=10.0, fs=100.0):
        
        
        global DT,SIGMA

        self.fs = fs # Sampling Frequency
        self.dt = 1.0/fs
        
        # Set Global Variables
        DT = self.dt
        SIGMA = sigma 
        
        self.T =  T # measuremnt time
        self.m = int(self.T/self.dt) # number of measurements
        self.sigma = sigma

        self.px= 0.0 # x Position Start
        self.py= 0.0 # y Position Start
        self.pz= 1.0 # z Position Start

        self.vx = 10.0  # m/s Velocity at the beginning
        self.vy = 0.0 # m/s Velocity
        self.vz = 0.0 # m/s Velocity

        c = 0.1 # Drag Resistance Coefficient

        self.Xr=[]
        self.Yr=[]
        self.Zr=[]
        
        self.Vx=[]
        self.Vy=[]
        self.Vz=[]
        
        self.ax =[]
        self.az =[]

        for i in range(int(self.m)):
            
            # Just to simulate a trajectory
            accx = -c*self.vx**2  
            
            self.vx += accx*self.dt
            self.px += self.vx*self.dt

            accz = -9.806 + c*self.vz**2 
            self.vz += accz*self.dt
            self.pz += self.vz*self.dt
                
            self.Xr.append(self.px)
            self.Yr.append(self.py)
            self.Zr.append(self.pz)
            
            self.Vx.append(self.vx)
            self.Vy.append(self.vy)
            self.Vz.append(self.vz)

            self.az.append(accz)
            self.ax.append(accx)
    
        aux = self.Xr
        self.Xr = self.Zr
        self.Zr = aux

        aux = self.Vx
        self.Vx = self.Vz
        self.Vz = aux

        aux = self.ax
        self.ax = self.az
        self.az = aux


    def get_velocities(self):
        return self.Vx, self.Vy, self.Vz

    def get_trajectory_position(self):
        return np.array(self.Xr), np.array(self.Yr), np.array(self.Zr)

    def get_acceleration(self):
        return self.ax, self.az
    
    def get_measurements(self):

        #adding Noise
        np.random.seed(25)
        self.Xm = self.Xr + self.sigma * (np.random.randn(self.m))
        self.Ym = self.Yr + self.sigma * (np.random.randn(self.m))
        self.Zm = self.Zr + self.sigma * (np.random.randn(self.m)) 
        
        return self.Xm, self.Ym, self.Zm


def plot_planets(x, y, z, ax):
    ax.scatter(x[0], y[0], z[0], c='b', s=850, facecolor='b')
    ax.scatter(x[-1], y[-1], z[-1], c='gray', s=350, facecolor='b')
    e_txt = ax.text(x[0]-3, y[0], z[0]-10.5,"Earth", weight='bold', c="b", fontsize=10)
    m_txt = ax.text(x[-1]-4, y[-1], z[-1]+4,"Moon", weight='bold', c="gray", fontsize=10)

    return e_txt, m_txt


def plot_measurements_3D(traj, ax, title=""):
    
    x,y,z = traj.get_measurements()
    
    plot_planets(x, y, z, ax)
    
    
    ax.scatter(x, y, z, c='g', alpha=0.3, facecolor=None, label="Measurements")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=15)

    #ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # Axis equal
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 3.0
   
    mean_x = x.mean()
    mean_y = y.mean()
    mean_z = z.mean()
    ax.set_xlim(mean_x - max_range, mean_x + max_range)
    ax.set_ylim(mean_y - max_range, mean_y + max_range)
    ax.set_zlim(mean_z - max_range, mean_z + max_range)
    ax.legend(loc='best',prop={'size':15})

    
def plot_trajectory_3D(traj, ax, title=""):
    
    x,y,z = traj.get_trajectory_position()
    
    plot_planets(x, y, z, ax)
    
    
    ax.plot(x, y, z, c='r', lw=2, ls="--", label="Trajectory")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=15)

    #ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # Axis equal
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 3.0
   
    mean_x = x.mean()
    mean_y = y.mean()
    mean_z = z.mean()
    ax.set_xlim(mean_x - max_range, mean_x + max_range)
    ax.set_ylim(mean_y - max_range, mean_y + max_range)
    ax.set_zlim(mean_z - max_range, mean_z + max_range)
    ax.legend(loc='best',prop={'size':15})

def plot_prediction(preds,traj, ax):
    
    global SIGMA

    xt, yt, zt = preds[:,0], preds[:,1], preds[:,2]
    Xr, Yr, Zr = traj.get_trajectory_position()
    Xm, Ym, Zm = traj.get_measurements()
    
    print("Xm: ", Xm.shape)
    print("Ym: ", Ym.shape)
    print("Zm: ", Zm.shape)

    plot_planets(Xr, Yr, Zr, ax)

    ax.plot(xt,yt,zt, lw=2, label='Kalman Filter Estimate')
    ax.plot(Xr, Yr, Zr, lw=2, label='Real Trajectory Without Noise')
    
    ax.scatter(Xm, Ym, Zm, edgecolor='g', alpha=0.1, lw=2, label="Measurements")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='best',prop={'size':15})
    ax.set_title("Kalman Filter Estimate - Sigma={}".format(SIGMA), fontsize=15)

    # Axis equal
    max_range = np.array([Xm.max()-Xm.min(), Ym.max()-Ym.min(), Zm.max()-Zm.min()]).max() / 3.0
    mean_x = Xm.mean()
    mean_y = Ym.mean()
    mean_z = Zm.mean()
    ax.set_xlim(mean_x - max_range, mean_x + max_range)
    ax.set_ylim(mean_y - max_range, mean_y + max_range)
    ax.set_zlim(mean_z - max_range, mean_z + max_range)




def plot_x_z_2D(ax, traj, preds):
    
    global SIGMA

    xt, yt, zt = preds[:,0], preds[:,1], preds[:,2]
    Xr, Yr, Zr = traj.get_trajectory_position()
    Xm, Ym, Zm = traj.get_measurements()

    ax.plot(xt,zt, label='Kalman Filter Estimate')
    ax.scatter(Xm,Zm, label='Measurement', c='gray', s=15, alpha=0.5)
    ax.plot(Xr, Zr, label='Real')
    ax.set_title("Kalman Filter Estimate 2D - Sigma={}".format(SIGMA), fontsize=15)
    ax.legend(loc='best',prop={'size':15})

    ax.set_xlabel('X ($m$)')
    ax.set_ylabel('Y ($m$)')

#-------------------------- KALMAN FUNCTIONS -------------------------------------------------

def init_kalman(traj):
    
    global SIGMA, DT

    #Transition_Matrix matrix
    PHI =   np.array([[1.0, 0.0, 0.0, DT, 0.0, 0.0, 1/2.0*DT**2, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0,  DT, 0.0, 0.0, 1/2.0*DT**2, 0.0],
                       [0.0, 0.0, 1.0, 0.0, 0.0,  DT, 0.0, 0.0, 1/2.0*DT**2],
                       [0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  DT, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  DT, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  DT],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    # Matrix Observation_Matrix
    #We are looking for the position of the spaceship x,y,z
    H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    x, y, z = traj.get_measurements()
    vx, vy, vz = traj.get_velocities()
    ax, az = traj.get_acceleration()
    #initial state
    init_states = np.array([x[0], y[0], z[0], vx[0], vy[0], vz[0], ax[0], 0., az[0]])

    P = np.eye(9)*(0.5**2)
   
    rp = 1  # Noise of Position Measurement
    R = np.eye(3)* rp
    
    G = np.array([  [(DT**2)/2],
                    [(DT**2)/2],
                    [(DT**2)/2],
                    [    DT   ],
                    [    DT   ],
                    [    DT   ],
                    [    1.   ],
                    [    1.   ],
                    [    1.   ]])

    acc_noise = 0.1 # acceleration proccess noise
    Q= np.dot(G, G.T)* acc_noise**2
    #Q = Q_discrete_white_noise(3, dt=DT, var=50, block_size=3) 

    return init_states, PHI, H, Q, P, R



def Ship_tracker(traj):
    
    global DT

    init_states, PHI, H, Q, P, R = init_kalman(traj)
    
    tracker= KalmanFilter(dim_x = 9, dim_z=3)
    tracker.x = init_states

    tracker.F = PHI
    tracker.H = H   # Measurement function
    tracker.P = P   # covariance matrix
    tracker.R = R  # state uncertainty
    
    tracker.Q =  Q # process uncertainty
    
    return tracker


def run(tracker, traj):
    
    x, y, z = traj.get_measurements() 
    zs = np.asarray([ x, y, z]).T
   
    preds, cov = [],[]
    
    for z in zs:
        tracker.predict()
        tracker.update(z=z)
        
        preds.append(tracker.x)
        cov.append(tracker.P)
    
    return np.array(preds), np.array(cov)

def run_half_measures(tracker, traj):
    
    x, y, z = traj.get_measurements()
    zs = np.asarray([ x, y, z]).T
    
    preds, cov = [],[]
    
    for i,z in enumerate(zs):
        tracker.predict()
        
        if i  <= len(zs)//2:
            tracker.update(z=z)
    
        preds.append(tracker.x)
        cov.append(tracker.P)
    
    return np.array(preds), np.array(cov)



def run_even_index_update(tracker, traj):
    
    x, y, z = traj.get_measurements()
    zs = np.asarray([ x, y, z]).T

    preds, cov = [],[]
    
    for i, z in enumerate(zs):
        
        tracker.predict()
        
        if( i  % 2 == 0):
            tracker.update(z=z)
        
        preds.append(tracker.x)
        cov.append(tracker.P)
    
    return np.array(preds), np.array(cov)


def run_update_every_5(tracker, traj):
    
    x, y, z = traj.get_measurements()
    zs = np.asarray([ x, y, z]).T
    preds, cov = [],[]
    
    for i, z in enumerate(zs):
        
        tracker.predict()
        
        if( i  % 5 == 0):
            tracker.update(z=z)
        
        preds.append(tracker.x)
        cov.append(tracker.P)
    
    return np.array(preds), np.array(cov)

def run_update_hole_in_middle(tracker, traj):
    
    x, y, z = traj.get_measurements()
    zs = np.asarray([ x, y, z]).T
    preds, cov = [],[]
    
    chunk = len(zs) // 3
    for i, z in enumerate(zs):
        
        tracker.predict()
        if i <= chunk or i >= 2*chunk:
            tracker.update(z=z)
        
        preds.append(tracker.x)
        cov.append(tracker.P)
    
    return np.array(preds), np.array(cov)



class SpaceAnimation3D:
    
    """
    :predictions: matrix with the predictions of the states
    :measurements: dataframe with the measurements with noise
    :target_x: target x of the position
    :target_y: target y of the position 
    
    """
    def __init__(self, predictions, traj):

        self.fig =  plt.figure(figsize=(16,13))
        self.ax = Axes3D(self.fig)
        
        self.x_target, self.y_target, self.z_target = traj.get_measurements()
        Xr, Yr, Zr = traj.get_trajectory_position()
      

        self.x_pred = predictions[:,0]
        self.y_pred = predictions[:,1]
        self.z_pred = predictions[:,2]

        plot_planets(Xr,Yr,Zr, self.ax)
     
        
        self.spaceship_pred, = self.ax.plot([], [], [], lw=5, c="r", label="Predictions")
        self.measurements, = self.ax.plot([], [], [], lw=2, alpha=0.6, c="g", label="Measurements")
        
        max_range = np.array([self.x_pred.max()-self.x_pred.min(), self.y_pred.max()-self.y_pred.min(), self.z_pred.max()-self.z_pred.min()]).max() / 3.0
   
        mean_x = self.x_pred.mean()
        mean_y = self.y_pred.mean()
        mean_z = self.z_pred.mean()
        
        self.ax.set_xlim3d(mean_x - max_range, mean_x + max_range)
        self.ax.set_ylim3d(mean_y - max_range, mean_y + max_range)
        self.ax.set_zlim3d(mean_z - max_range, mean_z + max_range)

        self.ax.legend(loc='best',prop={'size':15})
    
    def init(self):
        
        self.spaceship_pred.set_data_3d([])
        self.measurements.set_data_3d([])
        return self.spaceship_pred, self.measurements,

    def animate(self,i):
        
        self.spaceship_pred.set_data_3d(self.x_pred[:i], self.y_pred[:i],self.z_pred[:i])
        self.measurements.set_data_3d(self.x_target[:i], self.y_target[:i], self.z_target[:i])
        
        return self.spaceship_pred,self.measurements,
    

    def save_and_visualize_animation(self, path):

        anim= FuncAnimation(fig=self.fig, func=self.animate, init_func=self.init, frames=len(self.x_pred),interval=50, blit=True)
        
    
        writer = PillowWriter(fps=25)  
        anim.save( path, writer=writer, dpi=90)
        plt.close()
    
        with open(path,'rb') as f:
            display(Image(data=f.read(), format='gif'))
