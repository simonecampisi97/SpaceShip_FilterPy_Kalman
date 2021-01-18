import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch, HandlerCircleCollection
import pandas as pd

import os
from IPython.display import Image, display

from model_evaluation_3D import plot_covariance_ellipsoide


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
        SIGMA = 0.5
        
        self.T =  T # measuremnt time
        self.m = int(self.T/self.dt) # number of measurements
        self.sigma = sigma

        self.px= 0.0 # x Position Start
        self.py= 0.0 # y Position Start
        self.pz= 1.0 # z Position Start

        self.vx = 100.0  # m/s Velocity at the beginning
        self.vy = 0.0 # m/s Velocity
        self.vz = 0.0 # m/s Velocity

        c = 0.1 # Drag Resistance Coefficient

        self.Xr=[]
        self.Yr=[]
        self.Zr=[]
        
        self.Vx=[]
        self.Vy=[]
        self.Vz=[]
        
        for i in range(int(self.m)):
            
            accx = -c*self.vx**2  # Drag Resistance
            
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
    
        
    def get_trajectory_position(self):
        return self.Xr, self.Yr, self.Zr

    
    def get_measurements(self):

        #adding Noise
        self.Xm = self.Xr + self.sigma * (np.random.randn(self.m))
        self.Ym = self.Yr + self.sigma * (np.random.randn(self.m))
        self.Zm = self.Zr + self.sigma * (np.random.randn(self.m)) 
        
        return self.Xm, self.Ym, self.Zm



def plot_measurements_3D(traj, ax, title=""):
    
    x,y,z = traj.get_measurements()
    
    
    ax.scatter(x, y, z, c='gray')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    #ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # Axis equal
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 3.0
   
    mean_x = x.mean()
    mean_y = y.mean()
    mean_z = z.mean()
    ax.set_xlim(mean_x - max_range, mean_x + max_range)
    ax.set_ylim(mean_y - max_range, mean_y + max_range)
    ax.set_zlim(mean_z - max_range, mean_z + max_range)



def plot_prediction(preds,traj, ax):
    global SIGMA

    xt, yt, zt = preds[:,0], preds[:,1], preds[:,2]
    Xr, Yr, Zr = traj.get_trajectory_position()
    Xm, Ym, Zm = traj.get_measurements()

    
    ax.plot(xt,yt,zt, lw=2, label='Kalman Filter Estimate')
    ax.plot(Xr, Yr, Zr, lw=2, label='Real Trajectory Without Noise')
    ax.scatter(Xm, Ym, Zm, edgecolor='g', facecolor='none', alpha=0.1, lw=2, label="Measurements")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title("Kalman Filter Estimate - Sigma={}".format(SIGMA), fontsize=15)


    # Axis equal
    max_range = np.array([Xm.max()-Xm.min(), Ym.max()-Ym.min(), Zm.max()-Zm.min()]).max() / 3.0
    mean_x = Xm.mean()
    mean_y = Ym.mean()
    mean_z = Zm.mean()
    ax.set_xlim(mean_x - max_range, mean_x + max_range)
    ax.set_ylim(mean_y - max_range, mean_y + max_range)
    ax.set_zlim(mean_z - max_range, mean_z + max_range)



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

    #initial state
    init_states = np.array([x[0], y[0], z[0], 100., 0., 0., 0., 0., -9.81])

    P = np.eye(9)*(SIGMA**2)
   
    rp = 1.0**2  # Noise of Position Measurement
    R = np.eye(3)* rp

    #acc_noise = (0.1)**2
    
    G = np.array([  [(DT**2)/2],
                    [(DT**2)/2],
                    [(DT**2)/2],
                    [    DT   ],
                    [    DT   ],
                    [    DT   ],
                    [    1    ],
                    [    1    ],
                    [    1    ]])

    #Q = Q_discrete_white_noise(2, dt=DT, var=10, block_size=3) 
    acc_noise = 0.1 # acceleration proccess noise
    Q= np.dot(G, G.T)* acc_noise

    return init_states, PHI, H, Q, P, R



def Ship_tracker(traj):
    
    global DT

    init_states, PHI, H, Q, P,R = init_kalman(traj)
    
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

def run_half_measures(tracker, zs):
    
    preds, cov = [],[]
    
    for i,z in enumerate(zs):
        tracker.predict()
        
        if i  <= len(zs)//2:
            tracker.update(z=z)
       
        preds.append(tracker.x)
        cov.append(tracker.P)
    
    return np.array(preds), np.array(cov)



def run_even_index_update(tracker, zs):
    
    preds, cov = [],[]
    
    for i, z in enumerate(zs):
        
        tracker.predict()
        
        if( i  % 2 == 0):
            tracker.update(z=z)
        
        preds.append(tracker.x)
        cov.append(tracker.P)
    
    return np.array(preds), np.array(cov)


def run_update_every_5(tracker, zs):
    
    preds, cov = [],[]
    
    for i, z in enumerate(zs):
        
        tracker.predict()
        
        if( i  % 5 == 0):
            tracker.update(z=z)
        
        preds.append(tracker.x)
        cov.append(tracker.P)
    
    return np.array(preds), np.array(cov)

def run_update_hole_in_middle(tracker, zs):
    
    preds, cov = [],[]
    
    chunk = len(zs) // 3
    for i, z in enumerate(zs):
        
        tracker.predict()
        if i <= chunk or i >= 2*chunk:
            tracker.update(z=z)
        
        preds.append(tracker.x)
        cov.append(tracker.P)
    
    return np.array(preds), np.array(cov)



class HandlerEllipse(HandlerPatch):
    
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        
        p = mpatches.Ellipse(xy=center, width=orig_handle.width,
                                        height=orig_handle.height)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


def plot_comparison_ellipse_covariance(measurements, preds, cov):

    fig_cov = plt.figure(figsize=(13,10))
    ax_cov =  fig_cov.add_subplot(1,1,1)
    ellipse_step = 50
    
    plot_planets(measurements, ax_cov)
    
    i=0
    for x, P, x_pos, y_pos in zip(preds, cov, measurements.x_pos, measurements.y_pos):
        mean, covariance = x[0:2], P[0:2,0:2]
        if i % ellipse_step == 0:
            e,_,_,_=plot_covariance_ellipsoide(mean=mean, ax=ax_cov, std=205, cov=covariance, fc='g', alpha=0.3, ls='dashed')
            scatter = ax_cov.scatter(x_pos, y_pos, edgecolor='k', facecolor='none', lw=2)
        i+=1
    
    ax_cov.plot(preds[:,0], preds[:,1], label='filter', c='b')

    legend_earth = plt.Line2D([0], [0], ls='None', color="blue", marker='o')
    legend_moon = plt.Line2D([0], [0], ls='None', color="grey", marker='o')
    legend_pred = plt.Line2D([0], [0], ls='None', color="b", marker='_')
    legend_ellipse = mpatches.Ellipse((), width=15, height=5, facecolor="g", alpha=0.3, ls='dashed')
    ax_cov.legend([legend_earth, legend_moon, legend_ellipse,legend_pred, scatter],["Earth","Moon","Ellipse","Kalamn Filter","Measurments"],
                    handler_map={mpatches.Ellipse: HandlerEllipse()}, loc='best')
    ax_cov.grid()
    ax_cov.set_title("Covariance Ellipsoide vs Measurments vs Kalman Filter")
    fig_cov.savefig(os.path.join("Plots","covariance_ellipsoide.png"), dpi=100)
    

class SpaceAnimation:
    
    """
    :predictions: matrix with the predictions of the states
    :measurements: dataframe with the measurements with noise
    :target_x: target x of the position
    :target_y: target y of the position 
    
    """
    def __init__(self, predictions, measurements):
        

        self.predictions= predictions

        self.x_target = measurements.x_pos.to_list()
        self.y_target = measurements.y_pos.to_list()

        self.x_pred = predictions[:,0]
        self.y_pred = predictions[:,1]

        self.fig =  plt.figure(figsize=(13,10))
        self.ax = self.fig.add_subplot(1,1,1)
        self.ax.set( xlim=(-9, np.max(self.x_pred)+4), ylim=(-9, np.max(self.y_pred)+4 ) )
        
        plot_planets(measurements, self.ax)
        
        self.spaceship_pred = plt.Circle((0., 0.), 2, fc='r')

        #create a patch for the target
        self.patch_width = 4.6
        self.patch_height = 4.6
        self.target = plt.Rectangle((0-(self.patch_width/2),0-(self.patch_height/2)), self.patch_width, self.patch_height,
                                            linewidth=2,edgecolor='green',facecolor='none') 
       
    
    def init(self):
        
        self.spaceship_pred.center = (0 - (self.patch_width/2), 0 - (self.patch_height/2) )
        self.target.set_xy( (0,0) )
        self.ax.add_patch(self.spaceship_pred)
        self.ax.add_patch(self.target)

       
        legend_pred = plt.Line2D([0], [0], ls='None', color="red", marker='o')
        
        self.ax.legend([legend_pred],["Prediction"])
    
        self.target_text =  self.ax.text(-2, 2,"", weight='bold', c="green", fontsize=10)
        return self.spaceship_pred, self.target,

    
    def animate(self,i):
        
        x, y = self.x_pred[i], self.y_pred[i]

        x_t, y_t = self.x_target[i], self.y_target[i]
        self.target_text.remove()
        self.target_text =  self.ax.text(x_t-3, y_t+3,"Target", weight='bold', c="green", fontsize=10)

        self.spaceship_pred.center=(x,y)
        self.target.set_xy( (x_t - self.patch_width/2, y_t - self.patch_height/2) )
        return self.spaceship_pred, self.target,
    

    def save_and_visualize_animation(self, path):

        anim= FuncAnimation(fig=self.fig, func=self.animate, 
        init_func=self.init,frames=len(self.x_target),interval=50, blit=True)
        
    
        writer = PillowWriter(fps=25)  
        anim.save( path, writer=writer, dpi=100)
        plt.close()
    
        with open(path,'rb') as f:
            display(Image(data=f.read(), format='gif'))