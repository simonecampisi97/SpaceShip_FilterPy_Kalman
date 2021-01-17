import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch, HandlerCircleCollection

import os
from IPython.display import Image, display

from model_evaluation import plot_covariance_ellipsoide


from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.common import Saver


const_acceleration_x = 2
const_acceleration_y = 1
dt=0.001
t= np.arange(0, 1.01, dt)
N = len(t)
traj = (2*(t**5)- 1.5*(t**4) + 0.05*(t**3) - 3*(t**2)+3*t)

t= (t)*100
traj= (traj)*100

cov = []


def init_global(const_acc_x, const_acc_y,dt_, t_, N_, traj_):
    
    global const_acceleration_x, const_acceleration_y,dt, t, N, traj

    const_acceleration_x, const_acceleration_y = const_acc_x, const_acc_y
    dt, N, traj, t = dt_, N_, traj_, t_




def get_x_y_velocities():
    
    global const_acceleration_x, const_acceleration_y, dt, t, traj


    x_velocities = np.zeros(len(t))
    y_velocities = np.zeros(len(t))
    np.random.seed(25)
  
    for i in range(1,len(t)) :
        
        x_velocities[i] = ( t[i] - (t[i-1]+ (1/2)*const_acceleration_x*dt**2))/dt 
        y_velocities[i] = ( traj[i] - (traj[i-1]+ (1/2)*const_acceleration_y*dt**2))/dt
    
    return x_velocities, y_velocities, const_acceleration_x, const_acceleration_y


def plot_measurements(measurements,ax):
    
    x_moon, y_moon = measurements.x_pos[len(measurements.x_pos)-1],  measurements.y_pos[len(measurements.y_pos)-1]
    x_earth, y_earth = measurements.x_pos[0], measurements.y_pos[0]
    
    #ax.set( xlim=(-9, np.max(measurements.x_pos)+9), ylim=(-10, np.max(measurements.y_pos)+9 ) )

    plt.figure(figsize=(13,10))
    ax.plot(measurements.x_pos, measurements.y_pos, ls = "--",c='black', label = "Target Trajectoy")
    
    ax.set_title("Target Trajectory", fontsize=15)
    earth = plt.Circle(( x_earth, y_earth), 3, color='blue')
    moon  = plt.Circle((x_moon, y_moon ), 1.5, color='grey')
    ax.add_patch(earth)
    ax.add_patch(moon)

    #legend_trajectory = plt.Line2D([0], [0], ls='--', color="black")
    ax.text(-2,-4,"Earth", weight='bold', c="b", fontsize=10)
    ax.text(measurements.x_pos.to_list()[-1]-2, measurements.y_pos.to_list()[-1]+2,"Moon", weight='bold', c="gray", fontsize=10)
    ax.legend()


def plot_planets(measurements,ax):
    
    x_moon, y_moon = measurements.x_pos[len(measurements.x_pos)-1],  measurements.y_pos[len(measurements.y_pos)-1]
    x_earth, y_earth = measurements.x_pos[0], measurements.y_pos[0]
    
    earth = plt.Circle(( x_earth, y_earth), 3, color='blue', label = "Earth")
    moon  = plt.Circle((x_moon, y_moon ), 1.5, color='grey', label = "Moon")
    
    ax.add_patch(earth)
    ax.add_patch(moon)

    ax.text(-2,-4,"Earth", weight='bold', c="b", fontsize=10)
    ax.text(measurements.x_pos.to_list()[-1]-2, measurements.y_pos.to_list()[-1]+2,"Moon", weight='bold', c="gray", fontsize=10)



def plot_prediction(predictions, measurements, ax):

    plot_measurements(measurements,ax)
    ax.plot(predictions[:,0],predictions[:,1], c='r', label='Kalman Prediction')
    ax.legend()



def plot_residual_limits(Ps, stds=1.):
    """ plots standand deviation given in Ps as a yellow shaded region. One std
    by default, use stds for a different choice (e.g. stds=3 for 3 standard
    deviations.
    """

    std = np.sqrt(Ps) * stds

    plt.plot(-std, color='k', ls=':', lw=2)
    plt.plot(std, color='k', ls=':', lw=2)
    plt.fill_between(range(len(std)), -std, std,
                 facecolor='#ffff00', alpha=0.3)



def init_kalman(measurements, sigma=0.3):
    global cov, const_acceleration_x, const_acceleration_y
    #Transition_Matrix matrix
    PHI =   np.array([[1, 0, dt, 0, (dt**2)/2, 0],
                     [0, 1, 0, dt, 0, (dt**2)/2],
                     [0, 0, 1,  0, dt, 0,],
                     [0, 0, 0,  1, 0, dt],
                     [0, 0, 0,  0,  1 , 0],
                     [0, 0, 0,  0,  0 , 1] ])


    # Matrix Observation_Matrix
    #We are looking for the position of the spaceship
    H = np.array([[1,0,0,0,0,0],
                  [0,1,0,0,0,0]])


    #initial state
    init_states = np.array([measurements.x_pos[0], measurements.y_pos[0], 0, 0, const_acceleration_x, const_acceleration_y])

    P = np.eye(6)*(sigma**2)
    cov=P

    R = np.eye(2)* (0.001)

    #acc_noise = (0.1)**2
    
    """ 
    G = np.array([ [(dt**2)/2],
                    [(dt**2)/2],
                    [    dt   ],
                    [    dt   ],
                    [    1    ],
                    [    1    ]])
    """
    Q = Q_discrete_white_noise(2, dt=dt, var=10, block_size=3) 
    #Q= np.dot(G, G.T)*(0.3)**2

    return init_states, PHI, H, Q, P, R



def Ship_tracker(measurements,sigma=0.3):
    
    global dt

    init_states, PHI, H, Q, P,R = init_kalman(measurements,sigma)
    
    tracker= KalmanFilter(dim_x = 6, dim_z=2)
    tracker.x = init_states

    tracker.F = PHI
    tracker.H = H   # Measurement function
    tracker.P = P   # covariance matrix
    tracker.R = R  # state uncertainty
    
    tracker.Q =  Q # process uncertainty
    
    return tracker


def run(tracker, zs):
    
    preds, cov = [],[]
    
    for z in zs:
        tracker.predict()
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