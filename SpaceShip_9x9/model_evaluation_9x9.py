import numpy as np
import matplotlib.pyplot as plt
import math




#taken from filterpy documentation
def covariance_ellipse(P, deviations=1):
    """
    Returns a tuple defining the ellipse representing the 2 dimensional
    covariance matrix P.

    Parameters
    ----------

    P : nd.array shape (2,2)
       covariance matrix

    deviations : int (optional, default = 1)
       # of standard deviations. Default is 1.

    Returns (angle_radians, width_radius, height_radius)
    """
    U, s, _ = np.linalg.svd(P)
    orientation = math.atan2(U[1, 0], U[0, 0])
    width = deviations * math.sqrt(s[0])
    height = deviations * math.sqrt(s[1])

    if height > width:
        raise ValueError('width must be greater than height')

    return (orientation, width, height)


#taken from filterpy documentation
def _std_tuple_of(var=None, std=None, interval=None):
    """
    Convienence function for plotting. Given one of var, standard
    deviation, or interval, return the std. Any of the three can be an
    iterable list.

    Examples
    --------
    >>>_std_tuple_of(var=[1, 3, 9])
    (1, 2, 3)

    """
    if std is not None:
        if np.isscalar(std):
            std = (std,)
        return std

    if interval is not None:
        if np.isscalar(interval):
            interval = (interval,)

        return np.norm.interval(interval)[1]

    if var is None:
        raise ValueError("no inputs were provided")

    if np.isscalar(var):
        var = (var,)
    return np.sqrt(var)


#taken and modified from filterpy documentation
def plot_covariance_ellipsoide(
        mean, cov, ax, variance=1.0, std=None, interval=None,
        title=None, axis_equal=True,
        show_semiaxis=False, show_center=True,
        fc='none', ec='#004080',
        alpha=1.0,
        ls='solid', plot=True):

    from matplotlib.patches import Ellipse

    if axis_equal:
        ax.axis('equal')

    if title is not None:
        ax.set_title(title)

    ax = plt.gca()

    ellipse = covariance_ellipse(cov)
    angle = np.degrees(ellipse[0])
    width = ellipse[1] * 2.
    height = ellipse[2] * 2.

    std = _std_tuple_of(variance, std, interval)
    for sd in std:
        e = Ellipse(xy=mean, width=sd*width, height=sd*height, angle=angle,
                    facecolor=fc,
                    edgecolor=ec,
                    alpha=alpha,
                    lw=2, ls=ls)
        if plot == True:
            ax.add_patch(e)
    
   
    x, y = mean
    
    if show_center and plot:
        ax.scatter(x, y, marker='+', color=ec)

    if show_semiaxis and plot:
        a = ellipse[0]
        h, w = height/4, width/4
        ax.plot([x, x+ h*np.cos(a+np.pi/2)], [y, y + h*np.sin(a+np.pi/2)])
        ax.plot([x, x+ w*np.cos(a)], [y, y + w*np.sin(a)])
    
    if std is not None:
        return e, angle, width, height
    else:
        return None
    


def plot_residuals(t, measurements, measurements_col, col, predictions, kind_of_residual=""):
    
    residual = measurements[measurements_col] - predictions[:,col]
    
    plt.plot(t, residual)
    plt.xlim((t[0], t[-1]))
    plt.xlabel('t')
    plt.ylabel(r'$z - \hat{z}$')
    plt.title(r"Residuals for " + kind_of_residual+ ": $\mu = {:2.3f}$ $\sigma^2 = {:2.3f}$".format(np.mean(residual), np.var(residual)))