import numpy as np
import matplotlib.pyplot as plt

def plot_residuals(traj, col, cov, preds, kind_of_residual="", stds=1.0):
    
    Zx, Zy, Zz = traj.get_measurements()
    z = np.asarray([ Zx, Zy, Zz]).T
    
    residual = z[:,col] - preds[:,col]
    
    plt.plot(residual)
    plt.xlabel('t')
    plt.ylabel(r'$z - \hat{z}$')
    plt.title(r"Residuals for " + kind_of_residual+ 
                    ": $\mu = {:2.3f}$ $\sigma^2 = {:2.3f}$".format(np.mean(residual), 
                                                             np.var(residual)), fontsize=15)