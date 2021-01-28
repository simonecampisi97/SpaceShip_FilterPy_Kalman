# Traking SpaceShip - FilterPy

*Kalman filtering* is an algorithm that takes a series of measurements over time, containing statistical noise, and produce an estimate that tends to be more accurate than a single measurement. This is very important, because the the sensors give us always noisy information or the environment could makes data collection difficult , and the predictions made using kalman filtering help or make a better estimate.  There are many applications of the kalman filtering, and in this case, it was used to track a spaceship in a simulated trip from the Earth to the Moon.

The trajectory from the Earth to the Moon is a simply "toy" trajectory generated synthetically. Has been generated all the coordinates, velocty and acceleration for a moving object in 3 dimentional space. Then has been added noise with a Gaussian distribuition, in order to simulate a more realistic sensor that provides measurements. This is the trajectory plotted in 3D and in 2D (only x and z axis):

<div class="row" style= "display: table;">
  <div class="column">
    <img src="SpaceShip_3D\Plots\Kalman_Filter_Estimate_2D___Sigma_0.5.png" width="50%" style="float: left;"></img>
  </div>
  <div class="column">
    <img src="SpaceShip_3D\Plots\Kalman_Filter_Estimate___Sigma_0.5.png"width="50%" style="float: left;" ></img>
  </div>
</div>


Kalman Prediction 2D      |  Kalman Prediction 3D 
:-------------------------:|:-------------------------:
![](https://github.com/simocampi/SpaceShip_FilterPy_Kalman/blob/master/SpaceShip_3D/Plots/Kalman_Filter_Estimate_2D___Sigma_0.5.png)  |  ![](https://github.com/simocampi/SpaceShip_FilterPy_Kalman/blob/master/SpaceShip_3D/Plots/Kalman_Filter_Estimate___Sigma_0.5.png)

The Kalman algorithm is the one implemented in the <a  href="https://filterpy.readthedocs.io/en/latest/"> FilterPy module</a>, and all the implementation choices are described in the <a href="https://github.com/simocampi/Project-SpaceShip_FilterPy_Kalman/blob/master/REPORT.pdf"> report </a>


The following animation shows how the Kalman filtering algorithm works: Takes noisy masurements and then makes an estimate that is better than the measurements.

<img src="SpaceShip_3D\Animations\animaion_3d_prediction.gif" style=" margin-left: auto; margin-right: auto;" ></img>
 
