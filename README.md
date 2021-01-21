# Traking SpaceShip - FilterPy_Kalman

## 1 - Introduction

### 1.1 - Kalman Filter - Overview

 $$
 \textbf{Kalman filtering} is an algorithm that takes a series of measurements over time, containing statistical noise, and produce an estimate that tends to be more accurate than a single measurement. This is very important, because the the sensors give us always noisy information or the environment could makes data collection difficult, and the predictions made using kalman filtering help or make a better estimate.  There are many applications of the kalman filtering, and in this case, it was used to track a spaceship in a simulated trip from the Earth to the Moon.
$$


 \textbf{Kalman filtering} is an algorithm that takes a series of measurements over time, containing statistical noise, and produce an estimate that tends to be more accurate than a single measurement. This is very important, because the the sensors give us always noisy information or the environment could makes data collection difficult, and the predictions made using kalman filtering help or make a better estimate.  There are many applications of the kalman filtering, and in this case, it was used to track a spaceship in a simulated trip from the Earth to the Moon.

\subsection{Kalman Filter - Algorithm}

The Kalman filter for tracking moving objects estimates a state vector, $s_k$ comprising the parameters of the target, such as position, velocity and acceleration, based on a dynamic/measurement mode.

The algorithm of Kalman filtering  consists of two phases: \textit{update} and \textit{estimation}. It was assumed that the upper "-" on the vectors and matrices denotes: "before the acquisition of the k-th measure".

\begin{enumerate}
  
  \item \textbf{Update: } First, it is necessary to acquire the new measurement $ m_k $, which is modeled as $$m_k = Hs_k + r_k$$ 
  where \textbf{H} is the \textit{ M x N measurement matrix}, and $ \boldsymbol{r_k}$ is \textit{the measurement noise} with the \textit{M x M   measurement noise covariance matrix} \textbf{R}.
  Then, it is possible to compute the \textbf{Kalman Gain}, $\boldsymbol{K_k}$ that minimizes the error of the estimates: $$K_k = P^-_k H^T(HP_k^- H^T + R )^{-1}$$.
  Now it is possible update the state $\hat{s_k}$: $$ \hat{s_k} = \hat{s_k}^- + K_k(m_k - H\hat{s_k}^-) $$ and the and the covariance $P_k$ $$ P_k = (I - K_kH)P_k^-$$
  In which \textbf{P} is the \textit{ N x N Posterior covariance matrix}.
 
  \item \textbf{Estimation: } Now it is possible to make the two estimations: the state estimate projection $$ \hat{s}^-_{k+1} = \Phi \hat{s_k}$$ and the covariance estimate projection $$\hat{P}^-_{k+1} = \Phi P_k \Phi^T + Q$$
  where  $\boldsymbol{\Phi}$ is the \textit{transition matrix}, which allows to pass from the state k to the state k+1, and \textbf{Q} is the \textit{covariance matrix of the process noise}, that allows some variability in the model.
\end{enumerate}

\cite{notes}

\subsection{Data}

In order to build a simulation of a trajectory from the Earth to the Moon, has been generated synthetic data to build "toy" trajectory from the Earth to the Moon.
The Kalman filter model is a \textit{constant acceleration (CA)} model, in fact, it is assumed that the spaceship has a uniformly accelerated motion. But, in order to build a trajectory, that it may remotely resemble to a trip from the Earth to the Moon, it was necessary to add the acceleration and other factors to generate the data.
The data are generated in a 3-dimensional space, so, at the end of the generation, the data are composed by the three coordinates of the position (x, y, z), the velocities of the spaceship in the 3 directions (vx, vy, vz) and also the acceleration in the 3 directions (ax, ay, az).
Then, in order to simulate measurements obtained by a sensor, it was necessary add noise to the data.
The Fig \ref{fig:trajectory} shows the results of the data generation. As mentioned before, the measurements simulate a sensor that provides measures with uncertainty, and this measures are represented with a scatter plot.

\begin{figure}[!htb]
    \centering
    \includegraphics[width=\textwidth]{Figures/Measurements_Trajectory.png}% here goes the figure name
    \caption{Noisy Measurements (on the left) and real trajectory (on the right)}
    \label{fig:trajectory}
\end{figure}


\section{Methods - Initialization and Execution} \label{methods}

Now it's possible to model the kalman filter in order to track the spaceship from the Earth to the moon.
The Kalman Filter algorithm has not been implemented from scratch, but has been used the implementation of the algorithm provided by the module \textit{filterPy} \cite{filterpy}.
First it was necessary to initialize  the matrices $\Phi$, H, P, R, Q and to define the state and, as mentioned before, it was assumed that the model is uniformly accelerated. The dynamics of a moving object in one-dimension can be described by the Newtonian distance equation:

\begin{equation}
    x_t = \frac{1}{2}\Ddot{x} \Delta t^2 + \dot{x}_{t-1} \Delta t + x_{t-1}
\end{equation}

\begin{equation}
    \dot{x}_t = \Ddot{x} \Delta t + \dot{x}_{t-1}
\end{equation}

and since the motion is uniformly accelerated:

\begin{equation}
    \Ddot{x}_t = \ddot{x}_{t-1}
\end{equation}

In which $ x_t, \dot{x}_t, \Ddot{x}_t$ denote respectively the position, the velocity and the acceleration at time t.So, the dynamics of a moving object in one-dimension can be modeled by the position and the ﬁrst derivation of it. Now, this notation can extended for a 3D object, that is described by   $ x_t, y_t, z_t $ for the position, $\dot{x}_t, \dot{y}_t, \dot{z_t} $, for the velocity, and $\Ddot{x}_t, \Ddot{y}_t, \Ddot{z}_t$ for the acceleration.
So, for the 3D object, the state vector
\begin{equation}
s_t = \begin{bmatrix}
             x_t & y_t & z_t & \dot{x}_t & \dot{y}_t & \dot{z}_t & \ddot{x}_t & \ddot{y}_t &\ddot{z}_t
         \end{bmatrix}^T
\end{equation}
which, at start has been initialized with the starting position, velocity and acceleration.

Hence, with this assumptions the transition matrix $\Phi$ can be defined as the following 9x9 matrix:

\begin{equation}
    \Phi =  \begin{bmatrix}
        1 & 0 & 0 & \Delta t & 0 & 0 & \frac{1}{2} \Delta t^2 & 0 & 0\\
        0 & 1 & 0 &    0  & \Delta t & 0 & 0 & \frac{1}{2} \Delta t^2 & 0\\
        0 & 0 & 1 &    0  &   0 & \Delta t & 0 & 0 & \frac{1}{2} \Delta t^2\\
        0 & 0 & 0 &    1  &   0 &     0 & \Delta t  & 0 & 0\\
        0 & 0 & 0 &    0  &   1 &     0 &  0  & \Delta t &  0\\
        0 & 0 & 0 &    0  &   0 &     1 &  0 & 0 &  \Delta t\\
        0 & 0 & 0 &    0  &   0 &      0 &   1 & 0 & 0\\
        0 & 0 & 0 &    0  &   0 &      0 &   0 & 1 & 0\\
        0 & 0 & 0 &    0  &   0 &      0 &   0 & 0 & 1
    \end{bmatrix}
\end{equation}

Since the aim of this project is to track the position of the spaceship, x, y, and z, the measurement matrix H, 9 x 3, will be:

\begin{equation}
    H = \begin{bmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 
    \end{bmatrix}
\end{equation}

Then the process noise Q must to be set in order o achieve tracking errors that are as small as possible. In the conventional tracking systems, the most commonly process noise matrix used is the \textit{random acceleration (RA) process noise}, which is often selected because it has a better performance \cite{kalman_filter_object_tracking}. The 9x9 random acceleration process noise matrix is defined as following:

\begin{equation}
   Q = GG^T \sigma^2_a
\end{equation}

in which $\sigma^2_a$ is the \textit{acceleration process noise}, and G, for a constant acceleration model is:

\begin{equation}
    G = \begin{bmatrix}
        \frac{1}{2} \Delta t & \frac{1}{2} \Delta t & \frac{1}{2} \Delta t & \Delta t & \Delta t & \Delta t & 1 & 1 & 1 
        \end{bmatrix}
\end{equation}
 
Finally, the matrix P can be set to random values, instead R, can be empirically carried out, in this case is a diagonal matrix 3 x 9, with all ones in the diagonal.
All the above matrices are used to initialize an object \textit{KalmanFilter()} of filterpy, which represents the tracker for the spaceship, and provides two fundamental methods:

\begin{itemize}
    
    \item \textit{predict()}, which Predict next state
    \item \textit{update(z)}, which add a new measurement, z, to the Kalman filter.

\end{itemize}

This two methods are recalled in a loop for each measurements, and, at each iteration are saved the predicted state ($\hat{s_t}$), and the predicted covariance matrix (P). At the end of the loop, are returned a matrix N x M, in which N is the total number of measures and M is the size of the state vector, that contains all the state vector for each measure acquired. Moreover, is returned also a matrix N x M x M, which contains all the covariance matrix P for each measure acquired.


\section{Results}

Initializing the matrices and iterating as mentioned in the section \ref{methods}, the results obtained are showed in figure \ref{fig:Kalman_esimatate_1}. As you can notice, the filter takes noisy measures ( the small circles in Fig. \ref{fig:kalman_est_3d} and Fig. \ref{fig:kalman_est_2d}) and makes an estimation that is better than the measurements, allowing the spaceship to travel with the correct trajectory from the Earth to the Moon, and the estimated trajectory is very close to the original one without noise.

Then, in order to evaluate the quality of the predictions, has been computed the residuals, that is the difference between the positions measured and the position estimated by the filter. The results of the residual are showed in figure \ref{fig:Residuals}.As you can notice the oscillations of the residuals are in a range close to 0. In Fact the 
variance, $\sigma^2$ and the mean, $\mu$, of the residuals, are very close to 0, especially for the x and y position. Since the "ideal" residual should be as much as possible close to 0, is possible to say that the model has produced a good prediction of the trajectory.

\begin{figure}[!ht]
     \centering
     \begin{subfigure}[b]{\textwidth}
         \centering
         \includegraphics[width=0.8\textwidth]{Figures/Kalman_Filter_Estimate___Sigma_0.5.png}
         \caption{Kalman Filter Estimate - 3D}
         \label{fig:kalman_est_3d}
     \end{subfigure}
     \hfill
     \begin{subfigure}[b]{\textwidth}
         \centering
         \includegraphics[width=0.6\textwidth]{Figures/Kalman_Filter_Estimate_2D___Sigma_0.5.png}
         \caption{Kalman Filter Estimate 2D - Visualization of axis \textit{x} and \textit{z}}
         \label{fig:kalman_est_2d}
     \end{subfigure}
    
    \caption{Kalman Filter Estimate 3D - 2D }
    \label{fig:Kalman_esimatate_1}
\end{figure}




\begin{figure}[!ht]
     \centering
     \begin{subfigure}[b]{\textwidth}
         \centering
         \includegraphics[width=0.6\textwidth]{Figures/x_residual.png}
         \caption{Residual between \textit{x} positions measured and estimate -  $\mu = 0.420$, $\sigma^2 = 0.071$}
         \label{fig:Residual_x}
     \end{subfigure}
     \hfill
     \begin{subfigure}[b]{\textwidth}
         \centering
         \includegraphics[width=0.6\textwidth]{Figures/y_residual.png}
         \caption{Residual between \textit{y} positions measured and estimate  -  $\mu = - 0.603$, $\sigma^2 = 0.092$}
         \label{fig:Residual_y}
     \end{subfigure}
     \begin{subfigure}[b]{\textwidth}
         \centering
         \includegraphics[width=0.6\textwidth]{Figures/y_residual.png}
         \caption{Residual between \textit{z} positions measured and estimate - $\mu = 0.305$, $\sigma^2 = 0.059$}
         \label{fig:Residual_z}
     \end{subfigure}
    
    \caption{Residuals for the positions \textit{x},\textit{y}, \textit{z}  }
    \label{fig:Residuals}
\end{figure}


\subsection{Experimental Results - Signal Loss Simulation}

The tracker modeled has been tested, in order to show the behavior of the tracker with different kind of loss
signal.

\subsubsection{Filter updated with only the first half of the measurements}\label{exp1}
This first experiment shows the worst case: the spaceship at half of the trip looses totally the signal. In fact in the loop explained in section \ref{methods}, the tracker is updated with only the first half of the measurements. The Results are showed in figure \ref{fig:half}. As you can notice, at half trip, the spaceship deviates totally from route and fails to reach the moon.

\begin{figure}[!ht]
     \centering
     \begin{subfigure}[b]{\textwidth}
         \centering
         \includegraphics[width=0.8\textwidth]{Figures/Kalman_Filter_Estimate___Sigma_0.5__First_Half_of_Measurements.png}
         \caption{Kalman Filter Estimate - 3D}
         \label{fig:half_3d}
     \end{subfigure}
     \hfill
     \begin{subfigure}[b]{\textwidth}
         \centering
         \includegraphics[width=0.6\textwidth]{Figures/Kalman_Filter_Estimate_2D___Sigma_0.5__First_Half_of_Measurements.png}
         \caption{Kalman Filter Estimate 2D - Visualization of axis \textit{x} and \textit{z}}
         \label{fig:half_2d}
     \end{subfigure}
    
    \caption{Kalman Filter Updated with only the first half of measurements}
    \label{fig:half}
\end{figure}



\subsubsection{Filter updated with only the even indices of the measurements}
In this case, the situation is different from the experiment \ref{exp1}, in fact, updating the tracker with only the even indices of the measurements, the spaceship is able to reach correctly the Moon with a trajectory very similar to the one predicted with all the measurements. The figure \ref{fig:even} shows the obtained results.



\begin{figure}[!ht]
     \centering
     \begin{subfigure}[b]{\textwidth}
         \centering
         \includegraphics[width=0.8\textwidth]{Figures/Kalman_Filter_Estimate___Sigma_0.5__Even_Index_of_Measurements.png}
         \caption{Kalman Filter Estimate - 3D}
         \label{fig:even_3d}
     \end{subfigure}
     \hfill
     \begin{subfigure}[b]{\textwidth}
         \centering
         \includegraphics[width=0.6\textwidth]{Figures/Kalman_Filter_Estimate_2D___Sigma_0.5__Even_Index_of_Measurements.png}
         \caption{Kalman Filter Estimate 2D - Visualization of axis \textit{x} and \textit{z}}
         \label{fig:even_2d}
     \end{subfigure}
    
    \caption{Kalman Filter Updated with only the even indices of measurements}
    \label{fig:even}
\end{figure}


\subsubsection{Filter updated with only one measurement every 5}

Here the filter is updated with a measurement every five. Also in this case, the spaceship is able to reach correctly the Moon following a trajectory very similar to the one estimated with all the measurements, as showed in figure \ref{fig:five}.

\begin{figure}[!ht]
     \centering
     \begin{subfigure}[b]{\textwidth}
         \centering
         \includegraphics[width=0.8\textwidth]{Figures/Kalman_Filter_Estimate___Sigma_0.5__One_Measurements_Every_5_.png}
         \caption{Kalman Filter Estimate - 3D}
         \label{fig:five_3d}
     \end{subfigure}
     \hfill
     \begin{subfigure}[b]{\textwidth}
         \centering
         \includegraphics[width=0.6\textwidth]{Figures/Kalman_Filter_Estimate_2D___Sigma_0.5__One_Measurements_Every_5.png}
         \caption{Kalman Filter Estimate 2D - Visualization of axis \textit{x} and \textit{z}}
         \label{fig:five_2d}
     \end{subfigure}
    
    \caption{Kalman Filter updated with only one measurement every 5}
    \label{fig:five}
\end{figure}



\subsubsection{Spaceship looses the signal halfway for a certain period}

Here has been simulated a situation in which the spaceship looses the signal for a certain period at about half trip. In this case, as showed in figure \ref{fig:hole}, the spaceship deviates a bit from the route, but then, when the tracker acquires again the measures, the spaceship return in the original route and reaches the Moon.


\begin{figure}[!ht]
     \centering
     \begin{subfigure}[b]{\textwidth}
         \centering
         \includegraphics[width=0.8\textwidth]{Figures/Kalman_Filter_Estimate___Sigma_0.5__Hole_At_Half_Trip.png}
         \caption{Kalman Filter Estimate - 3D}
         \label{fig:hole_3d}
     \end{subfigure}
     \hfill
     \begin{subfigure}[b]{\textwidth}
         \centering
\includegraphics[width=0.6\textwidth]{Figures/Kalman_Filter_Estimate_2D___Sigma_0.5__Hole_At_Half_Trip.png}
         \caption{Kalman Filter Estimate 2D - Visualization of axis \textit{x} and \textit{z}}
         \label{fig:hole_2d}
     \end{subfigure}
    
    \caption{Spaceship looses the signal halfway for a certain period}
    \label{fig:hole}
\end{figure}


\clearpage %add new page for references
\printbibliography

\end{document}
