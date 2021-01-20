# Project-SpaceShip_FilterPy_Kalman

\documentclass{Academic}

%References 
\RequirePackage[backend=bibtex]{biblatex}

%Add reference files!!!
\addbibresource{references.bib}



\begin{document}
%Easy customisation of title page
%TC:ignore
\myabstract{\include{abstract}}
\renewcommand{\myTitle}{Spaceship Tracking - Kalman Filter}
\renewcommand{\MyAuthor}{Simone Campisi}
\renewcommand{\MyDepartment}{Genoa University, Master in Artificial Intelligence}
\renewcommand{\ID}{4341240}
\maketitle

%\vspace{-1.9em}\noindent\rule{\textwidth}{1pt} %add this line if not using abstract
\onehalfspacing



\section{Introduction}

\subsection{Kalman Filter - Overview}

 \textbf{Kalman filtering} is an algorithm that takes a series of measurements over time, containing statistical noise, and produce an estimate that tends to be more accurate than a single measurement. This is very important, because the the sensors give us noisy information or the environment could makes data collection difficult, and the predictions made using kalman filtering help or make a better estimate.  There are many application of the kalman filtering, and in this case, it was used to track a simulated trip of a spaceship from the Earth to the Moon.

\subsection{Kalman Filter - Algorithm}

The Kalman filter for tracking moving objects estimates a state vector, $s_k$ comprising the parameters of the target, such as position, velocity and acceleration, based on a dynamic/measurement mode.

The algorithm of Kalman filtering  consists of two phases: \textbf{update} and \textbf{estimation}. It was assumed that the upper "-" on the vectors and matrices denotes: "before the acquisition of the k-th measure".

\begin{enumerate}
  
  \item \textbf{Update: } First, it is necessary to acquire the new measure $ \boldsymbol{m_k} $, which is modeled as $$m_k = Hs_k + r_k$$ 
  where \textbf{H} is the \textit{ M x N measurement matrix}, and $ \boldsymbol{r_k}$ is \textit{the measurement noise} with the \textit{M x M   measurement noise covariance matrix} \textbf{R}.
  Then, it is possible to compute the \textbf{Kalman Gain}, $\boldsymbol{K_k}$ that minimizes the error of the estimates: $$K_k = P^-_k H^T(HP_k^- H^T + R )^{-1}$$.
  Now it is possible update the state $\hat{s_k}$: $$ \hat{s_k} = \hat{s_k}^- + K_k(m_k - \hat{s_k}^-) $$ and the and the covariance $P_k$ $$ P_k = (I - K_kH)P_k^-$$
  In which \textbf{P} is the \textit{ N x N Posterior covariance matrix}.
  The state $\hat{s}_k $ is a weighted average between $\hat{s}^-_k$, that is the state estimate at time k before to acquire the k-th measurement, and the innovation $m_k - H\hat{s}_k$. The optimal weight, the Kalman gain marix, is obtained by minimising the trace of the covariance matrix.
  
  \item \textbf{Estimation: } Now it is possible o make the two estimations: the state estimate projection $$ \hat{s}^-_{k+1} = \Phi \hat{s_k}$$ and the covariance estimate projection $$\hat{P}^-_{k+1} = \Phi P_k \Phi^T + Q$$
  where  $\boldsymbol{\Phi}$ is the \textit{transition matrix}, which allows to pass from the state k to the state k+1, and \textbf{Q} is the \textit{covariance matrix of the process noise}, that allows some variability in the model.
\end{enumerate}

\subsection{Data}

In order to build a simulation of a trajectory from the Earth to the Moon, has been generated synthetic data, o build "toy" trajectory from the Earth to the Moon.
The kalman filter model is a \textit{constant acceleration (CA)} model, in fact, in order to model the kalman filter, has been assumed that the spaceship has a uniformly accelerated motion. But, in order to build a trajectory, that it may remotely resemble to a trip from the Earth to the Moon, it was necessary add the acceleration and other factors to generate the data.
The data are generated in a 3-dimensional space, so, at the end of the generation, the data are composed by the three coordinates of the position (x, y, z), the velocities of the spaceship in the 3 directions (vx, vy, vz) and also the acceleration in the 3 directions (ax, ay, az).
Then, in order to simulate measurements obtained by a sensor, it was necessary add noise to the data.
The Fig \ref{fig:trajectory} shows the results of the data generation. As mentioned before, the measurements simulate a sensor that provides measures with uncertainty, and this measures are represented with a scatter plot.

\begin{figure}[!htb]
    \centering
    \includegraphics[width=\textwidth]{Figures/Measurements_Trajectory.png}% here goes the figure name
    \caption{Noisy Measurements (on the left) and real trajectory (on the right)}
    \label{fig:trajectory}
\end{figure}


\section{Methods}

Now it's possible to model kalman filter in order to track the spaceship from the Earth to the moon.
The Kalman Filter algorithm has not been implemented from scratch, but has been used the algorithm provided by the module \textbf{filterpy}.
First it was necessary to initialize  the matrices $\Phi$, H, P, R, Q and to define the state and, as mentioned before, it was assumed that the model is uniformly accelerated. The dynamics of a moving object in one-dimension can be described as follows:

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

In which $ x, \dot{x}, \Ddot{x}$ denote respectively the position, the velocity and the acceleration at time t.So, the dynamics of a moving object in one-dimension can be modeled by the position and the ﬁrst derivation of it. Now, this notation can extended for a 3D object, that is described by   $ x, y, z $ for the position, $\dot{x}, \dot{y}, \dot{z} $, for the velocity, and $\Ddot{x}, \Ddot{y}, \Ddot{z}$ for the acceleration.
So, for the 3D object, the state vector
\begin{equation}
s_t = \begin{bmatrix}
             x & y & z & \dot{x} & \dot{y} & \dot{z} & \ddot{x} & \ddot{y} &\ddot{z}
         \end{bmatrix}^T
\end{equation}

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

Since the aim of this project is to track the position of the spaceship, x, y, and z, the measurement matrix 9 x 3, H will be:

\begin{equation}
    H = \begin{bmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 
    \end{bmatrix}
\end{equation}

Then the process noise Q must to be set in order o achieve tracking errors that are as small as possible. In the conventional tracking systems, the most commonly process noise matrix used is the \textit{random acceleration (RA)} process noise, which is often selected because it has a better performance. The 9x9 random acceleration process noise matrix is defined as following:

\begin{equation}
   Q = GG^T \sigma^2_a
\end{equation}

in which $\sigma^2_a$ is the noise of the acceleration, and G, for a constant acceleration, model is:

\begin{equation}
    G = \begin{bmatrix}
        \frac{1}{2} \Delta t & \frac{1}{2} \Delta t & \frac{1}{2} \Delta t & \Delta t & \Delta t & \Delta t & 1 & 1 & 1 
        \end{bmatrix}
\end{equation}
 
 Finally, the matrix P can be set to random values, instead R, can be empirically carried out.




\section{Results}

Graphs are presented with clear and appropriate axes, legends, labels and lines such that key evidence is precise and very convincing.  Bar charts are clear, precise and easy to understand.
Error bars are used consistently and appropriately with precise justification of how they were calculated.
Micrographs are of excellent quality with appropriate contrast and magnification, and clear scale bars.  Multiple magnifications and arrows highlighting key features are used to excellent effect.
Figure and table captions indicate all of the key conditions so that the methods used to obtain the data can be precisely traced to the Methods section, such that the data could be replicated with ease.
Each figure and table is precisely described to highlight the key features that provide evidence for the conclusions.
For modelling projects, clear and precise evidence is provided to show that numerical results are insensitive to input parameters through convergence tests.


\section{Discussion}

\begin{figure}[!htb]
    \centering
    \includegraphics[width=\textwidth]{example-image-a}% here goes the figure name
    \caption{This is the figure caption.}
    \label{fig:name_me_please}
\end{figure}

Figure \ref{fig:name_me_please}, is an example figure!
The discussion is carefully structured so that precise and robust evidence is provided to underpin each conclusion. This incorporates evidence from the results section (precisely referenced, highlighting key features) and evidence from previous work that was described in the introduction.  
A precise and well-structured critical comparison is made between the experimental/modelling results and previous work with a thorough exploration of all possible sources of uncertainty.
The significance of the findings is precisely described with reference to the question/hypothesis that has been addressed.


\section{Conclusion}

There is a brief and precise description of the context of the work such that it is easy to understand the significance of the conclusions.  
Each conclusion is described precisely, and correlates exactly with the evidence discussed in the discussion section.
The significance of the work is precisely described.
There is no new information that has not been discussed in the rest of the report.
The conclusions section correlates precisely with the abstract, and is easy to understand if taken out of context.

%TC:ignore
%\clearpage %add new page for references

\printbibliography %Prints bibliography

% \clearpage
% \begin{appendices}

% \section{Here go any appendices!}

% \end{appendices}

%TC:endignore
\end{document}
