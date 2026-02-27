# Bubble dynamics

Consider the bubble dynamics in an incompressible fluid. It will change it radius over time. The radius of the bubble is given by the following ordinary differential equation (Raileigh-Plesset):

$$\frac{P_{\infty}(t)-P_{B}(t)}{\rho_L} = R\ddot{R} + \frac{3}{2}\dot{R}^2 + \frac{4\nu_L}{ R}\dot{R} + \frac{2\gamma}{\rho_L R},$$

where


- $P_{\infty}(t)$ is the external pressure at infinity (very far away from the bubble)
- $P_{B}(t)$ is the pressure inside the bubble
- $\rho_L$ is the density of the liquid
- $R$ is the radius of the bubble
- $\nu_L$ is the kinematic viscosity of the liquid
- $\gamma$ is the surface tension of the liquid

1. You have train dataset `res_1000_1.npz` and test dataset `res_1000_08.npz`.  The data is saved in the following format:
    `dataset['del_p']` is the pressure difference between the external pressure and the pressure inside the bubble
    `dataset['t']` is the time
    `dataset['R']` is the radius of the bubble.
Using these datasets and preprocessing functions availabe in `bubble.py`, implement and train a DeepONet to predict the radius of the bubble at given times, based on the pressure difference.

2. Play with different discretizations of time for both branch and trunk networks. Try both sparse and dense discretizations. Compare the results with the training of a simple RNN model on the same data.

3. Try Fourier Neural Operator (FNO) to solve the same problem. Compare the results with the DeepONet and RNN models.