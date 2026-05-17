# Neural Operators for Hyperbolic PDEs

This repository develops neural-operator and graph-based architectures for approximating solutions of nonlinear hyperbolic PDEs with shock-forming dynamics.

The main test case is the LWR traffic equation, a scalar conservation law whose solutions can develop discontinuities. The project compares learned models against classical numerical solvers such as WENO-5.

## Highlights

- Space-time neural operator architecture for hyperbolic PDEs
- Physics-aware neighborhoods and message passing
- Evaluation on LWR equations with shock-inducing discontinuities
- Benchmarks against WENO-5
- Metrics: MSE, MAE, relative L2, shock-position error

## Why this is difficult

Hyperbolic PDEs are challenging for neural operators because small errors in transport speed can cause large spatial misalignment, especially near shocks and discontinuities. This project explores whether graph-based space-time message passing can better capture local propagation and discontinuous solution structure.
