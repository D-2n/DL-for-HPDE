# Coding-Agent Instructions: Visual Examples for 2D LWR Shocks and Rarefactions

## Goal

Write a standalone Python script that generates visual examples of shock and rarefaction formation for a 2D scalar LWR-type conservation law on

\[
(x,y)\in[-1,1]\times[-1,1],
\qquad
t\in[0,1].
\]

Use the scalar conservation law

\[
\rho_t + \partial_x f(\rho) + \partial_y g(\rho)=0,
\]

with

\[
f(\rho)=\rho(1-\rho),
\qquad
g(\rho)=\beta \rho(1-\rho),
\qquad
\beta=0.7.
\]

The flux vector is

\[
\mathbf F(\rho)=
\begin{pmatrix}
\rho(1-\rho)\\
0.7\rho(1-\rho)
\end{pmatrix}.
\]

The density should satisfy

\[
\rho\in[0,1].
\]

---

## Numerical solver

Implement a 2D finite-volume solver with an unsplit conservative update:

\[
\rho_{i,j}^{n+1}
=
\rho_{i,j}^{n}
-
\frac{\Delta t}{\Delta x}
\left(
F_{i+1/2,j}-F_{i-1/2,j}
\right)
-
\frac{\Delta t}{\Delta y}
\left(
G_{i,j+1/2}-G_{i,j-1/2}
\right).
\]

Use Rusanov / local Lax--Friedrichs numerical fluxes.

### x-faces

\[
\hat F(\rho_L,\rho_R)
=
\frac{1}{2}\left[f(\rho_L)+f(\rho_R)\right]
-
\frac{1}{2}a_x(\rho_R-\rho_L),
\]

where

\[
a_x=\max(|f'(\rho_L)|,|f'(\rho_R)|),
\qquad
f'(\rho)=1-2\rho.
\]

### y-faces

\[
\hat G(\rho_B,\rho_T)
=
\frac{1}{2}\left[g(\rho_B)+g(\rho_T)\right]
-
\frac{1}{2}a_y(\rho_T-\rho_B),
\]

where

\[
a_y=\max(|g'(\rho_B)|,|g'(\rho_T)|),
\qquad
g'(\rho)=\beta(1-2\rho).
\]

Use zero-gradient / outflow boundary conditions by padding with edge values.

Use:

```python
N = 256
CFL = 0.45
T = 1.0
beta = 0.7
```

Set the time step by

\[
\Delta t
=
\mathrm{CFL}
\left(
\frac{1}{\Delta x}+\frac{\beta}{\Delta y}
\right)^{-1}.
\]

Clip the solution lightly to \([0,1]\) after updates to avoid tiny numerical overshoots.

---

## Initial conditions to generate

Generate at least these six examples.

---

### 1. Planar shock in x-direction

Use

\[
\rho_L=0.1,
\qquad
\rho_R=0.6,
\qquad
\rho(x,y,0)=
\begin{cases}
\rho_L, & x<0,\\
\rho_R, & x>0.
\end{cases}
\]

For concave LWR flux, since

\[
\rho_L<\rho_R,
\]

this is a shock.

Overlay the theoretical shock line

\[
x=s t,
\qquad
s=1-\rho_L-\rho_R.
\]

Here

\[
s=0.3.
\]

---

### 2. Planar rarefaction in x-direction

Use

\[
\rho_L=0.8,
\qquad
\rho_R=0.2,
\qquad
\rho(x,y,0)=
\begin{cases}
\rho_L, & x<0,\\
\rho_R, & x>0.
\end{cases}
\]

For concave LWR flux,

\[
\rho_L>\rho_R
\]

gives a rarefaction.

Overlay the rarefaction fan boundaries:

\[
x=\lambda(\rho_L)t,
\qquad
x=\lambda(\rho_R)t,
\]

where

\[
\lambda(\rho)=1-2\rho.
\]

So the fan boundaries are

\[
x=(-0.6)t
\]

and

\[
x=(0.6)t.
\]

---

### 3. Oblique planar shock

Let

\[
n=\frac{1}{\sqrt{2}}(1,1),
\qquad
\xi=n_x x+n_y y.
\]

Use

\[
\rho_L=0.1,
\qquad
\rho_R=0.6,
\]

and

\[
\rho(x,y,0)=
\begin{cases}
\rho_L, & \xi<0,\\
\rho_R, & \xi>0.
\end{cases}
\]

The effective flux in the normal direction is

\[
H(\rho)=\mathbf F(\rho)\cdot n
=
(n_x+\beta n_y)\rho(1-\rho).
\]

The shock normal speed is

\[
s_n=(n_x+\beta n_y)(1-\rho_L-\rho_R).
\]

Overlay the theoretical line

\[
n_xx+n_yy=s_n t.
\]

---

### 4. Oblique planar rarefaction

Use the same normal direction

\[
n=\frac{1}{\sqrt{2}}(1,1),
\]

but set

\[
\rho_L=0.8,
\qquad
\rho_R=0.2.
\]

Again,

\[
\rho(x,y,0)=
\begin{cases}
\rho_L, & n\cdot(x,y)<0,\\
\rho_R, & n\cdot(x,y)>0.
\end{cases}
\]

Overlay the two fan boundaries

\[
n\cdot(x,y)=\lambda_n(\rho_L)t,
\]

\[
n\cdot(x,y)=\lambda_n(\rho_R)t,
\]

where

\[
\lambda_n(\rho)=(n_x+\beta n_y)(1-2\rho).
\]

---

### 5. Curved discontinuity / circular patch

Use a circular initial condition:

\[
r=\sqrt{(x+0.25)^2+y^2}.
\]

Set

\[
\rho(x,y,0)=
\begin{cases}
0.75, & r<0.35,\\
0.20, & r\ge 0.35.
\end{cases}
\]

This should show that in 2D shocks are not just lines. A discontinuity is a curve in physical space, and as it evolves, it bends and deforms.

Do not overlay theory here; just visualize the numerical evolution.

---

### 6. Quadrant Riemann problem / wave interaction

Use four constant states:

\[
\rho(x,y,0)=
\begin{cases}
0.8, & x<0,\; y<0,\\
0.2, & x>0,\; y<0,\\
0.6, & x<0,\; y>0,\\
0.1, & x>0,\; y>0.
\end{cases}
\]

This should create interacting shocks and rarefactions near the origin.

---

## Plots to produce

For each example, produce a figure with snapshots at

\[
t=0,\quad 0.25,\quad 0.5,\quad 0.75,\quad 1.0.
\]

Use a single row of five panels per case.

Each panel should show \(\rho(x,y,t)\) using:

```python
imshow(
    rho,
    origin="lower",
    extent=[-1, 1, -1, 1],
    vmin=0,
    vmax=1,
    interpolation="nearest",
)
```

Use `interpolation="nearest"` so that the grid is honest and not artificially smoothed.

Add a colorbar for density \(\rho\).

For planar shock/rarefaction examples, overlay the theoretical shock line or fan boundaries using dashed black lines.

Save figures to:

```text
outputs/2d_lwr_planar_shock.png
outputs/2d_lwr_planar_rarefaction.png
outputs/2d_lwr_oblique_shock.png
outputs/2d_lwr_oblique_rarefaction.png
outputs/2d_lwr_circular_patch.png
outputs/2d_lwr_quadrant_riemann.png
```

Also produce one combined summary figure:

```text
outputs/2d_lwr_summary.png
```

with six rows, one per initial condition, and five columns for the time snapshots.

---

## Extra diagnostic plot

For each case, also compute

\[
|\nabla \rho|
\]

using centered finite differences and plot it at \(t=1\). This highlights shock curves.

Save:

```text
outputs/2d_lwr_gradients.png
```

This figure should show six panels, one per case, with the gradient magnitude at final time.

Use this to visually emphasize:

\[
\text{2D shock} = \text{moving curve in }(x,y),
\]

while rarefactions appear as broader smooth transition regions.

---

## Script structure

Please structure the script as:

```python
def flux_x(rho):
    ...

def flux_y(rho, beta=0.7):
    ...

def rusanov_flux_x(left, right):
    ...

def rusanov_flux_y(bottom, top):
    ...

def step(rho, dx, dy, dt, beta):
    ...

def solve(rho0, T, dx, dy, beta, cfl, snapshot_times):
    ...

def make_initial_condition(case_name, X, Y):
    ...

def plot_case(case_name, snapshots, times, X, Y):
    ...

def plot_summary(all_results):
    ...

def plot_gradient_diagnostics(all_results):
    ...

if __name__ == "__main__":
    ...
```

Use only:

```python
numpy
matplotlib
os
```

Optionally use `imageio` if making GIFs, but static PNGs are enough.

---

## Important comments to include in the script

Include short comments explaining:

1. In 2D, a shock at fixed time is a curve in physical space.
2. In space-time \((x,y,t)\), that moving curve sweeps out a surface.
3. For planar initial data, the 2D problem reduces to a 1D Riemann problem in the normal coordinate

\[
\xi=n_xx+n_yy.
\]

4. The normal shock speed is

\[
s_n=
\frac{
(\mathbf F(\rho_R)-\mathbf F(\rho_L))\cdot n
}{
\rho_R-\rho_L
}.
\]

5. For LWR flux, concavity means:

\[
\rho_L<\rho_R
\Rightarrow
\text{shock},
\]

\[
\rho_L>\rho_R
\Rightarrow
\text{rarefaction}.
\]

---

## Expected interpretation

The generated figures should make visually clear:

- A 1D shock becomes a moving **curve** in 2D physical space.
- A planar shock is a straight line, but general shocks can be curved.
- A rarefaction is not a sharp curve; it is a spreading smooth transition region.
- Planar 2D waves are basically 1D waves viewed along a chosen normal direction.
- More general 2D initial data creates curved and interacting wave patterns.
