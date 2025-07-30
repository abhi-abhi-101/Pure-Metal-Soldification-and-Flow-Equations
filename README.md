# Pure-Metal-Soldification-and-Flow-Equations

This project numerically solves the **2D lid-driven cavity flow** using the **finite difference method**, with **Staggered Grid Approach** for time integration and a **projection method** to enforce incompressibility through the pressure Poisson coupling  equation.

## Overview

The lid-driven cavity problem is a classic benchmark problem in computational fluid dynamics (CFD). It involves a square cavity where the top lid moves with a constant velocity, while all other walls remain stationary. The goal is to solve the incompressible Navier-Stokes equations to obtain the velocity and pressure fields.
However , we have modified the cavity from a 2D-Lid-Driven-Cavity to a two-fledged-temperature Junction Cavity where we have fixed the left and right  wall temperatures as -0.5 and +0.5 respectively and ,based on the different time intervals tried to observe the liquid Fraction contours for the different intervals.

### Key Features

- Finite difference discretization
- RK3/RK4 for time integration of intermediate velocities
- Projection method for incompressible flow
- Numba-accelerated performance
- Visualization using Matplotlib.
-Coded in Python programming Language and Jupyter Notebook
