# Fluid Dynamics Examples

This directory contains incompressible flow examples built with TensorMesh.

## Main Example: Cylinder Wake and Karman Vortex Street

### `cylinder_flow.py`

This script solves transient 2D incompressible Navier-Stokes flow in a channel with a circular cylinder.

- Geometry (DFG-style): channel `[0, 2.2] x [0, 0.41]`, cylinder center `(0.2, 0.2)`, radius `0.05`
- Reynolds number: default `Re=100`
- Time discretization: implicit Euler
- Nonlinearity: Picard linearization
- Stabilization: SUPG/PSPG for equal-order velocity-pressure discretization (P1/P1)
- Boundary conditions:
  - Inlet: parabolic profile
  - Top and bottom walls: no-slip
  - Cylinder wall: no-slip
  - Outlet: do-nothing for velocity, one pressure gauge point is pinned (`p=0`)

The script stores:

- `cylinder_flow.mp4`: time animation of vorticity/speed/pressure
- `cylinder_flow_final.png`: final snapshot

Run:

```bash
cd examples/fluid
python cylinder_flow.py
```

## Other Fluid Examples

### `cavity.py`

Lid-driven cavity benchmark for incompressible Navier-Stokes.

### `rayleigh_benard.py`

Boussinesq convection in a heated cavity.

### `flow_obstacles.py`

Steady flow through a channel with multiple circular obstacles.

### `taylor_green.py`

2D decaying Taylor-Green vortex with exact analytical solution. Runs a convergence study over multiple grid sizes and verifies O(h²) spatial convergence rate. Useful for validating the transient Navier-Stokes solver.

### `cavity_3d.py`

3D lid-driven cavity at Re=100 using tetrahedral elements. Demonstrates that the SUPG/PSPG stabilized Navier-Stokes assembler generalizes to 3D without modification. Exports results to VTU and renders mid-plane cross-sections via PyVista.


## Notes on Stabilization

For advection-dominated regimes, the standard Galerkin method with equal-order elements can become unstable.
This folder uses SUPG/PSPG terms to improve robustness for transient Navier-Stokes:

- SUPG improves stability in momentum equations for convection-dominated flow
- PSPG stabilizes pressure for equal-order interpolation

The stabilization parameter `tau` is updated using local velocity scale and mesh size:

```python
tau = h**2 / (4 * mu + 2 * rho * |u| * h)
```

