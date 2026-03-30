# Hyperelastic Beam (Neo-Hookean)

Large deformation of a rubber beam under torsion using a compressible Neo-Hookean material model.

## Problem Setup

- **Geometry:** 1.0 x 0.4 x 0.4 m beam (quadratic tetrahedra, order=2)
- **Material:** Rubber (E=10 MPa, nu=0.48, near-incompressible)
  - Strain energy: Psi = mu/2 (I1 - 3) - mu ln(J) + lam/2 (ln J)^2
- **Boundary Conditions:**
  - Left end (x=0): fully clamped
  - Right end (x=1): torsional force field F = C*(0, -dz, dy) with C=3e4
- **Solver:** LBFGS energy minimization with 10 load steps (ramped torsion)

## Usage

```bash
python hyperelastic_beam.py
```

## Output

- `hyperelastic_rubber.png`: deformed configuration (isometric view)
- Console: max displacement per load step
