# Emmentaler Solid Mechanics

Three progressive examples solving solid mechanics on the **Emmentaler geometry** — a 3D block (1.0 x 1.0 x 1.5) with 23 randomly-placed spherical holes (R=0.16), reproducing the setup from the `solidmechanics_datagen` project.

## Geometry & Boundary Conditions

All three examples share the same geometry and boundary conditions:

- **Bottom face (z=0):** fully clamped (u=0)
- **Top face (z=T):** prescribed displacement combining:
  - Tension: u_z = lambda * 0.010
  - Bending: curvature kappa = lambda * 0.014
  - Torsion: angle theta = lambda * 0.048 rad
- **Material:** E=12000, nu=0.3

The mesh is generated via Gmsh (no DOLFINx dependency) or loaded from file.

## Scripts

### Phase 1: `emmentaler_elasticity.py` — Linear Elasticity

Small-strain linear elastic solver. Single direct solve per load step.

```bash
python emmentaler_elasticity.py --h 0.08 --steps 11
```

**Output:** displacement, strain (Voigt), stress (Voigt), von Mises stress

### Phase 2: `emmentaler_hyperelastic.py` — Neo-Hookean Hyperelasticity

Finite-strain compressible Neo-Hookean model solved via LBFGS energy minimization with load stepping.

```bash
python emmentaler_hyperelastic.py --h 0.08 --load_steps 10
```

**Output:** displacement, Green-Lagrange strain, Cauchy stress, von Mises stress

### Phase 3: `emmentaler_phasefield.py` — Phase-Field Fracture (AT1)

Brittle phase-field fracture with Amor energy split. Staggered (alternating minimization) solver:
- u sub-problem: LBFGS with alpha frozen
- alpha sub-problem: projected LBFGS with irreversibility constraint (alpha >= alpha_old)

```bash
# Quick test (coarse mesh, large ell for resolution)
python emmentaler_phasefield.py --h 0.1 --ell 0.15 --Gc 0.05 --load_steps 20

# Production parameters (requires h < ell)
python emmentaler_phasefield.py --h 0.03 --ell 0.075 --Gc 0.0014 --load_steps 201
```

**Output:** displacement, damage field (alpha), strain, stress, von Mises stress

## Common Options

| Flag | Description | Default |
|------|-------------|---------|
| `--mesh_file` | Load existing mesh (.msh, .vtk, .xdmf) | generate via Gmsh |
| `--h` | Mesh element size | 0.15 |
| `--E` | Young's modulus | 12000 |
| `--nu` | Poisson's ratio | 0.3 |
| `--normal_mag` | Tension magnitude [0,1] | 1.0 |
| `--bending_mag` | Bending magnitude [0,1] | 1.0 |
| `--torsion_mag` | Torsion magnitude [0,1] | 1.0 |
| `--output_dir` | Output directory | varies |

## ParaView Visualization

1. Open the `.vtk` file
2. Coloring: select `von_mises_stress` or `alpha` (for phase-field)
3. Warp By Vector: select `displacement`, scale factor 10-20x
4. Clip: Normal [0,1,0] to see internal hole structure
