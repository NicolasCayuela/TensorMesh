# 2D phononic crystals

Two-dimensional acoustic **phononic-crystal** examples: Bloch-Floquet band
structures and a finite-slab transmission spectrum. They showcase
[`tensormesh.BlochReducer`](../../../tensormesh/operator/bloch.py), the
Bloch-Floquet periodic boundary-condition operator (the periodic counterpart of
`Condenser`).

These are deliberately example-only: no public API is added. They reuse the
existing scalar-Helmholtz assembly, the `BlochReducer` operator, and the complex
sparse solve.

| Script | Problem | What it exercises |
| --- | --- | --- |
| `band_structure_square.py` | square lattice, **rigid** cylinders in water | scalar Helmholtz, orthogonal lattice |
| `band_structure_triangular.py` | triangular lattice, **penetrable steel** in water | **two-medium** weighted assembly, **non-orthogonal** lattice |
| `transmission_slab.py` | finite slab of rigid cylinders, plane-wave drive | frequency-domain Helmholtz + first-order radiation BC |

## Model

Scalar pressure acoustics on a unit cell (band structure) or a finite strip
(transmission). A cylinder is either rigid (sound-hard, natural Neumann — meshed
as a hole) or a second penetrable acoustic medium.

**Band structure** — at each wavevector along the irreducible Brillouin-zone
path, `BlochReducer` ties the opposite cell faces with the Floquet phase and
reduces the operators to the independent (master) DOFs; a dense Hermitian
generalized eig gives the bands:

```text
mesh -> Laplace/Mass assembler -> BlochReducer -> generalized eig
```

- Square (rigid): $K_{ij}=\int\nabla\phi_i\!\cdot\!\nabla\phi_j$,
  $M_{ij}=\int\phi_i\phi_j$, $K p=(\omega/c)^2 M p$, $f=c\sqrt{\mu}/2\pi$;
  path $M\!-\!\Gamma\!-\!X\!-\!M$.
- Triangular (penetrable): material varies in space, so the operators are
  **weighted** — $K_{ij}=\int\frac1\rho\nabla\phi_i\!\cdot\!\nabla\phi_j$,
  $M_{ij}=\int\frac1{\rho c^2}\phi_i\phi_j$, $K p=\omega^2 M p$ — assembled with
  a per-element `ElementAssembler` carrying $1/\rho,\ 1/(\rho c^2)$ over a
  conformal steel/water mesh; path $M\!-\!\Gamma\!-\!K\!-\!M$.

Unit-cell meshes use gmsh `setPeriodic` (and `fragment` for the two-domain case)
so opposite edges carry matching nodes — the precondition `BlochReducer` needs.

**Transmission** — a normally-incident plane wave ($p_0=1$ Pa) crosses a finite
slab of rigid cylinders; the power transmission $T(f)=\langle|p|^2\rangle_{\rm
out}/|p_0|^2$ is swept over frequency:

```text
mesh -> Laplace/Mass assembler -> (K - k^2 M - i k B) p = -2 i k p0 e_in
```

The first-order radiation term $B$ and the incident load $e_{\rm in}$ are
hand-rolled boundary line integrals (no PML needed — this matches COMSOL's
first-order "Plane Wave Radiation"). At normal incidence the lateral periodic
walls are mirror-symmetry planes, equivalent to natural Neumann.

## Run

```bash
python band_structure_square.py
python band_structure_triangular.py
python transmission_slab.py
```

Each script exposes a `run_demo(...)` returning diagnostics and a `main()` with
`--no-plot`, `--output`, and a mesh-density / band-count flag. Each prints a
short summary; pass `--no-plot` to skip the figure entirely.

## What each script shows

- Band structure: a two-panel figure — the unit cell, and the computed band
  frequencies plotted as points along the IBZ path. The lowest band → 0 at
  $\Gamma$; band gaps open between branches.
- Transmission: a two-panel figure — the slab geometry, and $T(f)$, which drops
  to ~0 inside the band gap and recovers (with Fabry-Pérot ripples) in the pass
  bands.
