Example Gallery
===============

Hands-on, runnable examples — from FEM building-block visualizations,
through canonical PDE solves, to research-grade fluid and solid mechanics.
Each page maps the underlying mathematics to a concrete TensorMesh
script you can run.

.. grid:: 1 2 3 3
   :gutter: 4

   .. grid-item-card:: Basics & Visualization
      :link: basics
      :link-type: doc

      Basis points, shape functions, element ordering, and a mesh-generation gallery.

   .. grid-item-card:: Poisson Equation
      :link: poisson
      :link-type: doc

      Elliptic PDE in 2D / 3D, batched RHS, and h-adaptive refinement on the L-shape.

   .. grid-item-card:: Diffusion
      :link: diffusion
      :link-type: doc

      Transient heat equation and nonlinear Allen-Cahn phase-field evolution.

   .. grid-item-card:: Wave Equation
      :link: wave
      :link-type: doc

      Hyperbolic PDE with explicit central-difference time stepping.

   .. grid-item-card:: ML Datasets
      :link: dataset
      :link-type: doc

      Batch generation of heat and wave snapshots for training neural operators.

   .. grid-item-card:: Distributed FEM
      :link: distributed
      :link-type: doc

      Graph coloring, spectral partitioning, and multi-GPU assembly benchmarks.

   .. grid-item-card:: Fluid Mechanics
      :link: fluid/index
      :link-type: doc

      Incompressible Navier-Stokes: cavity, cylinder flow, Rayleigh-Bénard, and more.

   .. grid-item-card:: Solid Mechanics
      :link: solid/index
      :link-type: doc

      Linear / hyperelastic / plastic / contact / fracture problems and a paper-grade dataset.

   .. grid-item-card:: Inverse Design & Identification
      :link: inverse_design
      :link-type: doc

      Differentiable FEM: coefficient-field identification and density-based topology optimization.

   .. grid-item-card:: Physics-Informed Learning
      :link: physics_informed
      :link-type: doc

      Train a neural network to minimize the assembled Galerkin residual ``||K u - F||²``.


.. toctree::
   :hidden:
   :maxdepth: 2

   basics
   poisson
   diffusion
   wave
   dataset
   distributed
   fluid/index
   solid/index
   inverse_design
   physics_informed
