# TODO List


## tests
- [x] assemble
  - [x] elemenet
  - [x] node 
- [ ] sparse
  - [ ] solve
    - [x] scipy solve
    - [ ] petsc solve
    - [x] cupy(cusolve) solve
  - [x] mm
- [x] adjacency
  - [x] node adjacency
  - [x] element adjacency


## benchmark
- [ ] assemble speed / memory
- [ ] pipeline speed / memory


## Example
- [ ] poisson
  - [x] naive
  - [ ] adaptive mesh
- [ ] heat
  - [x] naive
  - [ ] pinn
- [ ] wave
  - [x] naive
  - [ ] pinn 
- [ ] linear elasiticity
  - [x] naive
  - [ ] dynamic
- [ ] Fluid mechanics

## torch_fem
### Function
- [x] condense
- [ ] quadrature
  - [x] infinite quadrature for euclidean element 
  - [x] large enough quadrature for triangle
  - [x] large enough quadrature for tetra 
  - [ ] large enough quadrature for wedge
- [ ] shape 
  - [ ] line
    - [x] 2 order of shape fn and shape grad 
    - [ ] infinite order of shape fn and shape grad
  - [ ] triangle
    - [x] 2 order of shape fn and shape grad 
    - [ ] infinite order of shape fn and shape grad
  - [ ] quadliteral
    - [x] 2 order of shape fn and shape grad 
    - [ ] infinite order of shape fn and shape grad
  - [ ] tetra
    - [x] 2 order of shape fn and shape grad 
    - [ ] infinite order of shape fn and shape grad 
  - [ ] brick
    - [ ] 2 order of shape fn and shape grad 
    - [ ] infinite order of shape fn and shape grad
  - [ ] wedge
    - [ ] 2 order of shape fn and shape grad 
    - [ ] infinite order of shape fn and shape grad
- [ ] mesh 
  - [x] adjacency(for gnn)
    - [x] node adjacency 
    - [x] edge adjacency 
  - [x] mixed mesh 
- [x] assembler
  - [x] element assembler 
  - [x] node assembler
- [ ] gnn 
- [ ] ODE
- [ ] dataset
  - [ ] mesh 
    - [ ] generator
      - [x] gmsh backend
      - [ ] add more function
      - [ ] add distmsh support, since gmsh requires complex dependencies
    - [x] (hollow)rectangle
    - [x] (hollow)circle 
    - [x] Lshape
    - [x] (hollow)cube
    - [x] (hollow)sphere
    - [ ] airfoil/aircraft
- [ ] sparse matrix 
  - [x] spmv/spmm 
  - [x] spsolve
  - [x] combine
    - [x] combine vector
    - [x] combine matrix
  - [ ] elementwise-op
    - [x] same layout 
    - [ ] different layout 
  - [ ] partition
  - [ ] io
  - [ ] det
  - [ ] is_pos_definite 
- [ ] strong form to weak form

### Efficiency
- [x] quadrature loop
- [x] PETsc backend
- [ ] distributed mesh 
  - [ ] distributed mesh assemble 
  - [ ] distributed linear system solve

### Bugs/others
- [ ] retain-graph = True issue fix 
