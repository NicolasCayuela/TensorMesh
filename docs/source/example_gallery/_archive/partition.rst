Graph Partitioning
==================

.. image:: ../_static/mesh_algo/graph_partition_exploded.png
   :width: 700
   :align: center
   :alt: Graph Partition Exploded View

Overview
--------
Mesh partitioning (or domain decomposition) is a fundamental preprocessing step for parallel finite element analysis. The goal is to divide the mesh into :math:`k` subdomains such that:

1.  **Load Balance**: Each subdomain has roughly the same number of elements.
2.  **Minimal Interface**: The number of shared nodes (communication cost) is minimized.

TensorMesh provides a highly efficient, GPU-accelerated implementation of **Recursive Spectral Bisection**.

Algorithm Details
-----------------

Spectral Partitioning
^^^^^^^^^^^^^^^^^^^^^
The method leverages the spectral properties of the **Graph Laplacian** matrix :math:`L = D - A`, where :math:`A` is the adjacency matrix and :math:`D` is the degree matrix.

1.  **Fiedler Vector**: The algorithm computes the eigenvector corresponding to the second smallest eigenvalue of the Laplacian (known as the Fiedler vector). This vector maps the graph vertices onto a line while preserving locality.
2.  **Bisection**: Nodes are partitioned into two sets based on the median value (or sign) of the Fiedler vector components.
3.  **Recursion**: To partition into :math:`k=2^m` parts, the algorithm is applied recursively to each subgraph.

**GPU Acceleration**
Solving the eigenvalue problem for large sparse matrices is computationally intensive. TensorMesh utilizes ``torch.lobpcg`` (Locally Optimal Block Preconditioned Conjugate Gradient) on the GPU to efficiently find the smallest eigenpairs of the sparse Laplacian without moving data to the CPU.

Ghost Nodes & Overlap
^^^^^^^^^^^^^^^^^^^^^
In element-based partitioning, elements are assigned to unique partitions. However, nodes on the boundary (interface) must be shared between partitions to maintain continuity.

*   **Internal Nodes**: Nodes belonging exclusively to one partition.
*   **Ghost Nodes**: Nodes shared by two or more partitions.

TensorMesh's ``partition_mesh`` function automatically handles this topology. It returns a list of independent ``Mesh`` objects (submeshes). Each submesh contains its own local nodes, but preserves the mapping to the original mesh via the ``orig_nid`` (original node ID) point data. This allows for easy data exchange during parallel computation.

Usage
-----

.. code-block:: python

    import torch
    from tensormesh.dataset.mesh import gen_rectangle
    from tensormesh.mesh.partition import partition_mesh

    # 1. Create a mesh (e.g., triangular)
    mesh = gen_rectangle(chara_length=0.02, element_type='tri')
    if torch.cuda.is_available():
        mesh.to('cuda')

    # 2. Partition into 4 parts using Spectral method
    # Returns a list of submeshes, each is a full Mesh object
    submeshes = partition_mesh(mesh, n_parts=4, method='spectral')

    print(f"Original elements: {mesh.cells['tri'].shape[0]}")
    for i, sub in enumerate(submeshes):
        print(f"Part {i}: {sub.cells['tri'].shape[0]} elements")
        # Access original node IDs for communication
        ghost_ids = sub.point_data['orig_nid']

API Reference
-------------

.. autofunction:: tensormesh.mesh.partition.partition_mesh

