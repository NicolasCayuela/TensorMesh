Graph Coloring
==============

.. image:: ../_static/mesh_algo/graph_coloring_result.png
   :width: 600
   :align: center
   :alt: Graph Coloring Result

Effect
------

Graph coloring assigns labels (colors) to mesh elements such that **no two adjacent elements share the same color**. 

This enables **lock-free parallel assembly**: elements of the same color can be processed simultaneously without race conditions, as they never share common nodes.

TensorMesh typically produces **6-10 colors** for planar meshes using a fast GPU-parallel algorithm.

Algorithm (Pseudocode)
----------------------

.. code-block:: python

    def graph_coloring(adjacency, max_iter=100):
        n_nodes = adjacency.shape[0]
        
        # Random weights for tie-breaking
        weights = random(n_nodes)
        
        # Initialize all nodes with color 0
        colors = zeros(n_nodes, dtype=int)
        
        for iteration in range(max_iter):
            # Step 1: Detect conflicts (adjacent nodes with same color)
            conflicts = find_edges_where(colors[u] == colors[v])
            
            if no_conflicts:
                break
            
            # Step 2: Resolve conflicts
            # For each conflict pair, the node with lower weight must change
            for (u, v) in conflicts:
                if weights[u] < weights[v]:
                    nodes_to_update.add(u)
                else:
                    nodes_to_update.add(v)
            
            # Step 3: Assign new random colors from expanding range
            color_range = 6 + (iteration // 5)
            for node in nodes_to_update:
                colors[node] = randint(0, color_range)
        
        return colors

Usage
-----

.. code-block:: python

    import torch
    from tensormesh.dataset.mesh import gen_rectangle

    # 1. Create a mesh
    mesh = gen_rectangle(chara_length=0.02, element_type='tri')
    if torch.cuda.is_available():
        mesh.to('cuda')

    # 2. Color the mesh elements
    colors = mesh.color()  # Returns [n_elements] tensor

    n_colors = colors.max().item() + 1
    print(f"Mesh colored with {n_colors} colors.")

    # 3. Parallel assembly by color groups
    for c in range(n_colors):
        mask = (colors == c)
        # All elements in this group can be processed in parallel
        # No shared nodes between them
        parallel_assemble(mesh.cells['tri'][mask])

API Reference
-------------

.. autofunction:: tensormesh.mesh.coloring.graph_coloring
