Adjacency 
=========

The adjacency information of a mesh describes how its components (nodes, elements) are connected to each other. This is crucial for building graph-based representations of the mesh.

2D Node Adjacency
-----------------

Node adjacency describes the connectivity between nodes. Two nodes $i$ and $j$ are considered adjacent if they belong to the same edge in the mesh.

The adjacency matrix $A \in \{0, 1\}^{N_v \times N_v}$ is defined as:

.. math::

    A_{ij} = \begin{cases} 
    1 & \text{if nodes } i \text{ and } j \text{ share an edge} \\
    0 & \text{otherwise}
    \end{cases}

where $N_v$ is the number of vertices (nodes).

.. code-block:: python 

    from tensormesh import mesh,MeshGen
    import tensormesh as tm
    import tensormesh.visualization as V

    GRAPH_ALPHA = 0.5
    GRAPH_COLOR = "lightblue"
    GRAPH_LINEWIDTH = 1 
    MESH_COLOR = "black"
    MESH_LINEWIDTH = 2

    mesh_gen = tm.MeshGen(element_type=None, chara_length=0.3, order=2)
    mesh_gen.add_rectangle(0,0,0.5,1, element="tri")
    mesh_gen.add_rectangle(0.5,0,0.5,1, element="quad")
    mesh_gen.remove_circle(0.5,0.5,0.1)
    mesh:tm.Mesh = mesh_gen.gen()
    ax = V.draw_graph(mesh.node_adjacency(), mesh.points, 
                      color=GRAPH_COLOR,
                      linewidth=GRAPH_LINEWIDTH,
                      alpha=GRAPH_ALPHA)
    mesh.plot(ax=ax, edgecolor=MESH_COLOR, linewidth=MESH_LINEWIDTH)


.. image:: ../_static/plot_mesh/node_adj_2d.png
    :width: 100%
    :align: center

2D Element Adjacency
--------------------

Element adjacency describes the connectivity between elements. Two elements $e_i$ and $e_j$ are considered adjacent if they share a common face (or an edge in 2D).

The element adjacency matrix $E \in \{0, 1\}^{N_e \times N_e}$ is defined as:

.. math::

    E_{ij} = \begin{cases} 
    1 & \text{if elements } e_i \text{ and } e_j \text{ share a common boundary (edge in 2D, face in 3D)} \\
    0 & \text{otherwise}
    \end{cases}

where $N_e$ is the number of elements.

.. code-block:: python

    from tensormesh import mesh,MeshGen
    import tensormesh as tm
    import tensormesh.visualization as V

    GRAPH_ALPHA = 0.5
    GRAPH_COLOR = "lightblue"
    GRAPH_LINEWIDTH = 1 
    MESH_COLOR = "black"
    MESH_LINEWIDTH = 2

    mesh_gen = tm.MeshGen(element_type=None, chara_length=0.3, order=2)
    mesh_gen.add_rectangle(0,0,0.5,1, element="tri") 
    mesh_gen.add_rectangle(0.5,0,0.5,1, element="quad")
    mesh_gen.remove_circle(0.5,0.5,0.1)
    mesh:tm.Mesh = mesh_gen.gen()
    points = []
    for key , elements in mesh.elements().items():
        points.append(mesh.points[elements].mean(-2))
    points = torch.cat(points, 0)

    ax = V.draw_graph(mesh.element_adjacency(), points,
                      color=GRAPH_COLOR,
                      linewidth=GRAPH_LINEWIDTH,
                      alpha=GRAPH_ALPHA)
    mesh.plot(ax=ax, edgecolor=MESH_COLOR, linewidth=MESH_LINEWIDTH)
    

.. image:: ../_static/plot_mesh/ele_adj_2d.png
    :width: 100%
    :align: center

3D Node Adjacency
-----------------

.. code-block:: python

    from tensormesh import mesh,MeshGen
    import tensormesh as tm
    import tensormesh.visualization as V

    GRAPH_ALPHA = 0.5
    GRAPH_COLOR = "lightblue"
    GRAPH_LINEWIDTH = 1 
    MESH_COLOR = "black"
    MESH_LINEWIDTH = 2

    mesh_gen = tm.MeshGen(element_type=None, chara_length=0.1, order=1, dimension=3)
    mesh_gen.add_cube(0,0,0,1,1,1,element="tet")
    # mesh_gen.remove_sphere(0.5,0.5,0.5,0.1)
    mesh:tm.Mesh = mesh_gen.gen()
    ax = V.draw_graph(mesh.node_adjacency(), mesh.points,
                      color=GRAPH_COLOR,
                      linewidth=GRAPH_LINEWIDTH,
                      alpha=GRAPH_ALPHA)
    fig = ax.get_figure()
    mesh.plot(ax=ax, edgecolor=MESH_COLOR, linewidth=MESH_LINEWIDTH)
    
.. raw:: html

    <div style="text-align: center;">
        <iframe src="../_static/plot_mesh/node_adj_3d.html" width="600px" height="500px"></iframe>
    </div>



3D Element Adjacency
--------------------

.. code-block:: python

    from tensormesh import mesh,MeshGen
    import tensormesh as tm
    import tensormesh.visualization as V

    GRAPH_ALPHA = 0.5
    GRAPH_COLOR = "lightblue"
    GRAPH_LINEWIDTH = 1 
    MESH_COLOR = "black"
    MESH_LINEWIDTH = 2

    mesh_gen = tm.MeshGen(element_type=None, chara_length=0.2, order=1, dimension=3)
    mesh_gen.add_cube(0,0,0,1,1,1,element="tet")
    # mesh_gen.remove_sphere(0.5,0.5,0.5,0.1)
    mesh:tm.Mesh = mesh_gen.gen()
    ax = V.draw_graph(mesh.element_adjacency(), mesh.points[mesh.elements()].mean(-2),
                      color=GRAPH_COLOR,
                      linewidth=GRAPH_LINEWIDTH,
                      alpha=GRAPH_ALPHA)
    fig = ax.get_figure()
    mesh.plot(ax=ax, edgecolor=MESH_COLOR, linewidth=MESH_LINEWIDTH)
    
.. raw:: html

    <div style="text-align: center;">
        <iframe src="../_static/plot_mesh/ele_adj_3d.html" width="600px" height="500px"></iframe>
    </div>