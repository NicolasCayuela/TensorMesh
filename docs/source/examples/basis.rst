Basis
=====


1D Line Basis 
-------------

The 1D line element basis functions of order $p$ are Lagrange polynomials defined at $p+1$ equidistant nodes $\xi_j$ on the reference interval $[-1, 1]$ (or $[0, 1]$).

The shape function associated with node $i$ is given by:

.. math::

    N_i(\xi) = \prod_{\substack{j=0 \\ j \neq i}}^p \frac{\xi - \xi_j}{\xi_i - \xi_j}

.. code-block:: python

    from tensormesh.element import Line
    from tensormesh.element.plot import plot_1d

    fig, axes = plt.subplots(ncols=4, figsize=(16,4))

    for i, order in enumerate(range(1,5)):
        basis = Line.get_basis(order)
        basis_fns = Line.get_basis_fns(order)
        plot_1d(basis, basis_fns, ax=axes[i], legend=False)
        axes[i].set_title(f"Order:{order}")

.. image:: ../_static/plot_basis/linear.png
    :width: 100%
    :align: center

2D Triangle Basis 
-----------------

The basis functions for triangular elements are typically defined using barycentric coordinates $(\lambda_1, \lambda_2, \lambda_3)$ on the reference triangle, where $\lambda_1 + \lambda_2 + \lambda_3 = 1$. The coordinates are related to the reference coordinates $(\xi, \eta)$ by:

.. math::

    \lambda_1 = 1 - \xi - \eta, \quad \lambda_2 = \xi, \quad \lambda_3 = \eta

For a Lagrange element of order $p$, the basis functions are defined at nodes located at $(i/p, j/p)$ where $i,j \ge 0$ and $i+j \le p$.

.. code-block:: python

    from tensormesh.element import Triangle
    import matplotlib.pyplot as plt 

    fig = plt.figure((16,4))
    for basis in range(1, 5):
        ax = fig.add_subplot(1, 4, order)
        basis = Triangle.get_basis(order)
        for i in range(n_basis):
            ax.scatter(basis[i, 0], basis[i, 1], s=scatter_size)
            ax.text(basis[i, 0], basis[i, 1], f'{i+1}', fontsize=font_size)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
    
        # Draw element edges
        edges = element.points[element.edge]
        for edge in edges:
            ax.plot(edge[:, 0], edge[:, 1], 'k-', alpha=0.5)

.. image:: ../_static/plot_basis/triangle.png
    :width: 100%
    :align: center

2D Quadrilateral Basis 
----------------------

Quadrilateral elements use tensor-product basis functions constructed from 1D Lagrange polynomials. For an order $p$ element, the shape functions are:

.. math::

    N_{ij}(\xi, \eta) = N_i(\xi) N_j(\eta)

where $N_i(\xi)$ and $N_j(\eta)$ are the 1D Lagrange polynomials of order $p$ in the $\xi$ and $\eta$ directions, respectively.

.. code-block:: python

    from tensormesh.element import Quadrilateral
    import matplotlib.pyplot as plt 

    fig = plt.figure((16,4))
    for basis in range(1, 5):
        ax = fig.add_subplot(1, 4, order)
        basis = Quadrilateral.get_basis(order)
        for i in range(n_basis):
            ax.scatter(basis[i, 0], basis[i, 1], s=scatter_size)
            ax.text(basis[i, 0], basis[i, 1], f'{i+1}', fontsize=font_size)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
    
        # Draw element edges
        edges = element.points[element.edge]
        for edge in edges:
            ax.plot(edge[:, 0], edge[:, 1], 'k-', alpha=0.5)

.. image:: ../_static/plot_basis/quadrilateral.png
    :width: 100%
    :align: center

3D Tetrahedron Basis 
--------------------

Tetrahedral basis functions are defined using 3D barycentric coordinates $(\lambda_1, \lambda_2, \lambda_3, \lambda_4)$ on the reference tetrahedron, with $\sum \lambda_i = 1$.

The relationship to Cartesian reference coordinates $(\xi, \eta, \zeta)$ is:

.. math::

    \lambda_1 = 1 - \xi - \eta - \zeta, \quad \lambda_2 = \xi, \quad \lambda_3 = \eta, \quad \lambda_4 = \zeta

Lagrange basis functions are defined at equidistant nodes determined by the polynomial order $p$.

.. code-block:: python

    from tensormesh.element import Tetrahedron
    import matplotlib.pyplot as plt 

    order = 3
    basis = Tetrahedron.get_basis(order)
    fax = fig.add_subplot(1, 4, order, projection='3d')
    for i in range(n_basis):
        ax.scatter(basis[i, 0], basis[i, 1], basis[i, 2], s=scatter_size)
        ax.text(basis[i, 0], basis[i, 1], basis[i, 2], f'{i+1}', fontsize=font_size)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    # Draw element edges
    edges = element.points[element.edge]
    for edge in edges:
        ax.plot(edge[:, 0].numpy(), edge[:, 1].numpy(), edge[:, 2].numpy(), 'k-', alpha=0.5)


.. raw:: html

    <div style="text-align: center;">
        <iframe src="../_static/plot_basis/tetra.html" width="600px" height="500px"></iframe>
    </div>

3D Hexahedron Basis 
-------------------

Hexahedral elements use a tensor product of three 1D Lagrange polynomials. For an order $p$ element, the basis functions are:

.. math::

    N_{ijk}(\xi, \eta, \zeta) = N_i(\xi) N_j(\eta) N_k(\zeta)

where $N(\cdot)$ are the standard 1D Lagrange basis functions.

.. code-block:: python

    from tensormesh.element import Hexahedron
    import matplotlib.pyplot as plt 

    order = 3
    basis = Hexahedron.get_basis(order)
    fax = fig.add_subplot(1, 4, order, projection='3d')
    for i in range(n_basis):
        ax.scatter(basis[i, 0], basis[i, 1], basis[i, 2], s=scatter_size)
        ax.text(basis[i, 0], basis[i, 1], basis[i, 2], f'{i+1}', fontsize=font_size)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    # Draw element edges
    edges = element.points[element.edge]
    for edge in edges:
        ax.plot(edge[:, 0].numpy(), edge[:, 1].numpy(), edge[:, 2].numpy(), 'k-', alpha=0.5)



.. raw:: html

    <div style="text-align: center;">
        <iframe src="../_static/plot_basis/hex.html" width="600px" height="500px"></iframe>
    </div>

3D Pyramid Basis 
----------------

Pyramid elements serve as transition elements between hexahedra and tetrahedra. Their basis functions are constructed by combining tensor products in the base with a linear interpolation to the apex.

The shape functions are typically defined to ensure continuity with adjacent hexahedral (quadrilateral faces) and tetrahedral (triangular faces) elements.

.. code-block:: python

    from tensormesh.element import Pyramid
    import matplotlib.pyplot as plt 

    order = 3
    basis = Pyramid.get_basis(order)
    fax = fig.add_subplot(1, 4, order, projection='3d')
    for i in range(n_basis):
        ax.scatter(basis[i, 0], basis[i, 1], basis[i, 2], s=scatter_size)
        ax.text(basis[i, 0], basis[i, 1], basis[i, 2], f'{i+1}', fontsize=font_size)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    # Draw element edges
    edges = element.points[element.edge]
    for edge in edges:
        ax.plot(edge[:, 0].numpy(), edge[:, 1].numpy(), edge[:, 2].numpy(), 'k-', alpha=0.5)



.. raw:: html

    <div style="text-align: center;">
        <iframe src="../_static/plot_basis/pyr.html" width="600px" height="500px"></iframe>
    </div>

3D Prism Basis 
--------------

Prismatic elements (wedges) are the tensor product of a 2D triangle in the $(\xi, \eta)$ plane and a 1D line in the $\zeta$ direction.

.. math::

    N_{k,l}(\xi, \eta, \zeta) = N_k^{\Delta}(\xi, \eta) \cdot N_l^{1D}(\zeta)

where $N_k^{\Delta}$ are the triangular basis functions and $N_l^{1D}$ are the 1D linear basis functions.

.. code-block:: python

    from tensormesh.element import Prism
    import matplotlib.pyplot as plt 

    order = 3
    basis = Prism.get_basis(order)
    fax = fig.add_subplot(1, 4, order, projection='3d')
    for i in range(n_basis):
        ax.scatter(basis[i, 0], basis[i, 1], basis[i, 2], s=scatter_size)
        ax.text(basis[i, 0], basis[i, 1], basis[i, 2], f'{i+1}', fontsize=font_size)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    # Draw element edges
    edges = element.points[element.edge]
    for edge in edges:
        ax.plot(edge[:, 0].numpy(), edge[:, 1].numpy(), edge[:, 2].numpy(), 'k-', alpha=0.5)



.. raw:: html

    <div style="text-align: center;">
        <iframe src="../_static/plot_basis/pri.html" width="600px" height="500px"></iframe>
    </div>