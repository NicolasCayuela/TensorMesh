Basis
=====

Line
----

The line basis functions are used to interpolate values along a 1D line element. For a line element of order n, there are n+1 basis functions.

For order 1 (linear), there are 2 basis functions:

.. math::
    \begin{align*}
    N_1(x) &= 1 - x \\
    N_2(x) &= x
    \end{align*}

For order 2 (quadratic), there are 3 basis functions:

.. math::
    \begin{align*}
    N_1(x) &= 2x^2 - 3x + 1 \\
    N_2(x) &= -4x^2 + 4x \\
    N_3(x) &= 2x^2 - x
    \end{align*}

For order 3 (cubic), there are 4 basis functions:

.. math::
    \begin{align*}
    N_1(x) &= -4.5x^3 + 9x^2 - 5.5x + 1 \\
    N_2(x) &= 13.5x^3 - 22.5x^2 + 9x \\
    N_3(x) &= -13.5x^3 + 18x^2 - 4.5x \\
    N_4(x) &= 4.5x^3 - 4.5x^2 + x
    \end{align*}

The basis functions have the following key properties:

1. **Partition of unity**: At any point, the sum of all basis functions equals 1
2. **Interpolation**: Each basis function equals 1 at its own node and 0 at all other nodes
3. **Continuity**: The functions are continuous across element boundaries

The figures below show the basis functions and node locations for different polynomial orders.

.. figure:: ../_static/basis/line.png
   :alt: Line Basis Function


The figures below show the basis points (nodes) for line elements of different polynomial orders. The basis points represent the locations where the basis functions take on values of 1 or 0.

For a line element:

The basis points (nodes) for line elements are the locations where the basis functions take on values of 1 or 0. These points are used to define the interpolation properties of the element.

For a line element, the basis points are distributed along the line segment [0,1] according to the polynomial order:

.. figure:: ../_static/basis/basis_points/line_comparison.png
   :alt: Line element basis points comparison showing node locations for polynomial orders 1-4

For a line element of polynomial order :math:`p`, the number of basis points :math:`n` is given by:

.. math::

    n = p + 1

Higher order elements (:math:`p > 1`) have more basis points, enabling higher-order polynomial interpolation for improved approximation of curved geometries and solutions.



Triangle
--------

The triangle basis functions are used to interpolate values across a 2D triangular element. For a triangle element of order n, there are (n+1)(n+2)/2 basis functions.

The basis functions are defined using barycentric coordinates (L1, L2, L3) where:
- L1 + L2 + L3 = 1 
- Each Li represents the relative distance from a vertex
- Li = 1 at vertex i and 0 at other vertices

For order 1 (linear), there are 3 basis functions:

.. math::
    \begin{align*}
    N_1(L_1,L_2,L_3) &= L_1 \\
    N_2(L_1,L_2,L_3) &= L_2 \\
    N_3(L_1,L_2,L_3) &= L_3
    \end{align*}

For order 2 (quadratic), there are 6 basis functions:

.. math::
    \begin{align*}
    N_1(L_1,L_2,L_3) &= L_1(2L_1-1) \\
    N_2(L_1,L_2,L_3) &= L_2(2L_2-1) \\
    N_3(L_1,L_2,L_3) &= L_3(2L_3-1) \\
    N_4(L_1,L_2,L_3) &= 4L_1L_2 \\
    N_5(L_1,L_2,L_3) &= 4L_2L_3 \\
    N_6(L_1,L_2,L_3) &= 4L_3L_1
    \end{align*}

The basis functions have the following key properties:

1. **Partition of unity**: At any point, the sum of all basis functions equals 1
2. **Interpolation**: Each basis function equals 1 at its own node and 0 at all other nodes
3. **Continuity**: The functions are continuous across element boundaries

.. figure:: ../_static/basis/triangle.png
   :alt: Triangle Basis Function

The figures below show the basis points (nodes) for triangle elements of different polynomial orders. The basis points represent the locations where the basis functions take on values of 1 or 0.

For a triangle element, the basis points are distributed according to the polynomial order:

.. figure:: ../_static/basis/basis_points/triangle_comparison.png
   :alt: Triangle element basis points comparison showing node locations for polynomial orders 1-4

For a triangle element of polynomial order p, the number of basis points n is given by:

.. math::

    n = \frac{(p+1)(p+2)}{2}

Higher order elements (p > 1) have more basis points, enabling higher-order polynomial interpolation for improved approximation of curved geometries and solutions. The trade-off is increased computational cost with more degrees of freedom.

Quadrilateral
-------------
For quadrilateral elements, the shape functions are constructed using tensor products of 1D Lagrange polynomials in the reference coordinates .. math:: (ξ,η) \in [-1,1]\times[-1,1]. For a polynomial order p, the shape functions are:

.. math::

    N_{ij}(ξ,η) = l_i(ξ)l_j(η)

where l_i(ξ) and l_j(η) are the 1D Lagrange polynomials of order p.

The shape functions for a linear quadrilateral element (p=1) are:

.. math::
    \begin{align*}
    N_1(ξ,η) &= \frac{1}{4}(1-ξ)(1-η) \\
    N_2(ξ,η) &= \frac{1}{4}(1+ξ)(1-η) \\
    N_3(ξ,η) &= \frac{1}{4}(1+ξ)(1+η) \\
    N_4(ξ,η) &= \frac{1}{4}(1-ξ)(1+η)
    \end{align*}

For quadratic elements, additional shape functions are added for the edge midpoints and center:

.. math::
    \begin{align*}
    N_5(ξ,η) &= \frac{1}{2}(1-ξ^2)(1-η) \\
    N_6(ξ,η) &= \frac{1}{2}(1+ξ)(1-η^2) \\
    N_7(ξ,η) &= \frac{1}{2}(1-ξ^2)(1+η) \\
    N_8(ξ,η) &= \frac{1}{2}(1-ξ)(1-η^2) \\
    N_9(ξ,η) &= (1-ξ^2)(1-η^2)
    \end{align*}

.. figure:: ../_static/basis/quadrilateral.png
   :alt: Quadrilateral Basis Function

The basis points represent the nodes where each shape function equals 1 while all others equal 0. For a quadrilateral element, the basis points are distributed according to the polynomial order:

.. figure:: ../_static/basis/basis_points/quadrilateral_comparison.png
   :alt: Quadrilateral element basis points comparison showing node locations for polynomial orders 1-4

For a quadrilateral element of polynomial order p, the number of basis points n is given by:

.. math::
    n = (p+1)^2

Higher order elements (p > 1) have more basis points, enabling higher-order polynomial interpolation for improved approximation of curved geometries and solutions. The trade-off is increased computational cost with more degrees of freedom.


Tetrahedron
-----------
For tetrahedral elements, the shape functions are defined using barycentric coordinates (L1, L2, L3, L4) where:

- L1 + L2 + L3 + L4 = 1
- Each Li represents the relative distance from a vertex
- Li = 1 at vertex i and 0 at other vertices

For order 1 (linear), there are 4 basis functions:

.. math::
    \begin{align*}
    N_1(L_1,L_2,L_3,L_4) &= L_1 \\
    N_2(L_1,L_2,L_3,L_4) &= L_2 \\
    N_3(L_1,L_2,L_3,L_4) &= L_3 \\
    N_4(L_1,L_2,L_3,L_4) &= L_4
    \end{align*}

For order 2 (quadratic), there are 10 basis functions:

.. math::
    \begin{align*}
    N_1(L_1,L_2,L_3,L_4) &= L_1(2L_1-1) \\
    N_2(L_1,L_2,L_3,L_4) &= L_2(2L_2-1) \\
    N_3(L_1,L_2,L_3,L_4) &= L_3(2L_3-1) \\
    N_4(L_1,L_2,L_3,L_4) &= L_4(2L_4-1) \\
    N_5(L_1,L_2,L_3,L_4) &= 4L_1L_2 \\
    N_6(L_1,L_2,L_3,L_4) &= 4L_2L_3 \\
    N_7(L_1,L_2,L_3,L_4) &= 4L_3L_1 \\
    N_8(L_1,L_2,L_3,L_4) &= 4L_1L_4 \\
    N_9(L_1,L_2,L_3,L_4) &= 4L_2L_4 \\
    N_{10}(L_1,L_2,L_3,L_4) &= 4L_3L_4
    \end{align*}

The basis functions have the following key properties:

1. **Partition of unity**: At any point, the sum of all basis functions equals 1
2. **Interpolation**: Each basis function equals 1 at its own node and 0 at all other nodes
3. **Continuity**: The functions are continuous across element boundaries

The figures below show the basis functions for tetrahedral elements of different orders:

* Order 1 (Linear):

  .. figure:: ../_static/basis/tetrahedron/1.png
     :alt: Tetrahedron Basis Function for order 1

* Order 2 (Quadratic): 

  .. figure:: ../_static/basis/tetrahedron/2.png
     :alt: Tetrahedron Basis Function for order 2

.. * Order 3 (Cubic):

..   .. figure:: ../_static/basis/tetrahedron/3.png
..      :alt: Tetrahedron Basis Function for order 3

.. * Order 4 (Quartic):

..   .. figure:: ../_static/basis/tetrahedron/4.png
..      :alt: Tetrahedron Basis Function for order 4

The figures below show the basis points (nodes) for tetrahedral elements of different polynomial orders. The basis points represent the locations where the basis functions take on values of 1 or 0.

For a tetrahedral element, the basis points are distributed according to the polynomial order:

.. figure:: ../_static/basis/basis_points/tetrahedron_comparison.png
   :alt: Tetrahedron element basis points comparison showing node locations for polynomial orders 1-4

For a tetrahedral element of polynomial order p, the number of basis points n is given by:

.. math::

    n = \frac{(p+1)(p+2)(p+3)}{6}

Higher order elements (p > 1) have more basis points, enabling higher-order polynomial interpolation for improved approximation of curved geometries and solutions. The trade-off is increased computational cost with more degrees of freedom.



Hexahedron
----------

The hexahedral basis functions are used to interpolate values within a 3D hexahedral element. For a hexahedral element of order n, there are (n+1)³ basis functions.

For order 1 (linear), there are 8 basis functions:

.. math::
    \begin{align*}
    N_1(x,y,z) &= (1-x)(1-y)(1-z) \\
    N_2(x,y,z) &= x(1-y)(1-z) \\
    N_3(x,y,z) &= xy(1-z) \\
    N_4(x,y,z) &= (1-x)y(1-z) \\
    N_5(x,y,z) &= (1-x)(1-y)z \\
    N_6(x,y,z) &= x(1-y)z \\
    N_7(x,y,z) &= xyz \\
    N_8(x,y,z) &= (1-x)yz
    \end{align*}

The basis functions have the following key properties:

1. **Partition of unity**: At any point, the sum of all basis functions equals 1
2. **Interpolation**: Each basis function equals 1 at its own node and 0 at all other nodes
3. **Continuity**: The functions are continuous across element boundaries

The figures below show the basis functions for hexahedral elements of different orders:

* Order 1 (Linear):

  .. figure:: ../_static/basis/hexahedron/1.png
     :alt: Hexahedron Basis Function for order 1

* Order 2 (Quadratic): 

  .. figure:: ../_static/basis/hexahedron/2.png
     :alt: Hexahedron Basis Function for order 2

.. * Order 3 (Cubic):

..   .. figure:: ../_static/basis/hexahedron/3.png
..      :alt: Hexahedron Basis Function for order 3

.. * Order 4 (Quartic):

..   .. figure:: ../_static/basis/hexahedron/4.png
..      :alt: Hexahedron Basis Function for order 4

The figures below show the basis points (nodes) for hexahedral elements of different polynomial orders. The basis points represent the locations where the basis functions take on values of 1 or 0.

For a hexahedral element, the basis points are distributed according to the polynomial order:

.. figure:: ../_static/basis/basis_points/hexahedron_comparison.png
   :alt: Hexahedron element basis points comparison showing node locations for polynomial orders 1-4

For a hexahedral element of polynomial order p, the number of basis points n is given by:

.. math::

    n = (p+1)^3

Higher order elements (p > 1) have more basis points, enabling higher-order polynomial interpolation for improved approximation of curved geometries and solutions. The trade-off is increased computational cost with more degrees of freedom.

Pyramid
-------

The pyramidal basis functions are used to interpolate values within a 3D pyramidal element. For a pyramidal element of order n, there are specific numbers of basis functions depending on the order.

For order 1 (linear), there are 5 basis functions:

.. math::
    \begin{align*}
    N_1(x,y,z) &= (1-x-z)(1-y-z)/(1-z) \\
    N_2(x,y,z) &= x(1-y-z)/(1-z) \\
    N_3(x,y,z) &= xy/(1-z) \\
    N_4(x,y,z) &= (1-x-z)y/(1-z) \\
    N_5(x,y,z) &= z
    \end{align*}

The basis functions have the following key properties:

1. **Partition of unity**: At any point, the sum of all basis functions equals 1
2. **Interpolation**: Each basis function equals 1 at its own node and 0 at all other nodes
3. **Continuity**: The functions are continuous across element boundaries

The figures below show the basis functions for pyramidal elements of different orders:

* Order 1 (Linear):

  .. figure:: ../_static/basis/pyramid/1.png
     :alt: Pyramid Basis Function for order 1

* Order 2 (Quadratic): 

  .. figure:: ../_static/basis/pyramid/2.png
     :alt: Pyramid Basis Function for order 2

.. * Order 3 (Cubic):

..   .. figure:: ../_static/basis/pyramid/3.png
..      :alt: Pyramid Basis Function for order 3

.. * Order 4 (Quartic):

..   .. figure:: ../_static/basis/pyramid/4.png
..      :alt: Pyramid Basis Function for order 4

The figures below show the basis points (nodes) for pyramidal elements of different polynomial orders. The basis points represent the locations where the basis functions take on values of 1 or 0.

For a pyramidal element, the basis points are distributed according to the polynomial order:

.. figure:: ../_static/basis/basis_points/pyramid_comparison.png
   :alt: Pyramid element basis points comparison showing node locations for polynomial orders 1-4

For a pyramidal element of polynomial order p, the number of basis points n is given by:

.. math::

    n = \frac{(p+1)(p+2)(2p+3)}{6}

Higher order elements (p > 1) have more basis points, enabling higher-order polynomial interpolation for improved approximation of curved geometries and solutions. The trade-off is increased computational cost with more degrees of freedom.

Prism
-----

The prismatic basis functions are used to interpolate values within a 3D prismatic element. For a prismatic element of order n, there are specific numbers of basis functions depending on the order.

For order 1 (linear), there are 6 basis functions:

.. math::
    \begin{align*}
    N_1(L_1,L_2,L_3,z) &= L_1(1-z) \\
    N_2(L_1,L_2,L_3,z) &= L_2(1-z) \\
    N_3(L_1,L_2,L_3,z) &= L_3(1-z) \\
    N_4(L_1,L_2,L_3,z) &= L_1z \\
    N_5(L_1,L_2,L_3,z) &= L_2z \\
    N_6(L_1,L_2,L_3,z) &= L_3z
    \end{align*}

The basis functions have the following key properties:

1. **Partition of unity**: At any point, the sum of all basis functions equals 1
2. **Interpolation**: Each basis function equals 1 at its own node and 0 at all other nodes
3. **Continuity**: The functions are continuous across element boundaries

The figures below show the basis functions for prismatic elements of different orders:

* Order 1 (Linear):

  .. figure:: ../_static/basis/prism/1.png
     :alt: Prism Basis Function for order 1

* Order 2 (Quadratic): 

  .. figure:: ../_static/basis/prism/2.png
     :alt: Prism Basis Function for order 2

.. * Order 3 (Cubic):

..   .. figure:: ../_static/basis/prism/3.png
..      :alt: Prism Basis Function for order 3

.. * Order 4 (Quartic):

..   .. figure:: ../_static/basis/prism/4.png
..      :alt: Prism Basis Function for order 4

The figures below show the basis points (nodes) for prismatic elements of different polynomial orders. The basis points represent the locations where the basis functions take on values of 1 or 0.

For a prismatic element, the basis points are distributed according to the polynomial order:

.. figure:: ../_static/basis/basis_points/prism_comparison.png
   :alt: Prism element basis points comparison showing node locations for polynomial orders 1-4

For a prismatic element of polynomial order p, the number of basis points n is given by:

.. math::

    n = \frac{(p+1)^2(p+2)}{2}

Higher order elements (p > 1) have more basis points, enabling higher-order polynomial interpolation for improved approximation of curved geometries and solutions. The trade-off is increased computational cost with more degrees of freedom.
