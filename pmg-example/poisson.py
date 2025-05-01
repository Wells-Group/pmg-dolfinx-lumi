from ufl import (Coefficient, Constant, FunctionSpace, Mesh, Measure,
                 TestFunction, TrialFunction, dx, div, grad, inner, SpatialCoordinate, sin, pi)
import basix
from basix.ufl import blocked_element, wrap_element

kx = 2
ky = 3
kz = 4

# Load namespace
ns = vars()
forms = []
for degree in range(1, 8):

    family = basix.ElementFamily.P
    cell_type = basix.CellType.hexahedron
    variant = basix.LagrangeVariant.gll_warped
    e = wrap_element(basix.create_tp_element(family, cell_type, degree, variant))

    coord_ele = basix.create_tp_element(family, cell_type, 1, variant)
    coord_element = blocked_element(wrap_element(coord_ele), (3,))
    mesh = Mesh(coord_element)

    V = FunctionSpace(mesh, e)

    u = TrialFunction(V)
    v = TestFunction(V)
    x = SpatialCoordinate(mesh)
    kappa = Constant(mesh)
    f = -kappa * div(grad(sin(kx*pi*x[0]) * sin(ky*pi*x[1]) * sin(kz*pi*x[2])))

    aname = 'a' + str(degree)
    Lname = 'L' + str(degree)

    Qdegree = {1:1, 2:3, 3:4, 4:6, 5:8, 6:10, 7:12}

    # Insert into namespace so that the forms will be named a1, a2, a3 etc.
    dx = Measure("dx", metadata={"quadrature_rule": "GLL", "quadrature_degree": Qdegree[degree]})
    ns[aname] = kappa * inner(grad(u), grad(v)) * dx
    ns[Lname] = inner(f, v) * dx


    # Delete, so that the forms will get unnamed args and coefficients
    # and default to v_0, v_1, w0, w1 etc.
    del u, v, f, kappa

    forms += [ns[aname], ns[Lname]]
