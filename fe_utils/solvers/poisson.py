"""Solve a model poisson problem with Dirichlet boundary conditions
using the finite element method.

If run as a script, the result is plotted. This file can also be
imported as a module and convergence tests run on the solver.
"""
from __future__ import division
from fe_utils import *
import numpy as np
from numpy import sin, pi
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from argparse import ArgumentParser


def assemble(fs, f):
    """Assemble the finite element system for the Poisson problem given
    the function space in which to solve and the right hand side
    function."""

    #raise NotImplementedError

    A = sp.lil_matrix((fs.node_count, fs.node_count))
    #A = np.zeros([fs.node_count,fs.node_count])
    l = np.zeros(fs.node_count)

    finele = fs.element
    QuadRule = gauss_quadrature(finele.cell,2*finele.degree)
    numq = QuadRule.points.shape[0]
    
    PhiTab = finele.tabulate(QuadRule.points)
    gPhiTab = finele.tabulate(QuadRule.points,grad=True)
    mesh = fs.mesh
    M = fs.cell_nodes
    
    for c in range(0,mesh.entity_counts[-1]):
        Jdet = np.absolute(np.linalg.det(mesh.jacobian(c)))
        JinvT = np.transpose(np.linalg.inv(mesh.jacobian(c)))

        for i in range(0,fs.element.node_count):
            #do for RHS l 
            ModW = np.multiply(QuadRule.weights,PhiTab[:,i])
            PhiW = np.dot(ModW,PhiTab)
            FInt = np.dot(f.values[M[c,:]],PhiW)
            l[M[c,i]] += FInt*Jdet

            #do for LHS A
            for j in range(0,fs.element.node_count):
                toadd = 0
                for q in range(0,numq):
                    gdotg = np.dot(np.dot(JinvT,gPhiTab[q,i,:]),np.dot(JinvT,gPhiTab[q,j,:]))
                    toadd += gdotg*QuadRule.weights[q]
                #A[np.ix_([M[c,i]],[M[c,j]])] += toadd*Jdet
                A[M[c,i],M[c,j]] += toadd*Jdet
                #print(A)
                
    #print(M)
    #print(A.toarray())
    bdrnodes = boundary_nodes(fs)
    for r in bdrnodes:
        A[r,:] = 0
        l[r] = 0
        A[r,r] = 1
    
    return A, l


def boundary_nodes(fs):
    """Find the list of boundary nodes in fs. This is a
    unit-square-specific solution. A more elegant solution would employ
    the mesh topology and numbering.
    """
    eps = 1.e-10

    f = Function(fs)

    def on_boundary(x):
        """Return 1 if on the boundary, 0. otherwise."""
        if x[0] < eps or x[0] > 1 - eps or x[1] < eps or x[1] > 1 - eps:
            return 1.
        else:
            return 0.

    f.interpolate(on_boundary)

    return np.flatnonzero(f.values)


def solve_poisson(degree, resolution, analytic=False, return_error=False):
    """Solve a model Poisson problem on a unit square mesh with
    ``resolution`` elements in each direction, using equispaced
    Lagrange elements of degree ``degree``."""

    # Set up the mesh, finite element and function space required.
    mesh = UnitSquareMesh(resolution, resolution)
    fe = LagrangeElement(mesh.cell, degree)
    fs = FunctionSpace(mesh, fe)

    # Create a function to hold the analytic solution for comparison purposes.
    analytic_answer = Function(fs)
    analytic_answer.interpolate(lambda x: sin(4*pi*x[0])*x[1]**2*(1.-x[1])**2)

    # If the analytic answer has been requested then bail out now.
    if analytic:
        return analytic_answer, 0.0

    # Create the right hand side function and populate it with the
    # correct values.
    f = Function(fs)
    f.interpolate(lambda x: (16*pi**2*(x[1] - 1)**2*x[1]**2 - 2*(x[1] - 1)**2 -
                             8*(x[1] - 1)*x[1] - 2*x[1]**2) * sin(4*pi*x[0]))

    # Assemble the finite element system.
    A, l = assemble(fs, f)

    # Create the function to hold the solution.
    u = Function(fs)

    # Cast the matrix to a sparse format and use a sparse solver for
    # the linear system. This is vastly faster than the dense
    # alternative.
    A = sp.csr_matrix(A)
    u.values[:] = splinalg.spsolve(A, l)

    # Compute the L^2 error in the solution for testing purposes.
    error = errornorm(analytic_answer, u)

    if return_error:
        u.values -= analytic_answer.values

    # Return the solution and the error in the solution.
    return u, error

if __name__ == "__main__":

    parser = ArgumentParser(
        description="""Solve a Poisson problem on the unit square.""")
    parser.add_argument("--analytic", action="store_true",
                        help="Plot the analytic solution instead of solving the finite element problem.")
    parser.add_argument("--error", action="store_true",
                        help="Plot the error instead of the solution.")
    parser.add_argument("resolution", type=int, nargs=1,
                        help="The number of cells in each direction on the mesh.")
    parser.add_argument("degree", type=int, nargs=1,
                        help="The degree of the polynomial basis for the function space.")
    args = parser.parse_args()
    resolution = args.resolution[0]
    degree = args.degree[0]
    analytic = args.analytic
    plot_error = args.error

    u, error = solve_poisson(degree, resolution, analytic, plot_error)

    u.plot()
