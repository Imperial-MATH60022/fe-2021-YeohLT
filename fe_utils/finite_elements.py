# Cause division to always mean floating point division.
from __future__ import division
import numpy as np
from .reference_elements import ReferenceInterval, ReferenceTriangle
np.seterr(invalid='ignore', divide='ignore')


def lagrange_points(cell, degree):
    """Construct the locations of the equispaced Lagrange nodes on cell.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct nodes.

    :returns: a rank 2 :class:`~numpy.array` whose rows are the
        coordinates of the nodes.

    The implementation of this function is left as an :ref:`exercise
    <ex-lagrange-points>`.

    """

    d = cell.dim
    p = degree

    if d == 1:
        numpts = p+1
        lagpts = np.zeros((numpts,1))
        for i in range (0,numpts):
            lagpts[i] = i/(numpts-1)
    else:
        numpts = ((p+2)*(p+1))//2
        lagpts = np.zeros((numpts,2))
        currpt = 0
        for i in range(0, p+1):
            for j in range(0, p+1-i):
                lagpts[currpt,:] = [j/p,i/p]
                currpt += 1

    return lagpts


def vandermonde_matrix(cell, degree, points, grad=False):
    """Construct the generalised Vandermonde matrix for polynomials of the
    specified degree on the cell provided.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct the matrix.
    :param points: a list of coordinate tuples corresponding to the points.
    :param grad: whether to evaluate the Vandermonde matrix or its gradient.

    :returns: the generalised :ref:`Vandermonde matrix <sec-vandermonde>`

    The implementation of this function is left as an :ref:`exercise
    <ex-vandermonde>`.
    """
    from math import factorial as mfact
    numcol = mfact(degree+cell.dim)//mfact(degree)//mfact(cell.dim)
    m = points.shape[0]

    if grad == False:
        Vmat = np.zeros((m,numcol))
    
        #do col by col
        if cell.dim == 1:
            for col in range(0,numcol):
                for r in range(0,m):
                    xval = points[r,0]
                    Vmat[r,col] = xval**col
        else:   
            colnum = 0
            for p in range(0,degree+1):
                #y has degree q
                for q in range(0,p+1):
                    #r for row
                    for r in range(0,m):
                        xval = points[r,0]
                        yval = points[r,1]
                        Vmat[r,colnum] = (yval**q)*(xval**(p-q))
                    colnum += 1
    else:
        Vmat = np.zeros((m,numcol,cell.dim))
        #do col by col
        if cell.dim == 1:
            for col in range(1,numcol):
                for r in range(0,m):
                    xval = points[r,0]
                    Vmat[r,col,0] = col*xval**(col-1)
        else:   
            if numcol>1:               
                #r for row
                for r in range(0,m):
                    colnum = 1
                    xval = points[r,0]
                    yval = points[r,1]

                    for p in range(1,degree+1):
                        print(p)
                        #y has degree q
                        #case q=0
                        gradvec = [p*xval**(p-1),0]
                        Vmat[r,colnum,:] = np.nan_to_num(gradvec,nan=0.0)
                        colnum += 1

                        #case q = 1 to (p-1)
                        for q in range(1,p):
                            gradvec = [(p-q)*(yval**q)*(xval**(p-q-1)),q*(yval**(q-1))*(xval**(p-q))]
                            Vmat[r,colnum,:] = np.nan_to_num(gradvec,nan=0.0)
                            colnum += 1

                        #case q = p
                        gradvec = [0,p*yval**(p-1)]
                        Vmat[r,colnum,:] = np.nan_to_num(gradvec,nan=0.0)
                        colnum += 1

    return Vmat


class FiniteElement(object):
    def __init__(self, cell, degree, nodes, entity_nodes=None):
        """A finite element defined over cell.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.
        :param nodes: a list of coordinate tuples corresponding to
            the nodes of the element.
        :param entity_nodes: a dictionary of dictionaries such that
            entity_nodes[d][i] is the list of nodes associated with entity `(d, i)`.

        Most of the implementation of this class is left as exercises.
        """

        #: The :class:`~.reference_elements.ReferenceCell`
        #: over which the element is defined.
        self.cell = cell
        #: The polynomial degree of the element. We assume the element
        #: spans the complete polynomial space.
        self.degree = degree
        #: The list of coordinate tuples corresponding to the nodes of
        #: the element.
        self.nodes = nodes
        #: A dictionary of dictionaries such that ``entity_nodes[d][i]``
        #: is the list of nodes associated with entity `(d, i)`.
        self.entity_nodes = entity_nodes

        if entity_nodes:
            #: ``nodes_per_entity[d]`` is the number of entities
            #: associated with an entity of dimension d.
            self.nodes_per_entity = np.array([len(entity_nodes[d][0])
                                              for d in range(cell.dim+1)])

        # Replace this exception with some code which sets
        # self.basis_coefs
        # to an array of polynomial coefficients defining the basis functions.

        Vmat = vandermonde_matrix(cell,degree,nodes)
        self.basis_coefs = np.linalg.inv(Vmat)

        #: The number of nodes in this element.
        self.node_count = nodes.shape[0]

    def tabulate(self, points, grad=False):
        """Evaluate the basis functions of this finite element at the points
        provided.

        :param points: a list of coordinate tuples at which to
            tabulate the basis.
        :param grad: whether to return the tabulation of the basis or the
            tabulation of the gradient of the basis.

        :result: an array containing the value of each basis function
            at each point. If `grad` is `True`, the gradient vector of
            each basis vector at each point is returned as a rank 3
            array. The shape of the array is (points, nodes) if
            ``grad`` is ``False`` and (points, nodes, dim) if ``grad``
            is ``True``.

        The implementation of this method is left as an :ref:`exercise
        <ex-tabulate>`.

        """
        
        Vmat = vandermonde_matrix(self.cell,self.degree,points,grad)
        if grad == False:
            Amat = np.matmul(Vmat,self.basis_coefs)
        else:
            Amat = np.einsum("ijk,jl->ilk",Vmat,self.basis_coefs)
        
        return Amat

        #raise NotImplementedError

    def interpolate(self, fn):
        """Interpolate fn onto this finite element by evaluating it
        at each of the nodes.

        :param fn: A function ``fn(X)`` which takes a coordinate
           vector and returns a scalar value.

        :returns: A vector containing the value of ``fn`` at each node
           of this element.

        The implementation of this method is left as an :ref:`exercise
        <ex-interpolate>`.

        """
        m = self.nodes.shape[0]
        fvec = np.zeros(m)

        for p in range(0,m):
            fvec[p] = fn(self.nodes[p,:])
        
        return fvec

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.cell,
                               self.degree)


class LagrangeElement(FiniteElement):
    def __init__(self, cell, degree):
        """An equispaced Lagrange finite element.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.

        The implementation of this class is left as an :ref:`exercise
        <ex-lagrange-element>`.
        """

        nodes = lagrange_points(cell,degree)


        #raise NotImplementedError
        # Use lagrange_points to obtain the set of nodes.  Once you
        # have obtained nodes, the following line will call the
        # __init__ method on the FiniteElement class to set up the
        # basis coefficients.
        super(LagrangeElement, self).__init__(cell, degree, nodes)

        p = degree

        if cell.dim == 1:
            entnod = {0: {0: [0],
                          1: [p]},
                      1: {0: range(1,p)}}
        else:
            m = nodes.shape[0]        
            node00 = np.array([],dtype=int)
            node01 = node00
            node02 = node00
            node10 = node00
            node11 = node00
            node12 = node00
            node20 = node00

            for i in range(0,m):
                #ptvtx = False
                pt = nodes[i,:]
                
                if cell.point_in_entity(pt,[0,0]) == True:
                    node00 = np.append(node00,[i])

                elif cell.point_in_entity(pt,[0,1]) == True:
                    node01 = np.append(node01,[i])

                elif cell.point_in_entity(pt,[0,2]) == True:
                    node02 = np.append(node02,[i])

                elif cell.point_in_entity(pt,[1,0]) == True:
                    node10 = np.append(node10,[i])

                elif cell.point_in_entity(pt,[1,1]) == True:
                    node11 = np.append(node11,[i])

                elif cell.point_in_entity(pt,[1,2]) == True:
                    node12 = np.append(node12,[i])

                else:
                    node20 = np.append(node20,[i])

            entnod = {0: {0: node00,
                          1: node01,
                          2: node02},
                      1: {0: node10,
                          1: node11,
                          2: node12},
                      2: {0: node20}}
       
        self.entity_nodes = entnod
        print (entnod)