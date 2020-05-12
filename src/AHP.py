#!/usr/bin/env python

####### TODO:
# Make the return values for the EigenVector values easy to insert into a dataframe

import numpy as np

class analyic_hierarchy():
    """ Class to calculate the hierarchy of a Matrix of Ratios. Please Read Documentation for Details:
    Original Article: https://www.sciencedirect.com/science/article/pii/0270025587904738 by T. Saaty (1989)
    1. German Implementation of the process in decision making: https://link.springer.com/article/10.1007/s40275-014-0011-8
    2. More reasonable detailed explanation. https://people.revoledu.com/kardi/tutorial/AHP/AHP-Example.htm
    3. Detailed Explanation. https://www.pmi.org/learning/library/analytic-hierarchy-process-prioritize-projects-6608
    _____________________________________________

    Please be aware that this uses an approximation to calculate the Eigenvector, but provides other methods for more accurate definitions if necessary
    """
    # Random Consistency - acquired empirically - see usage above. CR should be >0.1
    RANDOMCONSISTENCY =  [0.00, 0.00, 0.52, 0.89, 1.11, 1.25, 1.35, 1.40, 1.45, 1.49]
    CONSISTENT = 0.10

    # Constructor
    def __init__(self, array):
        """ Constructor. Requires a Matrix.
        Returns nothing
        The normalised EigenVector values can be accessed via attribute eigVec"""
        self.array = array
        self.width = array.shape[0]
        self.length = array.shape[1]
        self.array = array.astype(dtype=np.single)   #to convert to float
        for i in range(self.length):
            for j in range(self.width):
                if (i==j):
                    self.array[i,j] = 1
                if (i>j):
                    self.array[i,j] = 1/self.array[j,i]
        self.eigVec, self.ColSums = self.SimplifiedEig()
        self.MaxEigenValue = np.sum(self.eigVec*self.ColSums)
        self.CI = self.ConsistencyIndex()
        self.consistency = self.ConsistencyRatio()
    
    # Boolean Consistency check
    def consistent(self):
        """Consistency Check. Returns a Bool to tell if the matrix is consistent enough.
        """ 
        if self.consistency <= self.CONSISTENT: 
            return True
        else: return False
        
    # could also be written as a lambda
    def ConsistencyIndex (self):
        """ Calculates how consistent the relationships are.
        Args: Principal eigenvalue eig, The sum of column elements of the concurrent Eigenvector.
        Returns: Consistency Index CI, to be used with Consistency Ratio
        """
        return (self.MaxEigenValue - self.width)/(self.width-1)
    
    # could be written as a lambda
    def ConsistencyRatio (self):
        """Calculates the Ratio of Consistency.
        Args: the CI calculated from the Eigenvector, and the empirical consistency.
        Returns: an index"""
        return (self.CI)/(self.RANDOMCONSISTENCY[self.width-1])
    
    # Approximation of the Eigenvector
    def SimplifiedEig(self):
        """Approximation of row Eigenvectors, According to literature this gets within 10% of the original value
        following: https://www.pmi.org/learning/library/analytic-hierarchy-process-prioritize-projects-6608
        Arg: A matrix from which to calculate these Eigenvectors
        Returns: the Eigenvector and the Column Sums"""
        sums = np.sum(self.array,axis=0)       #these are needed for the CI, col. eigenvectors
        newMat = self.array/sums
        finalVec = np.sum(newMat, axis=1)/newMat.shape[1]   # I do not think this is correct - please check!
        return finalVec, sums

    # Power Method for calculating the Eigenvector
    def ConvergingEig(self):
        """ Following the example of squaring the matrix, normalising and then calculating the EigenVector (Row Totals)
            Alternatively: square, calculate row totals and then normalise these - see if they change
            This is a so-called power-method.
            Args: A matrix
            Returns: the principal Eigenvector as well as the first column sums (for CI)
        """
        Prior = [0, 0, 0, 0]
        FirstSum = np.sum(self.array,axis=0)    #for calculating the maximum eigenvalue
        diff = 5        # random value
        while (diff>0.001):
            # Squaring the Matrix
            matrix = np.matmul(self.array, self.array)
            # Normalise - divide by col totals
            sums = np.sum(matrix,axis=0)
            matrix = matrix/sums
            # Row totals - local
            RowTot = np.sum(matrix, axis=1)/length
            # To calculate whether the different changed majorly
            diff = np.subtract(RowTot, Prior)
            Prior = RowTot
            diff = np.linalg.norm(diff)
        return RowTot, FirstSum          # RowTot is the Eigenvector

    # Real calculation of the Eigenstuff, 
    def ComplicatedEig(self):
        """Correct calculation of the principal Eigenvector using the numpy library.
        Args: A matrix
        Returns: The principal eigenvector, as well as the eigenvalues"""
        val, vec = np.linalg.eig(np.array(self.array))
        vec = vec/np.sum(vec)
        maxvalidx = np.argmax(val)
        EigVal = val[maxvalidx]
        EigVec = vec[:,maxvalidx]
        return EigVec, EigVal