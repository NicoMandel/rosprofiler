#!/usr/bin/env python

####### TODO:
# Make the return values for the EigenVector values easy to insert into a dataframe

import numpy as np
import pandas as pd

class ahp_mat:

    RANDOMCONSISTENCY =  [0.00, 0.00, 0.52, 0.89, 1.11, 1.25, 1.35, 1.40, 1.45, 1.49]
    CONSISTENT = 0.10
    poss = None

    def __init__(self, arr, collist=None, name=None):
        """
            A class to calculate an ahp matrix.
            Params: a (right-triangular) matrix filled in with the appropriate values
        """
        tril = arr[np.tril_indices_from(arr,k=-1)]
        tril0 = np.where(tril==1.0,0.0,tril)
        if not np.all(tril0):
            arr = self.filtrilaltalt(arr)
        self.name = name
        self.df = pd.DataFrame(data=arr,index=collist,columns=collist)
        # Init of values that get set later on
        self.ci = None
        self.eigdf = None
        self.consratio = None
        self.geteig()
        self.getci()

    def __mul__(self, other):
        """
            Multiplication override. Allows for multiplication with another ahp object
        """

        return self.eigdf.to_numpy()*other.eigdf.to_numpy()

    __rmul__= __mul__


    def getRelWeights(self):
        """
            method to return the relative weights. Returns a dataframe
        """
        return self.eigdf

    def geteig(self):
        """
            Method to set the eigenvectors/values neccessary
        """
        val, vec = np.linalg.eig(np.array(self.df.values))
        # vec = vec/np.sum(vec) # Normalising the Eigenvector
        self.eigVal = val[np.argmax(val)].real
        eigVec = vec[:,np.argmax(val)].real
        eigVec = eigVec / np.sum(eigVec)
        self.eigdf = pd.DataFrame(data=eigVec,index=self.df.columns.tolist(),columns=["Relative Weight"])
        

    def getci(self):
        """
            Method to calculate the consistency index using the max eigenvalue 
        """
        self.ci = (self.eigVal - self.df.shape[0]) / (self.df.shape[0] - 1)
        self.consratio = self.ci / self.RANDOMCONSISTENCY[self.df.shape[0]]

    def getconsistency(self):
        """
            A method to calculate the consistency index.
            Boolean. Returns true if the array is deemed consistent. False if not 
        """
        if self.consratio < self.CONSISTENT:
            return True
        else:
            return False

    @classmethod
    def setconsistent(cls, val):
        """
            A setter method for the setting the consistency value across ALL members
        """
        if not (val < 1.0) and not (val > 0.0):
            raise ValueError("Not  correct type. Needs to be float between 0 and 1, is {}".format(
                    val
                    ))
        else:
            cls.CONSISTENT = val
            return True

    @classmethod
    def initrand(cls):
        """
            Function to initialize the random variables for the randarray to pick from
        """

        if cls.poss is not None:
            print("Already Initialized.")
        else:
            vals = np.arange(start=1, stop=10)
            invvals = 1/vals
            preposs = np.concatenate((vals, invvals), axis=0)
            cls.poss = np.unique(preposs)

    @classmethod
    def getrandarray(cls, n=3):
        """
            Static method.
            Returns a random array of size n conforming to the standards of the AHP 
        """

        if cls.poss is None:
            cls.initrand()
        arr = np.ones((n,n), dtype=np.float64)
        ind = np.triu_indices_from(arr,1)
        arr[ind] = np.random.choice(cls.poss,size=ind[0].size)
        # arr = cls.filtrilaltalt(arr)
        return arr

    @classmethod
    def relpercentage(cls, val1, val2):
        """
            Method to calculate the relative weight using the difference in percentage. Using the formula y=8x+1
            Which comes from linearly scaling the performance
        """

        sub = val1-val2
        if sub == 0.0:
            return 1.0
        elif sub > 0.0:
            y = cls.scalefunc(sub)
            return y
        else:
            # Use the inverse
            sub = np.abs(sub)
            y = cls.scalefunc(sub)
            return 1.0/y
    
    # Class method to return the function y=8x+1 - rounded to nearest integer
    scalefunc = classmethod(lambda cls, x: np.round(8.0*x+1.0))

    @staticmethod
    def filtril(arr):
        """
            Static method.
            Returns the lower triangular matrix of the array filled in conforming to the standards of the AHP
        """
        arr[np.tril_indices_from(arr,-1)] = 1.0/arr[np.triu_indices_from(arr,1)]
        return arr
    
    @staticmethod
    def filtrilalt(arr):
        """
            Static alternative to filling in the lower indices of an array
        """
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if i>=j:
                    continue
                else:
                    arr[j,i] = 1.0/arr[i,j]
        return arr

    @staticmethod
    def filtrilaltalt(arr):
        """
            Third alternative to fill the lower triangular matrix. More numpy-esque
        """

        inds = np.triu_indices_from(arr,k=1)
        arr[(inds[1], inds[0])] = 1.0/arr[inds]
        return arr

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
