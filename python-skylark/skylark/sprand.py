'''
Created on Dec 6, 2012

@author: Vikas Sindhwani (vsindhw@us.ibm.com)
'''
import scipy.stats
import scipy.sparse

def sample(m, n, density, nz_values, nz_prob_dist):
    """
    sample(m, n, density, nz_values, nz_prob_dist)
    
    Parameters
    -----------
    m : number of rows
    n : number of columns
    density : density of sparse matrix (0 means all-zero sparse matrix, 1 means all-dense)
    nz_values : possible non-zero values of the parse  assume
    nz_prob_dist : discrete probability distrbution over nz_values 
    
    Returns
    ---------
    An m x n sparse matrix with desired density whose non-zero entries are independently sampled 
    from nz_values according to the distribution specified in nz_prob_dist
    """
    
    # generate a sparse matrix 
    S = scipy.sparse.rand(m, n, density, format = 'csr')

    # overwrite values with nzvals as given in nzdist
    S.data = scipy.stats.rv_discrete(values=(nz_values, nz_prob_dist), name = 'dist').rvs(size=S.nnz)
    
    return S
    
    
    
def hashmap(t, n, distribution, dimension=0): 
    """
    hashmap(t,n)
    
    Sparse matrix representation of a random hash map h:[n] -> [t] so that 
    for each i in [n], h(i) = j for j drawn from distribution
    
    Parameters
    -----------
    t : number of bins
    n : number of items hashed
    distribution : distribution object. Needs to implement the rvs(size=n) as returns
                   an array of n samples. 
    dimension : 0 returns t x n matrix, 1 returns n x t matrix (for efficiency 
                later)
    
    Returns
    -------
    If dimension=0, Sparse binary matrix S of size t x n such that  
    S(h(i), i) = random and all other entries are 0.
    
    If dimension=1, s is of size n x t with S(i,h(i)) = random and all other entries
    are 0.
    """
    
    data = distribution.rvs(size=n)
    col = scipy.arange(n)
    row = scipy.stats.randint(0,t).rvs(n)
    if dimension==0:
        S = scipy.sparse.csr_matrix( (data, (row, col)), shape = (t,n))
    else:
        S = scipy.sparse.csr_matrix( (data, (col, row)), shape = (n,t))
        
    return S
    
