import numpy
import scipy.ndimage

default_structure = scipy.ndimage.generate_binary_structure(3,3)

'''
Utilities for running scipy.ndimage.label connected components algorithm on local domains and combining results, 
Useful for parallelization or data too large to fit in RAM
Assumes goal is to save as little data as possible per connected component, 
So focuses on reduced output and does not attempt to merge lists of cells

I use these with mpi4py and/or h5py which are python libraries which use MPI and HDF5. 
HDF5 is the data format of our terabyte-scale simulations and 
provides an easy interface to efficiently read local domains with numpy array indexing syntax. 
mpi4py is a message passing interface which allows communication between multiple processes. 
This approach attempts to minimize the necessary communications.

Algorithm overview:
scipy ndimage label on a local chunk (or domain)
identify connected components passing through boundary
identify 1 point per boundary connected component (to reduce data)
each boundary point exists in multiple domains, as member of subsections of a single connected component
union-find merge connected components across chunks 

add together linear properties (anything that is a sum over voxels is eligible) to save single number per connected component



'''

def remap(indices,offset,local_shape,global_shape):
    '''
    Remaps flattened indices of local_shape to an array of global_shape with offset
    This function is necessary to remap flattened indices of a local array (smaller local_shape) 
    to their corresponding flattened indices in the global array (larger global_shape)
    taking the position of the local array into account (offset)
    '''
    # use unravel_index to get i,j,k coordinates within array of local_shape
    # then adds the offset to each point using numpy broadcasting
    # and then ravels according to position in array of global_shape
    return numpy.ravel_multi_index(numpy.array(numpy.unravel_index(indices,local_shape)) + offset[:,None],global_shape)


def boundaries(array,offset,global_shape,structure=default_structure):
    '''
    For each 2-D boundary face, produce tuples which represent connections between objects and min remapped index of object on face.
    The same face will appear in two local domains with the same min remapped indices for objects but different object identifiers.
    So these pairs (links) will enable object identifiers connected across domains to be connected. 
    '''
    result = numpy.zeros([0,2],dtype=numpy.int64)
    for axis in range(array.ndim):
        for side in [0,array.shape[axis]-1]:
            # extract boundary layer on axis,side as a 3-D subarray with its own offset position within the global array
            select = [slice(None)]*array.ndim
            newaxis = [slice(None)]*array.ndim
            select[axis] = side
            newaxis[axis] = None
            face = array[tuple(select)][tuple(newaxis)]
            newoffset = offset.copy()
            newoffset[axis] += side

            # produce links from face
            links = link_boundary(face,structure=structure)
            links[:,0] = remap(links[:,0],newoffset,face.shape,global_shape) # min remapped indices of objects on face
            result = numpy.concatenate((result,links))
    # Nx2 array of [Remapped Min Index, Chunk Object Index] pairs
    return result
            
def link_boundary(array,structure=default_structure):
    '''
    Takes an array, assumed to be a boundary slice of an output from scipy.ndimage.label
    Finds contiguous regions of the boundary and associates with the relevant connected component
    result[:,0] is minimum linear/flattened index for each object in the boundary slice
    result[:,1] is the labelled array label for each object in the boundary slice
    '''
    label, num_features = scipy.ndimage.label(array,structure=structure)
    if not num_features:
        return numpy.zeros([0,2],dtype=numpy.int64)

    # note: min commutes with remap: min(remap(indices)) = remap(min(indices)) so remap later 
    slc = scipy.ndimage.labeled_comprehension(array,label,range(1,num_features+1),
                                              lambda labeled_array,linear_indices: numpy.array([numpy.min(linear_indices),labeled_array[0]]),
                                              list,
                                              None,
                                              pass_positions=True)
    slc = numpy.stack(slc)

    return numpy.array(slc).reshape(-1,2)

def process(mask,offset,global_shape,structure=default_structure):
    '''
    mask : 3-D Numpy array input for scipy.ndimage.label representing a sub-domain
    offset : 3-element arraylike representing sub-domain offset (global indices of local [0,0,0])
    global_shape : 3-element arraylike shape of global domain
    structure : scipy ndimage binary structure

    Returns
    slc : Numpy object array of integer arrays, each integer array is flattened local domain indices of a connected component
    ids : array of connected component ids from perspective of global domain
    boundary_list : N x 2 array representing N pairs for connecting components across domains
    '''
    offset = numpy.array(offset,dtype=numpy.int64)
    def subremap(indices):
        return numpy.min(remap(indices,offset,mask.shape,global_shape))

    label,num_clouds = scipy.ndimage.label(mask,structure=structure)
    if num_clouds == 0:
        slc = []
        ids = numpy.array([])
        boundary_list = numpy.zeros([0,2],dtype=numpy.int64)
        return slc,ids,boundary_list
    
    slc = scipy.ndimage.labeled_comprehension(mask,label,range(1,num_clouds+1),
                                              lambda a,linear_indices: linear_indices,
                                              list,
                                              None,
                                              pass_positions=True)

    ids = numpy.array([subremap(indices) for indices in slc])

    boundary_list = boundaries(label,offset,global_shape,structure=structure)
    boundary_list[:,1] = ids[boundary_list[:,1]-1]
    # by this step, both columns of boundary_list have been remapped to global indices
    # and ids have been remapped to global indices
    return slc,ids,boundary_list

# Boundary correction

def merge_tuples_unionfind(tuples):
    '''
    Classic algorithms union find with path compression
    https://en.wikipedia.org/wiki/Disjoint-set_data_structure
    
    tuples : a list of pairs of indices

    Returns
    parent_dict : dictionary[index] = parent index
    '''
    parent_dict = {}

    def subfind(x):
        # update roots while visiting parents 
        if parent_dict[x] != x:
            # update parent of x by finding parent of parent 
            # until the progenitor is reached parent_dict[progenitor] = progenitor
            parent_dict[x] = subfind(parent_dict[x])
        return parent_dict[x]

    def find(x):
        if x not in parent_dict:
            # x forms new set and becomes a root
            parent_dict[x] = x
            return x
        if parent_dict[x] != x:
            # follow chain of parents of parents to find root 
            parent_dict[x] = subfind(parent_dict[x])
        return parent_dict[x]

    # each tuple represents a connection between two items 
    # so merge them by setting root to be the lower root. 
    for p0,p1 in list(tuples):
        r0 = find(p0)
        r1 = find(p1)
        if r0 < r1:
            parent_dict[r1] = r0
        elif r1 < r0:
            parent_dict[r0] = r1

    # final cleanup
    for key in parent_dict:
        find(key)

    return parent_dict            

def combine_array(parent_dict,ids,array):
    '''
    Merge object field values

    parent_dict: the dictionary result of union-find to identify parents of objects by id, merging objects into parents
    ids: (num_unmerged_objects) NumPy array of object ids
    array: (num_unmerged_objects x num_fields) NumPy array of field values, assumed to be summable

    Returns
    new_array: (num_merged_objects x num_fields) NumPy array of field values, assumed to be summable
    uni : (num_merged_objects) NumPy array of object ids, unique new ids
    '''
    # for each line of array data, the ultimate id will be its parent's id
    new_ids = [parent_dict[x] if x in parent_dict else x for x in ids]
    # sorted list of ids
    uni = numpy.unique(new_ids)
    # new indices for bincount
    suni = numpy.searchsorted(uni,new_ids)
    # assert numpy.all([numpy.all(suni[new_ids == uni[i]] == i) for i in range(len(uni))]) 
    # suni = i whenever new_ids = uni[i] so suni is reduced indices
    new_array = numpy.zeros([len(uni),len(array[0])])
    
    for i in range(len(array[0])):
        weight = array[:,i].astype(float)
        new_array[:,i] = numpy.bincount(suni,weights=weight)
        
    return new_array,uni

def make_block_selection(x,y,z,size,global_shape):
    '''
    x,y,z : integers representing 3-D index of block
    size : integer size of block, number of elements in each dimension
    global_shape : 3-element arraylike shape of global domain

    Returns:
    select : slicing tuple to access local domain via global_array[select]
    offset : 3-element arraylike representing sub-domain offset (global indices of local [0,0,0])
    weight : array to ignore extra boundary values by weighing by 0
    
    '''
    xmin = x*size
    xmax = min((x+1)*size+1,global_shape[0])
    ymin = y*size
    ymax = min((y+1)*size+1,global_shape[1])
    zmin = z*size
    zmax = min((z+1)*size+1,global_shape[2])
    select = tuple((slice(xmin,xmax),slice(ymin,ymax),slice(zmin,zmax)))
    offset = numpy.array([xmin,ymin,zmin])

    local_shape = ((xmax-xmin),(ymax-ymin),(zmax-zmin))

    # Ignore extra boundary values when calculating sum by weighing by 0
    weight = numpy.ones(local_shape)
    maxs = [xmax,ymax,zmax]
    for dim in range(len(local_shape)):
        tempslice = [slice(None)]*len(local_shape)
        tempslice[dim] = local_shape[dim] - 1
        if maxs[dim] < global_shape[dim]:
            # for the very last chunk, when maxs[i] == shape[i], keep the weight
            weight[tuple(tempslice)] = 0.0
    weight = weight.reshape(-1)
    
    return select,offset,weight
    

def example(size):
    '''
    Example of 2x2x2 domain decomposition, 
    which generates a random mask field and random value field, 
    finds connected components, and compares the array of summed values over objects
    between the decomposed approach and the global approach
    '''

    global_shape = (2*size,2*size,2*size)

    mask = numpy.random.randint(0,2,size=global_shape)
    values = numpy.random.random(size=global_shape)

    # First calculate with domain decomposition
    blocks = [(i,j,k) for i in range(2) for j in range(2) for k in range(2)]

    total_sum_array = numpy.zeros([0,1])
    total_ids_array = numpy.zeros([0,1],dtype=numpy.int64)
    total_boundary_list = numpy.zeros([0,2],dtype=numpy.int64)
    
    for x,y,z in blocks:
        # select local array with 1 overlap face in each direction

        select,offset,weight = make_block_selection(x,y,z,size,global_shape)
        
        slc,ids,boundary_list = process(mask[select],offset,global_shape)
        
        # Prepare flattened values
        flat_values = values[select].reshape(-1) * weight

        sum_array = numpy.zeros([len(slc),1])        
        sum_array[:,0] = numpy.array([numpy.sum(flat_values[feature]) for feature in slc])
        ids_array = numpy.zeros([len(slc),1],dtype=numpy.int64)
        ids_array[:,0] = ids
        
        total_sum_array = numpy.concatenate((total_sum_array,sum_array))
        total_ids_array = numpy.concatenate((total_ids_array,ids_array))
        total_boundary_list = numpy.concatenate((total_boundary_list,boundary_list))

    parent_dict = merge_tuples_unionfind(total_boundary_list)
    block_array,block_ids = combine_array(parent_dict,total_ids_array.reshape(-1),total_sum_array)

    # Next calculate as a whole
    label,num_clouds = scipy.ndimage.label(mask,structure=default_structure)    
    global_slc = scipy.ndimage.labeled_comprehension(mask,label,range(1,num_clouds+1),
                                              lambda a,linear_indices: linear_indices,
                                              list,
                                              None,
                                              pass_positions=True)
    global_flat_values = values.reshape(-1)
    global_array = numpy.array([numpy.sum(global_flat_values[feature]) for feature in global_slc])
    global_array = global_array.reshape(-1,1)

    # Floating point addition is not associative or commutative so the two arrays can only be close
    assert numpy.all(numpy.isclose(block_array,global_array))

    # The block method gives objects in order of their minimum flattened cell index since the union-find step
    # always takes the minimum to be the parent
    # if scipy.ndimage.label is implemented in the same way, the arrays should match since the order of objects should match
    # otherwise, need to compare their sorted form
    
    return block_array,global_array
