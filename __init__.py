import numpy as n
import scipy.ndimage as sn

# scipy ndimage label on a chunk
# identify boundary points
# associate boundary points with connected components within chunk
# unionfind merge connected components across chunks 


def remap(indices,offset,shape0,shape1):
    # unravel_index gets i,j,k coords within array of shape0
    # then adds the offset
    # and then ravels according to position in array of shape1
    return n.ravel_multi_index(n.array(n.unravel_index(indices,shape0)) + offset[:,None],shape1)


def boundaries(array,offset,globalshape,structure=None):
    result = n.zeros([0,2],dtype=n.int64)
    for axis in range(array.ndim):
        for side in [0,array.shape[axis]-1]:
            select = [slice(None)]*array.ndim
            newaxis = [slice(None)]*array.ndim
            select[axis] = side
            newaxis[axis] = None
            face = array[tuple(select)][tuple(newaxis)]
            newoffset = offset.copy()
            newoffset[axis] += side

            lbc = link_boundary(face,structure=structure)
            lbc[:,0] = remap(lbc[:,0],newoffset,face.shape,globalshape) #indices 
            result = n.concatenate((result,lbc))
    # Nx2 array of [Corrected Index, Chunk Object Index] pairs
    return result
            
def link_boundary(array,structure=None):
    '''
    Takes an array, assumed to be a boundary slice of an output from scipy.ndimage.label
    Finds contiguous regions of the boundary and associates with the relevant connected component
    '''
    label,num = sn.label(array,structure=structure)
    if not num:
        return n.zeros([0,2],dtype=n.int64)
    slc = sn.labeled_comprehension(array,label,range(1,num+1),
                                   lambda a,b: n.array([n.min(b),a[0]]),
                                   list,
                                   None,
                                   pass_positions=True)
    slc = n.stack(slc)
    return n.array(slc).reshape(-1,2)

def process(mask,offset,globalshape,structure=None):
    offset = n.array(offset,dtype=n.int64)
    def subremap(indices):
        return n.min(remap(indices,offset,mask.shape,globalshape))

    label,num_clouds = sn.label(mask,structure=structure)
    if not num_clouds:
        slc = []
        ids = n.array([])
        boundary_list = n.zeros([0,2],dtype=n.int64)
        return slc,ids,boundary_list
    slc = sn.labeled_comprehension(mask,label,range(1,num_clouds+1),
                                   lambda a,b: b,
                                   list,
                                   None,
                                   pass_positions=True)

    ids = n.array([subremap(indices) for indices in slc])

    boundary_list = boundaries(label,offset,globalshape,structure=structure)
    boundary_list[:,1] = ids[boundary_list[:,1]-1]

    return slc,ids,boundary_list

# Boundary correction


def merge_tuples_unionfind(tuples):
    # use classic algorithms union find with path compression
    # https://en.wikipedia.org/wiki/Disjoint-set_data_structure
    parent_dict = {}

    def subfind(x):
        # update roots while visiting parents 

        # if parent of x is not x 
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

    # for unique parents, subfind the root, replace occurrences with root
    vs = set(parent_dict.values())
    for parent in vs:
        sp = subfind(parent)
        if sp != parent:
            for key in parent_dict:
                if parent_dict[key] == parent:
                    parent_dict[key] = sp

    return parent_dict

def numpy_get_indices(

def combine_array(parent_dict,ids,array):
    '''
    Assuming array of [NxM] for N objects and M properties
    Linearly add M properties for connected objects through parent_dict, output of merge_tuples_unionfind
    '''
    ids = list(ids)
    newarray = array.copy()
    # prepare mask and idsindex dict
    mask = n.ones(len(ids),dtype=bool)
    idsindex = {}

    # only compute on objects with parents which are also in ids 
    object_ids = set(parent_dict.keys()).intersection(set(ids))



    for object_id in object_ids:
        index = ids.index(object_id)
        idsindex[object_id] = index

    for object_id in object_ids:
        index = idsindex[object_id]
        # sets are sorted so index of parent should already be found previously in loop
        if parent_dict[object_id] != object_id:
            mask[index] = False
            parent_index = idsindex[parent_dict[object_id]]
            newarray[parent_index] += newarray[index]
    return newarray[mask]

            

