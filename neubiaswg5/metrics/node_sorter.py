import numpy as np

def findchildren(swc, parent_id):
    while np.argwhere(swc[:,6] == parent_id).size != 0: 
        direct_children_indices = np.argwhere(swc[:,6] == parent_id) 
        
        if len(direct_children_indices) > 1:
            try:
                children_nodes
            except NameError:
                # if first time running and children_nodes does not exist
                children_nodes = direct_children_indices
            else:
                # when children_nodes already exist
                children_nodes = np.concatenate((children_nodes, direct_children_indices))
            
            try:
                visited_node
            except NameError:
                # if first time running and visited node does not exist
                visited_node = np.argwhere(swc[:,0] == parent_id)
            else: 
                visited_node = np.concatenate((visited_node, np.argwhere(swc[:,0] == parent_id)))
            # new parent id from the list of children
            parent_id = swc[children_nodes[-1],0]

        elif len(direct_children_indices) == 1:        
            try:
                children_nodes
            except NameError:
                # if first time running and children_nodes does not exist
                children_nodes = direct_children_indices
            else:
                # when children_nodes already exist
                children_nodes = np.concatenate((children_nodes, direct_children_indices))
            
            try:
                visited_node
            except NameError:
                # if first time running and visited node does not exist
                visited_node = np.argwhere(swc[:,0] == parent_id)
            else: 
                visited_node = np.concatenate((visited_node, np.argwhere(swc[:,0] == parent_id)))          
            # new parent id from the list of children (last idx)    
            parent_id = swc[children_nodes[-1],0]
    try:
        visited_node
    except NameError:
        # if first time running and visited node does not exist
        visited_node = np.argwhere(swc[:,0] == parent_id)
    else:
        visited_node = np.concatenate((visited_node, np.argwhere(swc[:,0] == parent_id)))
    
    try: 
        children_nodes
    except NameError:
        unvisited_nodes = np.empty(shape = (0,0))
        children_nodes = np.empty(shape = (0,0))
    else:
        unvisited_nodes = children_nodes[np.isin(children_nodes, visited_node, invert = True)]
    
    # print('visited nodes: ')
    # print(visited_node)
    # print('children nodes: ')
    # print(children_nodes)
    # print('unvisited_nodes: ')
    # print(unvisited_nodes)
    # print(unvisited_nodes.shape)

    if unvisited_nodes.size != 0:
        for i in range(0,unvisited_nodes.size):
            # print('first unvisited node to check is ' + str(unvisited_nodes[i]))
            new_children_nodes = findchildren(swc,swc[unvisited_nodes[i],0])

            # print('new children nodes: ')
            # print(new_children_nodes)
            # print(new_children_nodes.size)
            # print(new_children_nodes.shape)
            # print('old children nodes: ')
            # print(children_nodes)
            # print(children_nodes.size)
            # print(children_nodes.shape)
            
            if new_children_nodes.size != 0:
                children_nodes = np.concatenate((children_nodes, new_children_nodes))
            
            # print('added children nodes (old + new)')
            # print(children_nodes)

    return children_nodes



def swc_node_sorter(swc_file_path):

    swc = np.loadtxt(swc_file_path)
    new_swc = np.empty(swc.shape)

    # some worflow outputs, such as the one from rivuletpy, have to be fixed
    # when the node id is its own parent, change parent id to -1
    nrows , ncols = swc.shape
    for row in range(0,nrows):
        if swc[row,0] == swc[row,6]:
            swc[row,6] = -1

    first_parents = np.isin(swc[:,6],swc[:,0],invert = True)
    first_parents = np.logical_or(first_parents,swc[:,6]==-1)
    parent_indices = np.argwhere(first_parents == 1)
    row_counter = 0

    for parent_idx in parent_indices:
        # get parent id 
        parent_id = swc[parent_idx,0]
        print('parent_id to check for children is '+ str(parent_id))
        # call findchildren to get the list of indices of all children of parent_id 
        children_idx_list = findchildren(swc,parent_id)
        print(children_idx_list)
        print('children idx list shape')
        print(children_idx_list.shape)
        print('swc shape ')
        nrows, ncols = swc.shape
        print(swc.shape)

        for i in range(0,len(children_idx_list)):
            if i == 0:
                new_swc[row_counter,:] = swc[parent_idx,:]
                new_swc[row_counter+1,:] = swc[children_idx_list[i],:]
                row_counter += 2
            else:
                new_swc[row_counter,:] = swc[children_idx_list[i],:]
                row_counter += 1
    print(new_swc)
    np.savetxt(swc_file_path, new_swc, fmt='%i %i %.2f %.2f %.2f %.2f %i', delimiter=' ')


