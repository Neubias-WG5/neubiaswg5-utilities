import numpy as np
from skan import csr
#Note: requires bleeding edge version of Skan, install with: pip install git+https://github.com/jni/skan

def skl2obj(Skl_np,smp,ZRatio,OBJToExport):

    # Analyze skeleton
    branch_data = csr.summarise(Skl_np)
    NBranches = branch_data.shape[0]
    skel_obj = csr.Skeleton(Skl_np)
    Brch_vox = skel_obj.path_coordinates
    NSkeletons = np.unique(branch_data['skeleton-id']).shape[0]

    # Extract required information on skeleton branches
    #NBranches = branch_data.shape[0]
    brclist = skel_obj.paths_list()
    NBranches = len(brclist)
    id_0 = np.zeros(NBranches,dtype=int)
    id_1 = np.zeros(NBranches,dtype=int)
    for i in range(NBranches):
        id_0[i] = brclist[i][0]
        id_1[i] = brclist[i][-1]

    # Parse all nodes to find unique vertices
    AllNodes = (np.unique([id_0,id_1]))
    MaxVertexIdx = (np.amax(AllNodes))
    NVertices = AllNodes.size

    # Over-estimate of total number of segments after skeleton sampling (node to node links)
    TotSegments = 1
    for i in range(NBranches):
        TotSegments = TotSegments + (1+np.floor(Brch_vox(i).shape[0]/smp)).astype(int)

    # Display model information
    #print("Number of skeletons: %i"%NSkeletons)
    #print("Number of branches: %i"%NBranches)
    #print("Number of nodes: %i"%NVertices)
    #print("Estimated number of segments: %i"%TotSegments)

    # Build re-indexing LUT
    LUTVertices = np.zeros([MaxVertexIdx+1,1],dtype=int)
    for i in range(NVertices):
        LUTVertices[AllNodes[i]] = i

    # Fill OBJ v-data
    OBJ_Vdata = np.zeros([TotSegments,3],dtype=int)
    for i in range(NBranches):
        OBJ_Vdata[LUTVertices[id_0[i]],0] = Brch_vox(i)[0,2]
        OBJ_Vdata[LUTVertices[id_0[i]],1] = Brch_vox(i)[0,1]
        OBJ_Vdata[LUTVertices[id_0[i]],2] = Brch_vox(i)[0,0]*ZRatio
        OBJ_Vdata[LUTVertices[id_1[i]],0] = Brch_vox(i)[-1,2]
        OBJ_Vdata[LUTVertices[id_1[i]],1] = Brch_vox(i)[-1,1]
        OBJ_Vdata[LUTVertices[id_1[i]],2] = Brch_vox(i)[-1,0]*ZRatio
    cntVertices = NVertices

    # Fill OBJ l-data
    OBJ_Ldata = np.ones((TotSegments,2),dtype=int)
    cntSegments = 0
    for i in range(NBranches):
        Vox = Brch_vox(i)
        L = Vox.shape[0]
        nNodes = (1+np.floor(L/smp)).astype(int)
        PrevNode = LUTVertices[id_0[i]]+1
        for s in range(1,nNodes-1):
            OBJ_Vdata[cntVertices,0] = Vox[np.round(s*(L-1)/(nNodes-1)).astype(int),2]
            OBJ_Vdata[cntVertices,1] = Vox[np.round(s*(L-1)/(nNodes-1)).astype(int),1]
            OBJ_Vdata[cntVertices,2] = Vox[np.round(s*(L-1)/(nNodes-1)).astype(int),0]*ZRatio
            OBJ_Ldata[cntSegments,0] = PrevNode
            OBJ_Ldata[cntSegments,1] = cntVertices+1
            PrevNode = cntVertices+1
            cntVertices = cntVertices + 1
            cntSegments = cntSegments + 1
        OBJ_Ldata[cntSegments,0] = PrevNode
        OBJ_Ldata[cntSegments,1] = LUTVertices[id_1[i]]+1
        cntSegments = cntSegments + 1

    # Display model statistics
    #print("Actual number of vertices: %i" %cntVertices)
    #print("Actual number of segments: %i" %cntSegments)

    with open(OBJToExport, "w") as f:
        f.write("")
    f.close()
    with open(OBJToExport, "a") as f:
        for i in range(cntVertices):
            f.write("v %i %i %i\n" %(OBJ_Vdata[i,0],OBJ_Vdata[i,1],OBJ_Vdata[i,2]))
    f.close()
    with open(OBJToExport, "a") as f:
        for i in range(cntSegments):
            f.write("l %i %i\n" %(OBJ_Ldata[i,0],OBJ_Ldata[i,1]))
    f.close()
