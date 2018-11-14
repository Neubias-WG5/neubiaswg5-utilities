import tifffile as tiff
from skan import csr
import numpy as np
import argparse
from argparse import RawTextHelpFormatter

# Note: requires bleeding edge version of Skan, install with: pip install git+https://github.com/jni/skan
#

# Insert a new node (and associated branch) in the tree
def insertNodes(cntVert, PrevNode, Vox, smp, ZRatio):
    global SWC_data
    L = Vox.shape[0]
    nNodes = (1+np.floor(L/smp)).astype(int)
    for s in range(nNodes):
        if s>0:
            cntVert = cntVert+1
            SWC_data[cntVert,0] = cntVert
            if s == 1:
                SWC_data[cntVert,6] = PrevNode
            else:
                SWC_data[cntVert,6] = cntVert-1
            if nNodes>1:
                SWC_data[cntVert,2] = Vox[np.round(s*(L-1)/(nNodes-1)).astype(int),2]
                SWC_data[cntVert,3] = Vox[np.round(s*(L-1)/(nNodes-1)).astype(int),1]
                SWC_data[cntVert,4] = Vox[np.round(s*(L-1)/(nNodes-1)).astype(int),0]*ZRatio
            else:
                SWC_data[cntVert,2] = Vox[-1,2]
                SWC_data[cntVert,3] = Vox[-1,1]
                SWC_data[cntVert,4] = Vox[-1,0]*ZRatio
    return (SWC_data, cntVert)

def mask_2_swc(TIFFileName, SWCFileName, smp=4, ZRatio=1):

    # Read binary mask
    Skl_ImFile = tiff.TiffFile(TIFFileName)
    Skl_np = Skl_ImFile.asarray()
    branch_data = csr.summarise(Skl_np)
    skel_obj = csr.Skeleton(Skl_np)
    Brch_vox = skel_obj.path_coordinates
    NSkeletons = np.unique(branch_data['skeleton-id']).shape[0]

    # Check that the mask only holds one skeleton
    if NSkeletons > 1:
        exit("Error: more than one skeleton found in the mask!")

    # Extract relevant data from skeleton branches
    NBranches = branch_data.shape[0]
    id_0 = np.zeros(NBranches,dtype=int)
    id_1 = np.zeros(NBranches,dtype=int)
    list = skel_obj.paths_list()
    for i in range(NBranches):
        id_0[i] = list[i][0]
        id_1[i] = list[i][-1]

    # Count number of unique vertices and build renumbering LUT
    AllNodes = (np.unique([id_0,id_1]))
    MaxVertexIdx = (np.amax(AllNodes))
    NVertices = AllNodes.size
    LUTVertices = np.zeros(MaxVertexIdx+1,dtype=int)
    VertExist = np.zeros(MaxVertexIdx+1,dtype=int)

    # Estimate total number of segments
    TotSegments = 1
    for i in range(NBranches):
        TotSegments = TotSegments + (1+np.floor(Brch_vox(i).shape[0]/smp)).astype(int)
    print("Number of branches: %i"%NBranches)
    print("Number of nodes: %i"%NVertices)
    print("Estimated number of segments: %i"%TotSegments)

    # Fill SWC array
    global SWC_data
    SWC_data = np.ones((TotSegments,7),dtype=int)
    BranchOrphaned = np.zeros(NBranches,dtype=int)
    cntVert = 0

    # Insert first branch
    Idx0 = id_0[0]
    Idx1 = id_1[0]
    SWC_data[cntVert,0] = cntVert
    LUTVertices[Idx0] = cntVert
    VertExist[Idx0] = 1
    SWC_data[cntVert,6] = -1
    SWC_data[cntVert,2] = Brch_vox(0)[0,2]
    SWC_data[cntVert,3] = Brch_vox(0)[0,1]
    SWC_data[cntVert,4] = Brch_vox(0)[0,0]*ZRatio
    Vox = Brch_vox(0);
    (SWC_data, cntVert) = insertNodes(cntVert, LUTVertices[Idx0], Vox, smp, ZRatio)
    LUTVertices[Idx1] = cntVert
    VertExist[Idx1] = 1

    # Main loop
    BrchToBeAdded = np.arange(NBranches, dtype=int)
    cntIt = 0
    for it in range(10):
        if np.sum(BrchToBeAdded)==0:
            break
        cntIt = cntIt + 1
        for j in range(1,NBranches):
            i = BrchToBeAdded[j]
            if i>0:
                Idx0 = id_0[i]
                Idx1 = id_1[i]
                if VertExist[Idx0]:
                    # First node exists, it is then an ancestor
                    Vox = Brch_vox(i)
                    (SWC_data, cntVert) = insertNodes(cntVert, LUTVertices[Idx0], Vox, smp, ZRatio)
                    LUTVertices[Idx1] = cntVert
                    VertExist[Idx1] = 1
                    BrchToBeAdded[j] = 0
                else:
                    if VertExist[Idx1]:
                        # Second node exists, it is then an ancestor
                        Vox = np.flip(Brch_vox(i),0)
                        (SWC_data, cntVert) = insertNodes(cntVert, LUTVertices[Idx1], Vox, smp, ZRatio)
                        LUTVertices[Idx0] = cntVert
                        VertExist[Idx0] = 1
                        BrchToBeAdded[j] = 0

    # Truncate SWC array and add 1 to all IDs (SWC convention)
    SWC_data = SWC_data[0:cntVert+1,:]
    for i in range(SWC_data.shape[0]):
        SWC_data[i,0] = SWC_data[i,0]+1
        if SWC_data[i,6] > -1:
            SWC_data[i,6] = SWC_data[i,6]+1

    # Check for duplicated nodes
    unique_rows = np.unique(SWC_data[:,2:4], axis=0)
    if unique_rows.shape[0] != SWC_data.shape[0]:
        #print("Warning: the skeleton holds loop(s), this is incompatible with SWC format and it will be encoded with duplicated nodes!")
        exit("Error: the skeleton holds loop(s), this is incompatible with SWC format!")

    # Display status
    print("Performed %i iterations" %cntIt)
    print("Remaining branches: %i " %np.count_nonzero(BrchToBeAdded))
    print("Number of segments: %i" %SWC_data.shape[0])

    # Write SWC file
    with open(SWCFileName, "w") as f:
        f.write("# ORIGINAL_SOURCE Mask2SWC 1.0\n# SCALE 1.0 1.0 1.0\n\n")
    f.close()
    with open(SWCFileName, "a") as f:
        np.savetxt(f, SWC_data, fmt='%i', delimiter=" ")
    f.close()

def mask_2_obj(TIFFileName, OBJFileName, smp=4, ZRatio=1):

    # Read skeleton image
    Skl_ImFile = tiff.TiffFile(TIFFileName)
    Skl_np = Skl_ImFile.asarray()

    # Analyze skeleton
    branch_data = csr.summarise(Skl_np)
    NBranches = branch_data.shape[0]
    skel_obj = csr.Skeleton(Skl_np)
    Brch_vox = skel_obj.path_coordinates
    NSkeletons = np.unique(branch_data['skeleton-id']).shape[0]

    # Extract relevant data from skeleton branches
    NBranches = branch_data.shape[0]
    id_0 = np.zeros(NBranches,dtype=int)
    id_1 = np.zeros(NBranches,dtype=int)
    list = skel_obj.paths_list()
    for i in range(NBranches):
        id_0[i] = list[i][0]
        id_1[i] = list[i][-1]

    # Parse all nodes to find unique vertices
    AllNodes = (np.unique([id_0,id_1]))
    MaxVertexIdx = (np.amax(AllNodes))
    NVertices = AllNodes.size

    # Estimate total number of segments
    TotSegments = 1
    for i in range(NBranches):
        TotSegments = TotSegments + (1+np.floor(Brch_vox(i).shape[0]/smp)).astype(int)

    # Display model information
    print("Number of skeletons: %i"%NSkeletons)
    print("Number of branches: %i"%NBranches)
    print("Number of nodes: %i"%NVertices)
    print("Estimated number of segments: %i"%TotSegments)

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

    # Display status
    print("Actual number of vertices: %i" %cntVertices)
    print("Actual number of segments: %i" %cntSegments)

    # Export OBJ file
    with open(OBJFileName, "w") as f:
        f.write("")
    f.close()
    with open(OBJFileName, "a") as f:
        for i in range(cntVertices):
            f.write("v %i %i %i\n" %(OBJ_Vdata[i,0],OBJ_Vdata[i,1],OBJ_Vdata[i,2]))
    f.close()
    with open(OBJFileName, "a") as f:
        for i in range(cntSegments):
            f.write("l %i %i\n" %(OBJ_Ldata[i,0],OBJ_Ldata[i,1]))
    f.close()