# Test script parsing images (ome-tif or tif) from two folders (in and ref)
# and computing metrics on these two images for a specific problemclass (see compute metrics)
#
# Usage: python Test_ComputeMetrics.py in ref problemclass tmp extra_params
# in: input folder with workflow output images
# ref: reference folder with ground truth images
# problemclass: configure metrics computation depending on problem class
# tmp: path to a folder with I/O permission (used to store temporary data)
# extra_params: additional parameters required by some of the metrics
#
# Note: Sample images are provided for the different problem classes
#
# Sample calls: 
# python Test_ComputeMetrics.py imgs/in_objseg_tiflbl imgs/ref_objseg_tiflbl "ObjSeg" tmp
# python Test_ComputeMetrics.py imgs/in_sptcnt imgs/ref_sptcnt "SptCnt" tmp
# python Test_ComputeMetrics.py imgs/in_pixcla imgs/ref_pixcla "PixCla" tmp
# python Test_ComputeMetrics.py imgs/in_lootrc imgs/ref_lootrc "LooTrc" tmp gating_dist
# python Test_ComputeMetrics.py imgs/in_tretrc imgs/ref_tretrc "TreTrc" tmp gating_dist
# python Test_ComputeMetrics.py imgs/in_objdet imgs/ref_objdet "ObjDet" tmp gating_dist
# python Test_ComputeMetrics.py imgs/in_prttrk imgs/ref_prttrk "PrtTrk" tmp gating_dist
# python Test_ComputeMetrics.py imgs/in_objtrk imgs/ref_objtrk "ObjTrk" tmp 

import sys
import os
from os import walk
from neubiaswg5.metrics.compute_metrics import computemetrics

infolder = sys.argv[1]
reffolder = sys.argv[2]
problemclass = sys.argv[3]
tmpfolder = sys.argv[4]
extra_params = None
if len(sys.argv) > 5:
        extra_params = sys.argv[5:]

# Assume that matched TIFF images appear in the same order in both lists
infilenames = [os.path.join(infolder,filename) for _, _, files in walk(infolder) for filename in files if filename.endswith(".tif")]
reffilenames = [os.path.join(reffolder,filename) for _, _, files in walk(reffolder) for filename in files if filename.endswith(".tif")]

for i in range(0,len(infilenames)):
    bchmetrics = computemetrics(infilenames[i],reffilenames[i],problemclass,tmpfolder,extra_params)
    print(bchmetrics)
