# Usage:            ComputeMetrics infile reffile problemclass tmpfolder (extra_params)
# infile:           Worflow output image (prediction)
# reffile:     	    Reference images (ground truth)
# problemclass:     Problem class (6 character string, see below)
# tmpfolder:        A temporary folder required for some metric computation
# extra_params:     A list of possible extra parameters required by some of the metrics (passed as extra arguments)
#
# Returns:
#  metrics_dict: mapping metrics name with their value
#  params_dict: mapping metric parameters with their value
#
# problemclass:
# "ObjSeg"      Object segmentation (DICE, AVD), binary 2D/3D mask images (regular multipage TIFF or OME-TIFF)
# "SptCnt"      Spot counting (Normalized spot count difference), binary 2D/3D mask images (regular multipage TIFF or OME-TIFF)
# "PixCla"    	Pixel classification (Confusion matrix, F1-score, accuracy, precision, recall), 2D/3D class masks with 0 background (regular multipage TIFF or OME-TIFF)
# "TreTrc"      Filament tree tracing - tbc by call to DIADEM metric taking 2 SWC input files
# "LooTrc"      Filament networks tracing (unmatched skeleton voxel rate + NetMets metric), 3D skeleton masks (regular multipage TIFF or OME-TIFF)
# "LndDet"      Landmark detection (landmark true positive rate, landmark false detection rate), 2D/3D class masks with 0 background, exactly 1 pixel / object (regular multipage TIFF or OME-TIFF)
# "ObjDet"      Object detection (TP, FN, FP, Recall, Precision, F1-score, RMSE over TP), 2D/3D binary masks, exactly 1 pixel / object (regular multipage TIFF or OME-TIFF prediction, reference must be OME-TIFF)
# "PrtTrk"      Particle tracking (Particle Tracking Challenge metric), label masks with track IDs, exactly 1 pixel / particle (regular multipage TIFF or OME-TIFF prediction, reference must be OME-TIFF)
# "ObjTrk"      Object tracking (Cell Tracking Challenge metrics), label masks with track IDs (regular multipage TIFF or OME-TIFF prediction, reference must be OME-TIFF) + object divisions text files

import os
import re
import shutil
import sys

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy import ndimage
import tifffile as tiff
from scipy.spatial import cKDTree
from neubiaswg5 import *
from neubiaswg5 import CLASS_LNDDET
from .img_to_xml import *
from .img_to_seq import *
from .skl2obj import *
from .netmets_obj import netmets_obj

def computemetrics_batch(infiles, refiles, problemclass, tmpfolder, verbose=True, **extra_params):
    """Runs compute metrics for all pairs of in and ref files.
    Metrics and parameters values are returned in a dictionary mapping the metrics and parameters names with
    a list of respective values (as many as pair of files).
    """
    results = dict()
    for infile, reffile in zip(infiles, refiles):
        metrics, params = computemetrics(infile, reffile, problemclass, tmpfolder, verbose=verbose, **extra_params)

        def extend_list_dict(all_dict, curr_dict):
            for metric_name, metric_value in curr_dict.items():
                results[metric_name] = all_dict.get(metric_name, []) + [metric_value]

        extend_list_dict(results, metrics)
        extend_list_dict(results, params)

    return results


def computemetrics(infile, reffile, problemclass, tmpfolder, verbose=True, **extra_params):
    # to suppress output
    try:
        with open(os.path.devnull, "w") as devnull:
            if not verbose:
                sys.stderr, sys.stdout = devnull, devnull
            outputs = _computemetrics(infile, reffile, problemclass, tmpfolder, **extra_params)
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    return outputs


def _computemetrics(infile, reffile, problemclass, tmpfolder, **extra_params):
    # Remove all xml and txt (temporary) files in tmpfolder
    filelist = [ f for f in os.listdir(tmpfolder) if (f.endswith(".xml") or f.endswith(".txt")) ]
    for f in filelist:
        os.remove(os.path.join(tmpfolder, f))

    # Remove all (temporary) subdirectories in tmpfolder
    for subdir in next(os.walk(tmpfolder))[1]:
        shutil.rmtree(os.path.join(tmpfolder, subdir), ignore_errors=True)

    metrics_dict = {}
    params_dict = {}

    # Switch problemclass
    if problemclass == CLASS_OBJSEG:

        # Call Visceral (compiled) to compute DICE and average Hausdorff distance
        os.system("Visceral "+infile+" "+reffile+" -use DICE,AVGDIST -xml "+tmpfolder+"/metrics.xml"+" > nul 2>&1")
        with open(tmpfolder+"/metrics.xml", "r") as myfile:
            # Parse returned xml file to extract all value fields
            data = myfile.read()
            inds = [m.start() for m in re.finditer("value", data)]
            bchmetrics = [data[ind+7:data.find('"',ind+7)] for ind in inds]

        metric_names = ["DC", "AHD"]
        metrics_dict.update({name: value for name, value in zip(metric_names, bchmetrics)})

    elif problemclass == CLASS_SPTCNT:

        Pred_ImFile = tiff.TiffFile(infile)
        Pred_Data = Pred_ImFile.asarray()
        y_pred = np.array(Pred_Data).ravel()	# Convert to 1-D array
        True_ImFile = tiff.TiffFile(reffile)
        True_Data = True_ImFile.asarray()
        y_true = np.array(True_Data).ravel()	# Convert to 1-D array
        cnt_pred = np.count_nonzero(y_pred)
        cnt_true = np.count_nonzero(y_true)
        bchmetrics = abs(cnt_pred-cnt_true)/cnt_true

        metrics_dict["REC"] = bchmetrics

    elif problemclass == CLASS_PIXCLA:

        Pred_ImFile = tiff.TiffFile(infile)
        Pred_Data = Pred_ImFile.asarray()
        y_pred = np.array(Pred_Data).ravel()	# Convert to 1-D array
        True_ImFile = tiff.TiffFile(reffile)
        True_Data = True_ImFile.asarray()
        y_true = np.array(True_Data).ravel()	# Convert to 1-D array

        # Clean the predictions and ground truths (labels: 1,2,3... anything else is discarded)
        y_true_cleaned = []
        y_pred_cleaned = []

        for i in range(0, len(y_true)):
            if y_true[i] > 0 or y_pred[i]>0:
                y_true_cleaned.append(y_true[i]>0)
                y_pred_cleaned.append(y_pred[i]>0)

        #metrics_dict["CM"] = confusion_matrix(y_true_cleaned, y_pred_cleaned)
        metrics_dict["F1"] = f1_score(y_true_cleaned, y_pred_cleaned, labels=None, pos_label=1, average='weighted', sample_weight=None)
        metrics_dict["ACC"] = accuracy_score(y_true_cleaned, y_pred_cleaned, normalize=True, sample_weight=None)
        metrics_dict["PR"] = precision_score(y_true_cleaned, y_pred_cleaned, labels=None, pos_label=1, average='weighted', sample_weight=None)
        metrics_dict["RE"] = recall_score(y_true_cleaned, y_pred_cleaned, labels=None, pos_label=1, average='weighted', sample_weight=None)

    elif problemclass == CLASS_TRETRC:

        pass

    elif problemclass == CLASS_LOOTRC:

        Pred_ImFile = tiff.TiffFile(infile)
        Pred_Data = Pred_ImFile.asarray()
        True_ImFile = tiff.TiffFile(reffile)
        True_Data = True_ImFile.asarray()

        # First metric is the rate of unmatched voxels between both trees (at a distance > gating_dist)
        Dst1 = ndimage.distance_transform_edt(Pred_Data==0)
        Dst2 = ndimage.distance_transform_edt(True_Data==0)
        indx = np.nonzero(np.logical_or(Pred_Data,True_Data))
        Dst1_onskl = Dst1[indx]
        Dst2_onskl = Dst2[indx]
        # the third parameter represents the gating distance
        gating_dist = extra_params.get("gating_dist", 5)
        unmatched_voxel_rate = (sum(Dst1_onskl > gating_dist)+sum(Dst2_onskl > gating_dist))/(Dst1_onskl.size+Dst2_onskl.size)

        metrics_dict["UVR"] = unmatched_voxel_rate
        params_dict["GATING_DIST"] = gating_dist

        pixel_smp = 3           # Skeleton sampling step is set to 3 to ensure accurate reconstruction
        ZRatio = 1              # Assumed equal to 1 in BIAFLOWS
        sigma = gating_dist     # NetMets sigma is set to gating_dist since both concepts are related
        subdiv = 4              # Set to default value

        # Convert skeleton masks to OBJ files
        skl2obj(True_Data,pixel_smp,ZRatio,os.path.join(tmpfolder, "GT.obj"))
        skl2obj(Pred_Data,pixel_smp,ZRatio,os.path.join(tmpfolder, "Pred.obj"))

        # Call NetMets on OBJ files
        metres = netmets_obj(os.path.join(tmpfolder, "GT.obj"),os.path.join(tmpfolder, "Pred.obj"),sigma,subdiv)

        metrics_dict["FNR"] = metres['FNR']
        metrics_dict["FPR"] = metres['FPR']
        params_dict['PIX_SMP'] = pixel_smp
        params_dict['SIGMA'] = sigma
        params_dict['SUBDIV'] = subdiv

    elif problemclass == CLASS_OBJDET:

        # Read metadata from reference image (OME-TIFF)
        img = tiff.TiffFile(reffile)
        T = img.ome_metadata.get('Image').get('Pixels').get('SizeT')
        Z = img.ome_metadata.get('Image').get('Pixels').get('SizeZ')
        Y = img.ome_metadata.get('Image').get('Pixels').get('SizeY')
        X = img.ome_metadata.get('Image').get('Pixels').get('SizeX')

        # Convert non null pixels coordinates to track files (single time point)
        ref_xml_fname = os.path.join(tmpfolder, "reftracks.xml")
        tracks_to_xml(ref_xml_fname, img_to_tracks(reffile,X,Y,Z,T), False)
        in_xml_fname = os.path.join(tmpfolder, "intracks.xml")
        tracks_to_xml(in_xml_fname, img_to_tracks(infile,X,Y,Z,T), False)

        # Call point matching metric code
        # the third parameter represents the gating distance
        gating_dist = extra_params.get("gating_dist", 5)
        #os.system('java -jar bin/win/DetectionPerformance.jar ' + ref_xml_fname + ' ' + in_xml_fname + ' ' + str(gating_dist))
        os.system('java -jar /usr/bin/DetectionPerformance.jar ' + ref_xml_fname + ' ' + in_xml_fname + ' ' + str(gating_dist))

        # Parse *.score.txt file created automatically in tmpfolder
        with open(in_xml_fname+".score.txt", "r") as f:
            bchmetrics = [line.split(':')[1].strip() for line in f.readlines()]

        metric_names = ["TP", "FN", "FP", "RE", "PR", "F1", "RMSE"]
        metrics_dict.update({name: value for name, value in zip(metric_names, bchmetrics)})
        params_dict["GATING_DIST"] = gating_dist

    elif problemclass == CLASS_LNDDET:

        Pred_ImFile = tiff.TiffFile(infile)
        Pred_Data = Pred_ImFile.asarray()
        True_ImFile = tiff.TiffFile(reffile)
        True_Data = True_ImFile.asarray()

        # Initialize metrics arrays
        maxlbl = np.maximum(np.amax(True_Data),np.amax(Pred_Data))
        N_REF = np.array([maxlbl,1])
        N_PRED = np.array([maxlbl,1])
        MRE = np.array([maxlbl,1],dtype='float')

        # Per class loop
        for i in range(maxlbl):
            coords_True = np.argwhere(True_Data == (i+1))
            coords_Pred = np.argwhere(Pred_Data == (i+1))
            min_dists, min_dist_idx = cKDTree(coords_True).query(coords_Pred, 1)
            N_REF[i] = coords_True.shape[0]
            N_PRED[i] = coords_Pred.shape[0]
            MRE[i] = np.mean(min_dists)

        metrics_dict['NREF'] = np.sum(N_REF)
        metrics_dict['NPRED'] = np.sum(N_PRED)
        metrics_dict['MRE'] = np.mean(MRE)
        
    elif problemclass == CLASS_PRTTRK:

        # Read metadata from reference image (OME-TIFF)
        img = tiff.TiffFile(reffile)
        T = img.ome_metadata.get('Image').get('Pixels').get('SizeT')
        Z = img.ome_metadata.get('Image').get('Pixels').get('SizeZ')
        Y = img.ome_metadata.get('Image').get('Pixels').get('SizeY')
        X = img.ome_metadata.get('Image').get('Pixels').get('SizeX')

        # Convert non null pixels coordinates to track files
        ref_xml_fname = os.path.join(tmpfolder, "reftracks.xml")
        tracks_to_xml(ref_xml_fname, img_to_tracks(reffile,X,Y,Z,T), True)
        in_xml_fname = os.path.join(tmpfolder, "intracks.xml")
        tracks_to_xml(in_xml_fname, img_to_tracks(infile,X,Y,Z,T), True)
        res_fname = in_xml_fname + ".score.txt"

        # Call tracking metric code
        gating_dist = extra_params.get("gating_dist", 5)
        # the fourth parameter represents the gating distance
        #os.system('java -jar bin/win/TrackingPerformance.jar -r ' + ref_xml_fname + ' -c ' + in_xml_fname + ' -o ' + res_fname + ' ' + str(gating_dist))
        os.system('java -jar /usr/bin/TrackingPerformance.jar -r ' + ref_xml_fname + ' -c ' + in_xml_fname + ' -o ' + res_fname + ' ' + str(gating_dist))

        # Parse the output file created automatically in tmpfolder
        with open(res_fname, "r") as f:
            bchmetrics = [line.split(':')[0].strip() for line in f.readlines()]

        metric_names = [
            "PD", "NPSA", "FNPSB", "NRT",  "NCT","JST", "NPT", "NMT", "NST", "NRD", "NCD", "JSD", "NPD", "NMD","NSD"
        ]
        metrics_dict.update({name: value for name, value in zip(metric_names, bchmetrics)})
        params_dict["GATING_DIST"] = gating_dist

    elif problemclass == CLASS_OBJTRK:

        # Convert the data into the Cell Tracking Challenge format
        ctc_gt_folder = os.path.join(tmpfolder, "01_GT")
        ctc_gt_seg = os.path.join(ctc_gt_folder, "SEG")
        ctc_gt_tra = os.path.join(ctc_gt_folder, "TRA")
        ctc_res_folder = os.path.join(tmpfolder, "01_RES")
        os.mkdir(ctc_gt_folder)
        os.mkdir(ctc_gt_seg)
        os.mkdir(ctc_gt_tra)
        os.mkdir(ctc_res_folder)

        # Read metadata from reference image (OME-TIFF)
        img = tiff.TiffFile(reffile)
        T = img.ome_metadata.get('Image').get('Pixels').get('SizeT')
        Z = img.ome_metadata.get('Image').get('Pixels').get('SizeZ')
        Y = img.ome_metadata.get('Image').get('Pixels').get('SizeY')
        X = img.ome_metadata.get('Image').get('Pixels').get('SizeX')

        # Convert image stack to image sequence (1 image per time point)
        img_to_seq(reffile, ctc_gt_seg, "man_seg",X,Y,Z,T)
        img_to_seq(reffile, ctc_gt_tra, "man_track",X,Y,Z,T)
        img_to_seq(infile, ctc_res_folder, "mask",X,Y,Z,T)

        # Copy the track text files into the created folders
        ref_txt_file = reffile[:reffile.find('.')]+".txt"
        in_txt_file = infile[:infile.find('.')]+".txt"
        shutil.copy2(ref_txt_file, os.path.join(ctc_gt_tra, "man_track.txt"))
        shutil.copy2(in_txt_file, os.path.join(ctc_res_folder, "res_track.txt"))

        # Run the evaluation routines
        measure_fname = os.path.join(tmpfolder, "measures.txt")
        #os.system("SEGMeasure " + tmpfolder + " 01 >> " + measure_fname)
        #os.system("TRAMeasure " + tmpfolder + " 01 >> " + measure_fname)
        os.system("/usr/bin/SEGMeasure " + tmpfolder + " 01 >> " + measure_fname)
        os.system("/usr/bin/TRAMeasure " + tmpfolder + " 01 >> " + measure_fname)

        #Parse the output file with the measured scores
        with open(measure_fname, "r") as f:
            bchmetrics = [line.split(':')[1].strip() for line in f.readlines()]

        metric_names = [ "SEG", "TRA" ]
        metrics_dict.update({name: value for name, value in zip(metric_names, bchmetrics)})

    return metrics_dict, params_dict
