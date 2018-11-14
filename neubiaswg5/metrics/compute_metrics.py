# Usage:            ComputeMetrics infile reffile problemclass tmpfolder
# infile:           Worflow output image
# reffile:     	    Reference images (ground truth)
# problemclass:     Problem class (6 character string, see below)
# tmpfolder:        A temporary folder required for some metric computation
# extra_params:     A list of possible extra parameters required by some of the metrics
#
# Returns:
#  metrics_dict: mapping metrics name with their value
#  params_dict: mapping metric parameters with their value
#
# problemclass:
# "ObjSeg"      Object segmentation (DICE, AVD), work with binary or label 2D/3D masks images (regular multipage tif / OME-tif)
# "SptCnt"      Spot counting (Normalized spot count difference), same as above
# "PixCla"    	Pixel classification (Confusion matrix, F1-score, accuracy, precision, recall), same as above (0 pixels ignored)
# "TreTrc"      Filament tracing (trees), we consider including DIADEM metric but that requires to convert skeletons (workflow outputs) to SWC format
# "LooTrc"      Filament tracing (loopy networks)
# "ObjDet"      Object detection matching (TP, FN, FP, Recall, Precision, F1-score, RMSE over TP), not working yet
# "PrtTrk"      Particle (point) tracking (Particle Tracking Challenge metric), maximum linking distance set to a fixed value
# "ObjTrk"      Object tracking (Cell Tracking Challenge metrics), for object divisions requires an extra text file encoding division locations

import os
import re
import sys
import shutil
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
#from skimage.morphology import ball
#from skimage.morphology import dilation
import numpy as np
from scipy import ndimage
import tifffile as tiff
from .img_to_xml import *
from .img_to_seq import *

def computemetrics_batch(infiles, refiles, problemclass, tmpfolder, extra_params=None):
    """Runs compute metrics for all pairs of in and ref files.
    Metrics and parameters values are returned in a dictionary mapping the metrics and parameters names with
    a list of respective values (as many as pair of files).
    """
    results = dict()
    for infile, reffile in zip(infiles, refiles):
        metrics, params = computemetrics(infile, reffile, problemclass, tmpfolder, extra_params=extra_params)

        def extend_list_dict(all_dict, curr_dict):
            for metric_name, metric_value in curr_dict.items():
                results[metric_name] = all_dict.get(metric_name, []) + [metric_value]

        extend_list_dict(results, metrics)
        extend_list_dict(results, params)

    return results


def computemetrics(infile, reffile, problemclass, tmpfolder, extra_params=None):

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
    if problemclass == "ObjSeg":

        # Call Visceral (compiled) to compute DICE and average Hausdorff distance
        os.system("Visceral "+infile+" "+reffile+" -use DICE,AVGDIST -xml "+tmpfolder+"/metrics.xml"+" > nul 2>&1")
        with open(tmpfolder+"/metrics.xml", "r") as myfile:
            # Parse returned xml file to extract all value fields
            data = myfile.read()
            inds = [m.start() for m in re.finditer("value", data)]
            bchmetrics = [data[ind+7:data.find('"',ind+7)] for ind in inds]

        metric_names = ["DICE_COEFFICIENT", "AVERAGE_HAUSDORFF_DISTANCE"]
        metrics_dict.update({name: value for name, value in zip(metric_names, bchmetrics)})

    elif problemclass == "SptCnt":

        Pred_ImFile = tiff.TiffFile(infile)
        Pred_Data = Pred_ImFile.asarray()
        y_pred = np.array(Pred_Data).ravel()	# Convert to 1-D array
        True_ImFile = tiff.TiffFile(reffile)
        True_Data = True_ImFile.asarray()
        y_true = np.array(True_Data).ravel()	# Convert to 1-D array
        cnt_pred = np.count_nonzero(y_pred)
        cnt_true = np.count_nonzero(y_true)
        bchmetrics = abs(cnt_pred-cnt_true)/cnt_true

        metrics_dict["RELATIVE_ERROR_COUNT"] = bchmetrics

    elif problemclass == "PixCla":

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

        metrics_dict["CONFUSION_MATRIX"] = confusion_matrix(y_true_cleaned, y_pred_cleaned)
        metrics_dict["F1_SCORE"] = f1_score(y_true_cleaned, y_pred_cleaned, labels=None, pos_label=1, average='weighted', sample_weight=None)
        metrics_dict["ACCURACY"] = accuracy_score(y_true_cleaned, y_pred_cleaned, normalize=True, sample_weight=None)
        metrics_dict["PRECISION"] = precision_score(y_true_cleaned, y_pred_cleaned, labels=None, pos_label=1, average='weighted', sample_weight=None)
        metrics_dict["RECALL"] = recall_score(y_true_cleaned, y_pred_cleaned, labels=None, pos_label=1, average='weighted', sample_weight=None)

    elif problemclass == "TreTrc":

        Pred_ImFile = tiff.TiffFile(infile)
        Pred_Data = Pred_ImFile.asarray()
        True_ImFile = tiff.TiffFile(reffile)
        True_Data = True_ImFile.asarray()
        Dst1 = ndimage.distance_transform_edt(Pred_Data==0)
        Dst2 = ndimage.distance_transform_edt(True_Data==0)
        indx = np.nonzero(np.logical_or(Pred_Data,True_Data))
        Dst1_onskl = Dst1[indx]
        Dst2_onskl = Dst2[indx]
        gating_dist = 5
        if extra_params is not None: gating_dist = int(extra_params[0])

        metrics_dict["UNMATCHED_VOXEL_RATE"] = (sum(Dst1_onskl > gating_dist)+sum(Dst2_onskl > gating_dist))/(Dst1_onskl.size+Dst2_onskl.size)
        params_dict["GATING_DIST"] = gating_dist

    elif problemclass == "LooTrc":

        Pred_ImFile = tiff.TiffFile(infile)
        Pred_Data = Pred_ImFile.asarray()
        True_ImFile = tiff.TiffFile(reffile)
        True_Data = True_ImFile.asarray()
        Dst1 = ndimage.distance_transform_edt(Pred_Data==0)
        Dst2 = ndimage.distance_transform_edt(True_Data==0)
        indx = np.nonzero(np.logical_or(Pred_Data,True_Data))
        Dst1_onskl = Dst1[indx]
        Dst2_onskl = Dst2[indx]
        gating_dist = 5
        if extra_params is not None: gating_dist = int(extra_params[0])

        metrics_dict["UNMATCHED_VOXEL_RATE"] = (sum(Dst1_onskl > gating_dist)+sum(Dst2_onskl > gating_dist))/(Dst1_onskl.size+Dst2_onskl.size)
        params_dict["GATING_DIST"] = gating_dist

        #Msk1 = dilation(Pred_Data, ball(5))
        #Msk2 = dilation(True_Data, ball(5))
        #Msk1 = np.array(Msk1).ravel()
        #Msk2 = np.array(Msk2).ravel()
        #dildice = 2*sum(Msk1&Msk2)/(sum(Msk1)+sum(Msk2))
        #bchmetrics = [dildice]

    elif problemclass == "ObjDet":

        ref_xml_fname = os.path.join(tmpfolder, "reftracks.xml")
        tracks_to_xml(ref_xml_fname, img_to_tracks(reffile), False)
        in_xml_fname = os.path.join(tmpfolder, "intracks.xml")
        tracks_to_xml(in_xml_fname, img_to_tracks(infile), False)
        # the third parameter represents the gating distance
        gating_dist = ''
        if extra_params is not None: gating_dist = extra_params[0]
        os.system('java -jar /usr/bin/DetectionPerformance.jar ' + ref_xml_fname + ' ' + in_xml_fname + ' ' + gating_dist)

        # Parse *.score.txt file created automatically in tmpfolder
        with open(in_xml_fname+".score.txt", "r") as f:
            bchmetrics = [line.split(':')[1].strip() for line in f.readlines()]

        metric_names = ["TRUE_POS", "FALSE_NEG", "FALSE_POS", "RECALL", "PRECISIOM", "F1_SCORE", "RMSE"]
        metrics_dict.update({name: value for name, value in zip(metric_names, bchmetrics)})
        params_dict["GATING_DIST"] = gating_dist

    elif problemclass == "PrtTrk":

        ref_xml_fname = os.path.join(tmpfolder, "reftracks.xml")
        tracks_to_xml(ref_xml_fname, img_to_tracks(reffile), True)
        in_xml_fname = os.path.join(tmpfolder, "intracks.xml")
        tracks_to_xml(in_xml_fname, img_to_tracks(infile), True)
        res_fname = in_xml_fname + ".score.txt"
        # the fourth parameter represents the gating distance
        gating_dist = ''
        if extra_params is not None: gating_dist = extra_params[0]
        os.system('java -jar /usr/bin/TrackingPerformance.jar -r ' + ref_xml_fname + ' -c ' + in_xml_fname + ' -o ' + res_fname + ' ' + gating_dist)

        # Parse the output file created automatically in tmpfolder
        with open(res_fname, "r") as f:
            bchmetrics = [line.split(':')[0].strip() for line in f.readlines()]

        metric_names = [
            "PAIRING_DST", "NORM_PAIRING_SCORE_ALPHA", "FULL_NORM_PAIRING_SCORE_BETA", "N_REF_TRACKS",  "N_CAND_TRACKS",
            "JACCARD_SIMILARITY_TRACKS", "N_PAIRED_TRACKS", "N_MISSED_TRACKS", "N_SPURIOUS_TRACKS", "N_REF_DETECTIONS",
            "N_CAND_DETECTIONS", "JACCARD_SIMILARITY_DET", "N_PAIRED_DETECTIONS", "N_MISSED_DETECTIONS",
            "N_SPURIOUS_DETECTIONS"
        ]
        metrics_dict.update({name: value for name, value in zip(metric_names, bchmetrics)})
        params_dict["GATING_DIST"] = gating_dist

    elif problemclass == "ObjTrk":

        # Convert the data into the Cell Tracking Challenge format
        ctc_gt_folder = os.path.join(tmpfolder, "01_GT")
        ctc_gt_seg = os.path.join(ctc_gt_folder, "SEG")
        ctc_gt_tra = os.path.join(ctc_gt_folder, "TRA")
        ctc_res_folder = os.path.join(tmpfolder, "01_RES")
        os.mkdir(ctc_gt_folder)
        os.mkdir(ctc_gt_seg)
        os.mkdir(ctc_gt_tra)
        os.mkdir(ctc_res_folder)
        img_to_seq(reffile, ctc_gt_seg, "man_seg")
        img_to_seq(reffile, ctc_gt_tra, "man_track")
        img_to_seq(infile, ctc_res_folder, "mask")

        # Copy the track text files into the created folders
        ref_txt_file = reffile[:reffile.find('.')]+".txt"
        in_txt_file = infile[:infile.find('.')]+".txt"
        shutil.copy2(ref_txt_file, os.path.join(ctc_gt_tra, "man_track.txt"))
        shutil.copy2(in_txt_file, os.path.join(ctc_res_folder, "res_track.txt"))

        # Run the evaluation routines
        measure_fname = os.path.join(tmpfolder, "measures.txt")
        os.system("/usr/bin/SEGMeasure.exe " + tmpfolder + " 01 >> " + measure_fname)
        os.system("/usr/bin/TRAMeasure.exe " + tmpfolder + " 01 >> " + measure_fname)

        #Parse the output file with the measured scores
        with open(measure_fname, "r") as f:
            bchmetrics = [line.split(':')[1].strip() for line in f.readlines()]

        metric_names = [ "SEG", "TRA" ]
        metrics_dict.update({name: value for name, value in zip(metric_names, bchmetrics)})

    return metrics_dict, params_dict
