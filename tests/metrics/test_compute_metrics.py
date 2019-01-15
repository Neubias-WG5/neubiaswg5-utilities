# Test script parsing images (ome-tif or tif) from two folders (in and ref)
# and computing metrics on these two images for a specific problemclass (see compute metrics)
#
# Usage: python test_compute_metrics.py in ref problemclass tmp extra_params
# in: input folder with workflow output images
# ref: reference folder with ground truth images
# problemclass: configure metrics computation depending on problem class
# tmp: path to a folder with I/O permission (used to store temporary data)
# extra_params: additional parameters required by some of the metrics
#
# Note: Sample images are provided for the different problem classes
#
# Sample calls: 
# python test_compute_metrics.py imgs/in_objseg_tiflbl imgs/ref_objseg_tiflbl "ObjSeg" tmp
# python test_compute_metrics.py imgs/in_sptcnt imgs/ref_sptcnt "SptCnt" tmp
# python test_compute_metrics.py imgs/in_pixcla imgs/ref_pixcla "PixCla" tmp
# python test_compute_metrics.py imgs/in_lootrc imgs/ref_lootrc "LooTrc" tmp gating_dist
# python test_compute_metrics.py imgs/in_tretrc imgs/ref_tretrc "TreTrc" tmp gating_dist
# python test_compute_metrics.py imgs/in_objdet imgs/ref_objdet "ObjDet" tmp gating_dist
# python test_compute_metrics.py imgs/in_prttrk imgs/ref_prttrk "PrtTrk" tmp gating_dist
# python test_compute_metrics.py imgs/in_objtrk imgs/ref_objtrk "ObjTrk" tmp

import os
from os import walk
from tempfile import TemporaryDirectory
from unittest import TestCase

from neubiaswg5.metrics import computemetrics, computemetrics_batch


class TestComputeMetrics(TestCase):
    def _test_metric(self, infolder, reffolder, problemclass, **extra_params):
        test_path = os.path.dirname(os.path.realpath(__file__))
        infolder = os.path.join(test_path, infolder)
        reffolder = os.path.join(test_path, reffolder)
        # Assume that matched TIFF images appear in the same order in both lists

        with TemporaryDirectory() as tmpfolder:
            infilenames = [os.path.join(infolder, filename) for _, _, files in walk(infolder) for filename in files if filename.endswith(".tif")]
            reffilenames = [os.path.join(reffolder, filename) for _, _, files in walk(reffolder) for filename in files if filename.endswith(".tif")]
            return computemetrics_batch(infilenames, reffilenames, problemclass, tmpfolder, **extra_params)

    def testObjSeg(self):
        results, params = self._test_metric(
            infolder="imgs/in_objseg_tiflbl",
            reffolder="imgs/ref_objseg_tiflbl",
            problemclass="ObjSeg"
        )
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 2)
        self.assertIn("DC", results)
        self.assertIn("AHD", results)

    def testSptCnt(self):
        results, params = self._test_metric(
            infolder="imgs/in_sptcnt",
            reffolder="imgs/ref_sptcnt",
            problemclass="SptCnt"
        )
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 1)
        self.assertIn("REC", results)

    def testPixCla(self):
        results, params = self._test_metric(
            infolder="imgs/in_pixcla",
            reffolder="imgs/ref_pixcla",
            problemclass="PixCla"
        )
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 5)
        self.assertIn("TP", results)
        self.assertIn("TN", results)
        self.assertIn("FP", results)
        self.assertIn("FN", results)
        self.assertIn("F1", results)
        self.assertIn("ACC", results)
        self.assertIn("PR", results)
        self.assertIn("RE", results)

    def testLooTrc(self):
        results, params = self._test_metric(
            infolder="imgs/in_lootrc",
            reffolder="imgs/ref_lootrc",
            problemclass="LooTrc",
            gating_dist=5
        )
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 1)
        self.assertIn("GATING_DIST", params)
        self.assertIn("UVR", results)

    def testTreTrc(self):
        results, params = self._test_metric(
            infolder="imgs/in_tretrc",
            reffolder="imgs/ref_tretrc",
            problemclass="TreTrc",
            gating_dist=5
        )
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 1)
        self.assertIn("GATING_DIST", params)
        self.assertIn("UVR", results)

    def testObjDet(self):
        results, params = self._test_metric(
            infolder="imgs/in_objdet",
            reffolder="imgs/ref_objdet",
            problemclass="ObjDet",
            gating_dist=5
        )
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 7)
        self.assertIn("GATING_DIST", params)
        self.assertIn("TP", results)
        self.assertIn("FN", results)
        self.assertIn("FP", results)
        self.assertIn("RE", results)
        self.assertIn("PR", results)
        self.assertIn("F1", results)
        self.assertIn("RMSE", results)

    def testPrtTrk(self):
        results, params = self._test_metric(
            infolder="imgs/in_prttrk",
            reffolder="imgs/ref_prttrk",
            problemclass="PrtTrk",
            gating_dist=5
        )
        expected_metrics = [
            "PD", "NDSA", "NFPSB", "NRT",  "NCT",
            "JST", "NPT", "NMT", "NST", "NRD",
            "NCD", "JSD", "NDP", "NMD", "NSD"
        ]
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), len(expected_metrics))
        self.assertIn("GATING_DIST", params)
        for exp_metric in expected_metrics:
            self.assertIn(exp_metric, results)

    def testObjTrk(self):
        results, params = self._test_metric(
            infolder="imgs/in_objtrk",
            reffolder="imgs/ref_objtrk",
            problemclass="ObjTrk"
        )
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 2)
        self.assertIn("SEG", results)
        self.assertIn("TRA", results)
