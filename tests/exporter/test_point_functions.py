import tempfile
import os
import numpy as np
from unittest import TestCase
from shapely.geometry import Point, Polygon, box

from neubiaswg5.exporter import mask_to_points_2d, csv_to_points, AnnotationSlice, slices_to_mask


class TestMaskToPoints(TestCase):
    def testSinglePoint(self):
        image = np.zeros([50, 50], dtype=np.int)
        image[5, 6] = 125

        slices = mask_to_points_2d(image)

        self.assertEqual(len(slices), 1)
        self.assertIsInstance(slices[0].polygon, Point)
        self.assertEqual(slices[0].polygon.x, 6)
        self.assertEqual(slices[0].polygon.y, 5)

    def testSinglePointEncodedSquare(self):
        image = np.zeros([50, 50], dtype=np.int)
        image[5, 6] = 125

        slices = mask_to_points_2d(image, points=False)

        self.assertEqual(len(slices), 1)
        self.assertIsInstance(slices[0].polygon, Polygon)
        self.assertTrue(slices[0].polygon.equals(box(5, 4, 7, 6)))


class TestCsvToPoints(TestCase):
    def _create_file(self, filepath, header=False, sep='\t', has_z=True, has_t=True):
        with open(filepath, "w+") as file:
            if header:
                file.write("x{sep}y{sep}z{sep}t\n".format(sep=sep))
            first = [1.25, 25.36, 663.3, 25]
            second = [63.3, 32, 29, 66]
            z_start, z_end = 2, 3 if has_z else 2
            t_start, t_end = 3, 4 if has_t else 3
            file.write(sep.join([str(n) for n in first[:2] + first[z_start:z_end] + first[t_start:t_end]]) + "\n")
            file.write(sep.join([str(n) for n in second[:2] + second[z_start:z_end] + second[t_start:t_end]]) + "\n")
            file.write("\n")

    def _asserts_for_tmp_file(self, slices):
        self.assertEqual(len(slices), 2)

    def _generic_test(self, has_header=True, sep='\t', has_z=True, has_t=True, parse_fn=None):
        filepath = "./tmp.csv"
        try:
            self._create_file(filepath, header=has_header, sep=sep, has_t=has_t, has_z=has_z)
            slices = csv_to_points(filepath, sep=sep, has_z=has_z, has_t=has_t, has_headers=has_header, parse_fn=parse_fn)
            self.assertIsInstance(slices[0].polygon, Point)
            self.assertAlmostEqual(slices[0].polygon.x, 1.25)
            self.assertAlmostEqual(slices[0].polygon.y, 25.36)
            self.assertAlmostEqual(slices[1].polygon.x, 63.3)
            self.assertAlmostEqual(slices[1].polygon.y, 32)
            if has_z:
                self.assertAlmostEqual(slices[0].depth, 663.3)
                self.assertAlmostEqual(slices[1].depth, 29)
            else:
                self.assertAlmostEqual(slices[0].depth, None)
                self.assertAlmostEqual(slices[1].depth, None)
            if has_t:
                self.assertAlmostEqual(slices[0].time, 25)
                self.assertAlmostEqual(slices[1].time, 66)
            else:
                self.assertAlmostEqual(slices[0].time, None)
                self.assertAlmostEqual(slices[1].time, None)
        except Exception:
            raise
        finally:
            if os.path.isfile(filepath):
                os.remove(filepath)

    def testFull(self):
        self._generic_test(has_header=True, sep=",", has_z=False, has_t=False)
        self._generic_test(has_header=True, sep=",", has_z=True, has_t=False)
        self._generic_test(has_header=True, sep="\t", has_z=False, has_t=True)
        self._generic_test(has_header=True, sep="\t", has_z=True, has_t=True)
        self._generic_test(has_header=False, sep=",", has_z=False, has_t=False)
        self._generic_test(has_header=False, sep=",", has_z=True, has_t=False)
        self._generic_test(has_header=False, sep="\t", has_z=False, has_t=True)
        self._generic_test(has_header=False, sep="\t", has_z=True, has_t=True)
        self._generic_test(has_header=False, sep=",", has_z=True, has_t=True,
                           parse_fn=lambda l, sep: [float(c) for c in l.split(sep)[:4]])


class TestSliceToMask(TestCase):
    def testOnePoint(self):
        _slice = AnnotationSlice(Point(2, 3), label=127, time=None, depth=None)
        dim = (5, 4)
        mask = slices_to_mask([_slice], dim)
        self.assertEqual(mask.shape, dim)
        ys, xs = mask.nonzero()
        self.assertEqual(len(ys), 1)
        self.assertEqual((ys[0], xs[0]), (3, 2))
        self.assertEqual(mask[3, 2], 127)

    def testTwoPointsWithZ(self):
        dim = (3, 4, 5)
        _slice1 = AnnotationSlice(Point(3, 2), label=125, time=None, depth=1)
        _slice2 = AnnotationSlice(Point(2, 2), label=126, time=None, depth=2)
        mask = slices_to_mask([_slice1, _slice2], dim)
        self.assertEqual(mask.shape, dim)
        ys, xs, zs = mask.nonzero()
        self.assertEqual(len(ys), 2)
        self.assertEqual((ys[0], xs[0], zs[0]), (2, 2, 2))
        self.assertEqual((ys[1], xs[1], zs[1]), (2, 3, 1))
        self.assertEqual(mask[2, 2, 2], 126)
        self.assertEqual(mask[2, 3, 1], 125)

    def testTwoPointsWithT(self):
        dim = (3, 4, 5)
        _slice1 = AnnotationSlice(Point(3, 2), label=125, depth=None, time=1)
        _slice2 = AnnotationSlice(Point(2, 2), label=126, depth=None, time=2)
        mask = slices_to_mask([_slice1, _slice2], dim)
        self.assertEqual(mask.shape, dim)
        ys, xs, zs = mask.nonzero()
        self.assertEqual(len(ys), 2)
        self.assertEqual((ys[0], xs[0], zs[0]), (2, 2, 2))
        self.assertEqual((ys[1], xs[1], zs[1]), (2, 3, 1))
        self.assertEqual(mask[2, 2, 2], 126)
        self.assertEqual(mask[2, 3, 1], 125)

    def testTwoPointsWithZandT(self):
        dim = (3, 4, 6, 5)
        _slice1 = AnnotationSlice(Point(3, 2), label=125, depth=3, time=1)
        _slice2 = AnnotationSlice(Point(2, 2), label=126, depth=5, time=2)
        mask = slices_to_mask([_slice1, _slice2], dim)
        self.assertEqual(mask.shape, dim)
        ys, xs, zs, ts = mask.nonzero()
        self.assertEqual(len(ys), 2)
        self.assertEqual((ys[0], xs[0], zs[0], ts[0]), (2, 2, 5, 2))
        self.assertEqual((ys[1], xs[1], zs[1], ts[1]), (2, 3, 3, 1))
        self.assertEqual(mask[2, 2, 5, 2], 126)
        self.assertEqual(mask[2, 3, 3, 1], 125)

