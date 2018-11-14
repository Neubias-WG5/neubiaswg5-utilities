import numpy as np
from PIL.Image import fromarray
from PIL.ImageDraw import ImageDraw
from shapely.geometry import Polygon, Point

from neubiaswg5.exporter.export_util import draw_poly


def draw_square(image, side, center, color):
    """Draw a square centered in 'center' and of which the side has 'side'"""
    top_left = (center[1] - side / 2, center[0] - side / 2)
    top_right = (center[1] + side / 2, center[0] - side / 2)
    bottom_left = (center[1] - side / 2, center[0] + side / 2)
    bottom_right = (center[1] + side / 2, center[0] + side / 2)
    p = Polygon([top_left, top_right, bottom_right, bottom_left, top_left])
    return draw_poly(image, p, color)


def draw_square_by_corner(image, side, top_left, color):
    top_left = (top_left[1], top_left[0])
    top_right = (top_left[0] + side, top_left[1])
    bottom_left = (top_left[0], top_left[1] + side)
    bottom_right = (top_left[0] + side, top_left[1] + side)
    p = Polygon([top_left, top_right, bottom_right, bottom_left, top_left])
    return draw_poly(image, p, color)


def draw_circle(image, radius, center, color=255, return_circle=False):
    """Draw a circle of radius 'radius' and centered in 'centered'"""
    circle_center = Point(*center)
    circle_polygon = circle_center.buffer(radius)
    image_out = draw_poly(image, circle_polygon, color)
    if return_circle:
        return image_out, circle_polygon
    else:
        return image_out

