import numpy as np
from random import Random
from math import pi, sin, cos, sqrt

from spirl.utils.general_utils import AttrDict


DEFAULT_SIDE = 0.1
DEFAULT_SAFETY_MARGIN = DEFAULT_SIDE * sqrt(2)


def quat_from_angle_and_axis(angle, axis=np.array([0.,0.,1.])):
    assert axis.shape == (3,)
    axis /= np.linalg.norm(axis)
    quat = np.concatenate([[np.cos(angle / 2.)], np.sin(angle / 2.) * axis])
    quat /= np.linalg.norm(quat)
    return quat


def get_func_deg1(p0, p1):
    (x0, y0), (x1, y1) = p0, p1
    if x0 == x1:
        return None
    a = (y0 - y1)/(x0 - x1)
    b = y0 - x0 * a
    return lambda x: a * x + b


def is_point_in_square(p, sq):
    x, y = p
    p0, p1, p2, p3 = sq
    side_func0 = get_func_deg1(p0, p1)
    side_func1 = get_func_deg1(p1, p2)
    side_func2 = get_func_deg1(p2, p3)
    side_func3 = get_func_deg1(p3, p0)
    if not side_func0 or not side_func1 or not side_func2 or not side_func3:
        xmin = min(p0[0], p2[0])
        xmax = max(p0[0], p2[0])
        ymin = min(p0[1], p2[1])
        ymax = max(p0[1], p2[1])
        return xmin <= x <= xmax and ymin <= y <= ymax
    return ((y - side_func0(x)) * (y - side_func2(x))) <= 0 and \
           ((y - side_func1(x)) * (y - side_func3(x))) <= 0


def squares_overlap(square0, square1):
    for p0 in square0:
        if is_point_in_square(p0, square1):
            return True
    for p1 in square1:
        if is_point_in_square(p1, square0):
            return True
    xc0 = (square0[0][0] + square0[2][0]) / 2
    yc0 = (square0[0][1] + square0[2][1]) / 2
    if is_point_in_square((xc0, yc0), square1):
        return True
    # The "reverse center check" not needed, since squares are congruent
    """
    xc1 = (square1[0][0] + square1[2][0]) / 2
    yc1 = (square1[0][1] + square1[2][1]) / 2
    if is_point_in_square((xc1, yc1), square0):
        return True
    """
    return False


def generate_random_point(rng, minx, miny, maxx, maxy, safety_margin=DEFAULT_SAFETY_MARGIN):
    if maxx - minx < 2 * safety_margin or maxy - miny < 2 * safety_margin:
        safety_margin = 0
    x = safety_margin + rng.random() * (maxx - minx - 2 * safety_margin) + minx
    y = safety_margin + rng.random() * (maxy - miny - 2 * safety_margin) + miny
    return x, y


def generate_random_angle(rng, max_val=pi/2):
    return rng.random() * max_val


def generate_random_square(rng, side=DEFAULT_SIDE, minx=0, miny=0, maxx=1, maxy=1, fixz=True, squares_to_avoid=()):
    while 1:
        restart = False
        x0, y0 = generate_random_point(rng, minx=minx, miny=miny, maxx=maxx, maxy=maxy)

        if fixz:
            angle = 0
        else:
            angle = generate_random_angle(rng)
        x1 = x0 + side * cos(angle)
        y1 = y0 + side * sin(angle)

        angle += pi / 2
        x2 = x1 + side * cos(angle)
        y2 = y1 + side * sin(angle)

        angle += pi / 2
        x3 = x2 + side * cos(angle)
        y3 = y2 + side * sin(angle)

        ret = (x0, y0), (x1, y1), (x2, y2), (x3, y3)
        for square in squares_to_avoid:
            if squares_overlap(ret, square):
                restart = True
        if restart:
            continue
        return ret


def fourpoints2posquat(square, minx, maxx, miny, maxy):
    xc, yc = np.array(square).mean(axis=0).tolist()
    x_side = (square[0][0] + square[1][0]) / 2
    y_side = (square[0][1] + square[1][1]) / 2
    angle = np.arctan2(y_side - yc, x_side - xc)
    xc = min(max(xc, minx), maxx)
    yc = min(max(yc, miny), maxy)
    return AttrDict({
        "pos": np.array([xc, yc]),
        "quat": quat_from_angle_and_axis(angle)
    })


class PlacementGenerator(object):
    def __init__(self, seed=None):
        self.rng = Random()
        self.rng.seed(seed)

    def generate_placement(self, num_squares, side=DEFAULT_SIDE,
                           minx=0, miny=0, maxx=1, maxy=1, fixz=True,
                           allow_overlapping=False, seed=None):
        if seed is not None:
            self.rng.seed(seed)

        squares = list()
        for _ in range(num_squares):
            square = generate_random_square(self.rng, side=side, minx=minx, miny=miny, maxx=maxx, maxy=maxy,
                                            fixz=fixz, squares_to_avoid=squares)
            if not allow_overlapping:
                squares.append(square)
        return [fourpoints2posquat(square, minx, maxx, miny, maxy) for square in squares]


if __name__ == "__main__":
    print(PlacementGenerator().generate_placement(num_squares=5, maxx=0))
