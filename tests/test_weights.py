import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
from motion_estimation.weights import tukey, median_absolute_deviation



def test_tukey():
    x = np.array([3, -2, 1, 0])
    GT = np.array([0, 0, 9/16, 1])
    assert_array_equal(tukey(x, b=2), GT)


def test_median_absolute_deviation():
    x = np.array([1, 1, 2, 2, 4, 6, 9])
    assert_equal(median_absolute_deviation(x), 1)


test_tukey()
test_median_absolute_deviation()
