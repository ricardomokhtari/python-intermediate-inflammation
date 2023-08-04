"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0,0], [0,0], [0,0]], [0,0]),
        ([ [1, 2], [3, 4], [5, 6] ], [3, 4]),
    ]
)
def test_daily_mean(test, expected):
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0,0], [0,0], [0,0]], [0.0,0.0]),
        ([[1, 2], [3, 4], [5, 6]], [1.63299316, 1.63299316]),
    ]
)
def test_daily_std(test, expected):
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_std
    npt.assert_almost_equal(daily_std(np.array(test)), np.array(expected))



@pytest.mark.parametrize(
    "test, expected, expect_raises",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], None),
        ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], None),
        ([[float('nan'), 1, 1], [1, 1, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1], [1, 1, 1]], None),
        ([[-1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]], ValueError),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]], None),
        ([[-1, 2, 3], [4, 5, 6], [7, 8, 9]],[[0, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]], ValueError,),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]], None,),
        (3,None,TypeError),
        ('hello', None, TypeError)

    ])
def test_patient_normalise(test, expected, expect_raises):
    """Test normalisation works for arrays of one and positive integers."""
    from inflammation.models import patient_normalise
    if isinstance(test, list):
        test = np.array(test)
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            npt.assert_almost_equal(patient_normalise(test), np.array(expected), decimal=2)
    else:
        npt.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal=2)




@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0,0], [0,0], [0,0]], [0,0]),
        ([ [1, 2], [3, 4], [5, 6] ], [5, 6]),
    ]
)
def test_daily_max(test, expected):
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_max
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))




@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0,0], [0,0], [0,0]], [0,0]),
        ([ [1, 2], [3, 4], [5, 6] ], [1, 2]),
    ]
)
def test_daily_min(test, expected):
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_min
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))

