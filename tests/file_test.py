import pytest

def test_calc_addition():
    output = 2+4
    assert output == 6

def test_calc_substraction():
  # Function test the output of 2-4
    output = 2-4
    assert output == -2

def test_calc_multiply():
  # Function test the output of 2*4
    output = 2*4
    assert output == 8

def test_coucou():
  # Function test if the output return 'hello'
    output='hello'
    assert output == 'hello'
