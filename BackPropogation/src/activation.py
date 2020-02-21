#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: activation.py

import numpy as np
import math


def sigmoid(z):
    """The sigmoid function."""
    
    return 1/(1+ math.e**-z)

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    s = 1/(1+ math.e**-z)
    ds = s*(1-s)
    return ds
