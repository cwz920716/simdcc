import numpy as np
import abc
from abc import ABCMeta
from itertools import zip_longest
import contextlib
from enum import Enum
import operator
import sys
import copy

import sympy
from sympy import symbols
from sympy.tensor import Indexed, IndexedBase, Idx

import islpy

DEBUG_LVL=0
def debug(msg, level=1):
    from inspect import currentframe, getframeinfo, getouterframes
    frameinfo = getouterframes( currentframe() )[1]
    if level > DEBUG_LVL:
        print( 'DEBUG(' + str(level) + '): ' + str(msg) + '\n\t\t\t\t\tFrom file: ' + frameinfo.filename + ' line: ' + str(frameinfo.lineno) + '\n', file=sys.stderr )
    return

def debug_type(exp, level=1):
    return debug(type(msg), level)

def debug_always(msg):
    return debug(msg, DEBUG_LVL + 1)
    


