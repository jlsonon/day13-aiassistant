"""Stub pyaudioop module for Python 3.13 environments where audioop is unavailable.
This satisfies pydub imports used indirectly by Gradio when audio features aren't needed.
Functions return benign defaults. If audio processing is required, install a Python version
with audioop support (<=3.12) for full fidelity.
"""

# Basic helpers expected by pydub when probing audio; we return neutral values.

def rms(fragment, width):
    return 0

def avg(fragment, width):
    return 0

def maxpp(fragment, width):
    return 0

def findmax(fragment, width):
    return 0

def cross(fragment, width):
    return 0

def add(fragment1, fragment2, width):
    return fragment1  # no-op

def bias(fragment, width, bias):
    return fragment

def mul(fragment, width, factor):
    return fragment

def tomono(fragment, width, left, right):
    return fragment

def tostereo(fragment, width, left, right):
    return fragment

def reverse(fragment, width):
    return fragment

def lin2lin(fragment, width, newwidth):
    return fragment

def ratecv(fragment, width, nchannels, inrate, outrate, state, weightA=1, weightB=0):
    # Return fragment unchanged and None state
    return fragment, None

def getsample(fragment, width, index):
    return 0
