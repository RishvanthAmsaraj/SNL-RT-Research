"""
kinarm_rt -- a reaction-time analysis pipeline for the KINARM interception task,
reproducing the SNL-RT-Research method (single-boundary shifted-Wald + LATER).
"""

from ._speeds import (SPEEDS, EFFECTORS, PHYSIO_FLOOR, FILTER_WINDOWS,  # noqa: F401
                      V_MAX, A_MAX, MIX_SHIFT_FLOOR, SPEED_RGB)

__version__ = "0.2.0"
