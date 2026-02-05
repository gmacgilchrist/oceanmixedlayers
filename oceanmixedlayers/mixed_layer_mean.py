"""
Compute thickness–weighted mixed layer mean of an arbitrary tracer to a specified depth.

This mirrors the broadcasting and partial–cell handling used in ``heat_content``
but returns the average tracer value rather than an integrated quantity.
"""

import numpy as np


class mixed_layer_mean:
    """
    Compute the mixed layer mean of a tracer to a specified depth.

    Parameters
    ----------
    tracer_layer : array-like
        Tracer averaged over each layer. Shape (z, ...).
    Zc : array-like
        Depth of layer centers [m, negative downward]. Same shape as ``tracer_layer``.
    dZi : array-like
        Layer thicknesses [m, positive]. Same shape as ``tracer_layer``.
    DML : float or array-like
        Target mixed layer depth [m, negative downward]; broadcastable to ``tracer_layer[0,...]``.

    Attributes
    ----------
    ML : ndarray
        Mixed layer mean tracer value, shape ``tracer_layer[0,...]``.
    DZ_ML : ndarray
        Total thickness of the mixed layer actually integrated (useful when DML exceeds column depth).
    """

    def __init__(self, tracer_layer, Zc, dZi, DML):
        # If a single column is passed in, convert to a 2D array (z, 1)
        if len(np.shape(tracer_layer)) == 1:
            tracer_layer = np.atleast_2d(tracer_layer).T

        # Broadcast Zc and dZi to match tracer_layer if needed
        if np.shape(tracer_layer) != np.shape(Zc):
            Zc = np.broadcast_to(Zc, np.shape(tracer_layer.T)).T
            dZi = np.broadcast_to(dZi, np.shape(tracer_layer.T)).T

        self.compute(tracer_layer, Zc, dZi, DML)

    def compute(self, tracer_layer, Zc, dZi, DML):
        dZ = np.copy(dZi)

        ND = tracer_layer.shape[1:]
        NZ = tracer_layer.shape[0]

        # Broadcast DML to match horizontal dims if scalar or single value
        if np.size(DML) <= 1:
            DML = np.broadcast_to(DML, ND)

        # Upper/lower interfaces for partial-cell handling
        Z_U = Zc + dZ / 2.0
        Z_L = Zc - dZ / 2.0

        ML = np.zeros(ND) + np.nan
        DZ_ML = np.zeros(ND) + np.nan

        ACTIVE = np.ones(ND, dtype="bool")
        ACTIVE[np.isnan(tracer_layer[0, ...])] = False

        # Zero depth yields zero mean and zero thickness
        ML[DML == 0] = 0.0
        DZ_ML[DML == 0] = 0.0
        ACTIVE[DML == 0] = False

        FINAL = np.zeros(ND, dtype="bool")

        z = -1
        while (z < NZ - 1) and (np.sum(ACTIVE) > 0):
            z += 1

            # Identify columns where this layer crosses the target depth
            FINAL[ACTIVE] = Z_L[z, ACTIVE] < DML[ACTIVE]

            # Clip the bottom interface to the target depth for those columns
            Z_L[z, FINAL] = DML[FINAL]
            dZ[z, FINAL] = Z_U[z, FINAL] - Z_L[z, FINAL]

            # Compute mixed layer mean for those columns up to the current layer
            if np.sum(FINAL) > 0:
                num = np.sum(tracer_layer[: z + 1, FINAL] * dZ[: z + 1, FINAL], axis=0)
                den = np.sum(dZ[: z + 1, FINAL], axis=0)
                ML[FINAL] = num / den
                DZ_ML[FINAL] = den

            ACTIVE[FINAL] = False
            FINAL[:] = False

        # If target depth is deeper than profile bottom, use whole column
        if np.sum(ACTIVE) > 0:
            num = np.sum(tracer_layer[:, ACTIVE] * dZ[:, ACTIVE], axis=0)
            den = np.sum(dZ[:, ACTIVE], axis=0)
            ML[ACTIVE] = num / den
            DZ_ML[ACTIVE] = den

        self.ML = ML
        self.DZ_ML = DZ_ML
