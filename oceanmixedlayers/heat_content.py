"""
Compute heat content per unit area down to a specified depth.

This mirrors the pe_anomaly module’s broadcasting and partial-cell handling,
but performs a simple vertical integral of (CT - T_ref) scaled by constants.
"""

import numpy as np


class heat_content:
    """
    Compute heat content to a specified depth.

    Parameters
    ----------
    CT_layer: array-like
        Conservative Temperature averaged over each layer [deg C]. Shape (z, ...).
    Zc: array-like
        Depth of layer centers [m, negative downward]. Same shape as CT_layer.
    dZi: array-like
        Layer thicknesses [m, positive]. Same shape as CT_layer.
    DPT: float or array-like
        Target depth [m, negative downward]; broadcastable to CT_layer[0,...].
    rho_ref: float
        Reference density [kg/m^3]. Default 1025.0.
    cp_ref: float
        Reference specific heat capacity [J/(kg K)]. Default TEOS-10 cp0.
    T_ref: float
        Reference temperature [deg C] to compute heat content anomaly. Default 0.0.

    Attributes
    ----------
    HC: ndarray
        Heat content per unit area integrated from 0 to DPT [J/m^2], shape CT_layer[0,...].
    """

    def __init__(
        self,
        CT_layer,
        Zc,
        dZi,
        DPT,
        rho_ref: float = 1025.0,
        cp_ref: float = 3991.86795711963,
        T_ref: float = 0.0,
    ):
        self.rho_ref = rho_ref
        self.cp_ref = cp_ref
        self.T_ref = T_ref

        # If a single column is passed in, convert to a 2D array (z, 1)
        if len(np.shape(CT_layer)) == 1:
            CT_layer = np.atleast_2d(CT_layer).T

        # Broadcast Zc and dZi to match CT_layer if needed
        if np.shape(CT_layer) != np.shape(Zc):
            Zc = np.broadcast_to(Zc, np.shape(CT_layer.T)).T
            dZi = np.broadcast_to(dZi, np.shape(CT_layer.T)).T

        self.compute(CT_layer, Zc, dZi, DPT)

    def compute(self, CT_layer, Zc, dZi, DPT):
        dZ = np.copy(dZi)

        ND = CT_layer.shape[1:]
        NZ = CT_layer.shape[0]

        # Broadcast DPT to match horizontal dims if scalar or single value
        if np.size(DPT) <= 1:
            DPT = np.broadcast_to(DPT, ND)

        # Upper/lower interfaces for partial-cell handling
        Z_U = Zc + dZ / 2.0
        Z_L = Zc - dZ / 2.0

        HC = np.zeros(ND) + np.nan

        ACTIVE = np.ones(ND, dtype="bool")
        ACTIVE[np.isnan(CT_layer[0, ...])] = False

        # Zero depth (DPT == 0) yields zero heat content
        HC[DPT == 0] = 0.0
        ACTIVE[DPT == 0] = False

        FINAL = np.zeros(ND, dtype="bool")

        z = -1
        while (z < NZ - 1) and (np.sum(ACTIVE) > 0):
            z += 1

            # Identify columns where this layer crosses the target depth
            FINAL[ACTIVE] = Z_L[z, ACTIVE] < DPT[ACTIVE]

            # Clip the bottom interface to the target depth for those columns
            Z_L[z, FINAL] = DPT[FINAL]
            dZ[z, FINAL] = Z_U[z, FINAL] - Z_L[z, FINAL]

            # Compute heat content for those columns up to the current layer
            if np.sum(FINAL) > 0:
                HC_z = np.sum(
                    (CT_layer[: z + 1, FINAL] - self.T_ref) * dZ[: z + 1, FINAL],
                    axis=0,
                )
                HC[FINAL] = self.rho_ref * self.cp_ref * HC_z

            ACTIVE[FINAL] = False
            FINAL[:] = False

        # If target depth is deeper than profile bottom, integrate whole column
        if np.sum(ACTIVE) > 0:
            HC_z = np.sum(
                (CT_layer[:, ACTIVE] - self.T_ref) * dZ[:, ACTIVE],
                axis=0,
            )
            HC[ACTIVE] = self.rho_ref * self.cp_ref * HC_z

        self.HC = HC

