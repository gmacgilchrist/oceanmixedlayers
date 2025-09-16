from pathlib import Path

import pytest

from oceanmixedlayers import oceanmixedlayers


@pytest.fixture(scope="session")
def idealized_columns():
    """Provide the three idealized columns used throughout the notebooks."""
    linear = oceanmixedlayers.column(
        kind="idealized",
        idealized_type="linear",
        T0=20.0,
        dTdz=0.01,
        S0=35.0,
        nz=1000,
        Dpt=100.0,
    )
    two_layer = oceanmixedlayers.column(
        kind="idealized",
        idealized_type="two-layer",
        T0=20.0,
        Tb0=10.0,
        nz=1000,
        Dpt=100.0,
    )
    shelf_thermocline = oceanmixedlayers.column(
        kind="idealized",
        idealized_type="two-layer",
        T0=20.0,
        Tb0=20.0,
        mixedfrac=0.8,
        dTdz=0.01,
        nz=1000,
        Dpt=100.0,
    )
    return {
        "linear": linear,
        "two_layer": two_layer,
        "shelf": shelf_thermocline,
    }


@pytest.fixture(scope="session")
def argo_data_path():
    # The notebooks use float 3900660; we vendor the same file for tests.
    return (Path(__file__).resolve().parent / "data" / "argo").as_posix()


@pytest.fixture(scope="session")
def argo_column(argo_data_path):
    return oceanmixedlayers.column(kind="Argo", ArgoPath=argo_data_path, ArgoID=3900660)
