import numpy as np
import pytest

from gennf.genetic_optimizer import GAConfig
from gennf.localization import GenNFConfig, GenNFLocalizer
from gennf.rssi_model import RSSIModel


def random_layout(rng):
    return rng.uniform((0, 0, 22), (200, 200, 50), size=(10, 3))


@pytest.fixture(scope="session")
def fast_localizer():
    cfg = GenNFConfig(n_train=600, epochs=20, ga=GAConfig(population_size=20, generations=5),
                      bounds_hi=(200, 200, 35), geometry="trilateration", seed=3)
    loc = GenNFLocalizer(RSSIModel(sigma_db=1.0), cfg)
    loc.fit(random_layout)
    return loc
