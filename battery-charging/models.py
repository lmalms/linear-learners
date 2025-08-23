from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class BoundaryConditions:
    capacity: float


@dataclass
class InitialConditions:
    state_of_charge: float


@dataclass
class Forecasts:
    demand: NDArray[np.float64]
    price: NDArray[np.float64]

    def __post_init__(self):
        assert self.demand.ndim == 1
        assert self.price.ndim == 1


@dataclass
class State:
    timestep: int
    state_of_charge: float
    boundary_conditions: BoundaryConditions
    forecasts: Forecasts


@dataclass
class Action:
    timestep: int
    grid_flow: float


@dataclass
class SimulationScenario:
    name: str
    timesteps: int
    initial_conditions: InitialConditions
    boundary_conditions: BoundaryConditions
    forecasts: Forecasts


@dataclass
class SimulationResults:
    grid_flow: list[float]
    state_of_charge: list[float]
    cost: list[float]
