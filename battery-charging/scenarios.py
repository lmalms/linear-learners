import numpy as np
from models import BoundaryConditions, Forecasts, InitialConditions, SimulationScenario
from numpy.typing import NDArray

horizon, period = 120, 24
rng = np.random.default_rng(42)
noise = rng.normal(loc=0, scale=1, size=(horizon))


def demand_forecast(
    horizon: float = horizon,
    period: float = period,
    xoffest: float = 0.0,
) -> NDArray[np.float64]:
    return 4 * np.sin(2 * np.pi * (np.arange(horizon) - xoffest) / period) + 6.0


def price_forecast(
    horizon: float = horizon,
    period: float = period,
    xoffest: float = 0.0,
) -> NDArray[np.float64]:
    return 3 * np.sin(2 * np.pi * (np.arange(horizon) - xoffest) / period) + 5.0


SCENARIOS: list[SimulationScenario] = [
    SimulationScenario(
        name="in_sync_price_and_demand",
        timesteps=72,
        initial_conditions=InitialConditions(state_of_charge=20.0),
        boundary_conditions=BoundaryConditions(capacity=50.0),
        forecasts=Forecasts(demand=demand_forecast(), price=price_forecast()),
    ),
    SimulationScenario(
        name="noisy_price_and_demand",
        timesteps=72,
        initial_conditions=InitialConditions(state_of_charge=20.0),
        boundary_conditions=BoundaryConditions(capacity=50.0),
        forecasts=Forecasts(
            demand=np.maximum(demand_forecast(xoffest=2) + noise, 0.01),
            price=np.maximum(price_forecast() + noise, 0.01),
        ),
    ),
]
