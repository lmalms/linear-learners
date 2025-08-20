from dataclasses import dataclass

import pulp as plp
from models import Action, State


@dataclass
class MpcSimulationParameters:
    horizon: int = 10
    discount_factor: float = 0.99


def mpc(state: State, parameters: MpcSimulationParameters) -> Action:

    demand_forecasts = state.forecasts.demand
    price_forecasts = state.forecasts.price
    capacity = state.boundary_conditions.capacity
    initial_state_of_charge = state.state_of_charge

    assert parameters.horizon <= len(demand_forecasts)
    assert parameters.horizon <= len(price_forecasts)

    timesteps = range(parameters.horizon)
    problem = plp.LpProblem(name="battery_charging", sense=plp.LpMinimize)

    # Decision variables
    grid_inflow = plp.LpVariable.dict(
        name="grid_inflow",
        indices=(timesteps,),
        lowBound=0,
        cat="Continuous",
    )
    state_of_charge = plp.LpVariable.dict(
        name="state_of_charge",
        indices=(timesteps,),
        lowBound=0,
        cat="Continuous",
    )

    # Objective function
    problem += plp.lpSum(
        [
            (parameters.discount_factor**k) * grid_inflow[k] * price_forecasts[k]
            for k in timesteps
        ]
    )

    # Constraints

    # total supply has to meet demand
    for k in timesteps:
        problem += grid_inflow[k] >= demand_forecasts[k] - state_of_charge[k]

    # cannot charge beyond capacity
    for k in timesteps:
        problem += grid_inflow[k] <= capacity - state_of_charge[k] + demand_forecasts[k]

    # transition constraint
    for k in timesteps[:-1]:
        problem += (
            state_of_charge[k + 1]
            == state_of_charge[k] - demand_forecasts[k] + grid_inflow[k]
        )

    # inital constraint
    problem += state_of_charge[0] == initial_state_of_charge

    problem.solve(plp.PULP_CBC_CMD(msg=False))
    return Action(timestep=state.timestep, grid_flow=grid_inflow[0].varValue)
