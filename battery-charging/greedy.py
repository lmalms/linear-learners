import pulp as plp

from models import Action, State


def greedy(state: State) -> Action:
    price = state.forecasts.price[0]
    demand = state.forecasts.demand[0]
    state_of_charge = state.state_of_charge
    capacity = state.boundary_conditions.capacity

    problem = plp.LpProblem(name="battery_charging", sense=plp.LpMinimize)

    # Decision variable
    grid_inflow = plp.LpVariable(name="grid_inflow", lowBound=0, cat="Continuous")

    # Objective function
    problem += grid_inflow * price

    # Constraints
    problem += grid_inflow >= demand - state_of_charge
    problem += grid_inflow <= capacity - state_of_charge + demand

    problem.solve(plp.PULP_CBC_CMD(msg=False))
    return Action(timestep=state.timestep, grid_flow=grid_inflow.varValue)
