from greedy import greedy
from models import Action, Forecasts, SimulationResults, SimulationScenario, State
from mpc import MpcSimulationParameters, mpc


def transition(state: State, action: Action) -> State:
    assert state.timestep == action.timestep

    current_demand: float = state.forecasts.demand[0]
    new_demand_forecast = state.forecasts.demand[1:]
    new_price_forecast = state.forecasts.price[1:]

    # Update state given current state and action
    new_state_of_charge = state.state_of_charge - current_demand + action.grid_flow

    return State(
        timestep=state.timestep + 1,
        state_of_charge=new_state_of_charge,
        forecasts=Forecasts(demand=new_demand_forecast, price=new_price_forecast),
        boundary_conditions=state.boundary_conditions,
    )


def compute_cost_of_charge(state: State, action: Action) -> float:
    assert state.timestep == action.timestep
    return state.forecasts.price[0] * action.grid_flow


def simulate_greedy(scenario: SimulationScenario) -> SimulationResults:

    initial_soc = scenario.initial_conditions.state_of_charge
    grid_flow_history: list[float] = [0.0]
    state_of_charge_history: list[float] = [initial_soc]
    cost_history: list[float] = [0.0]

    current_state = State(
        timestep=0,
        state_of_charge=initial_soc,
        boundary_conditions=scenario.boundary_conditions,
        forecasts=scenario.forecasts,
    )

    for t in range(scenario.timesteps):
        # Take action given current state
        action = greedy(state=current_state)
        grid_flow_history.append(action.grid_flow)

        # Calculate objective value
        cost_of_charge = compute_cost_of_charge(current_state, action)
        cost_history.append(cost_of_charge)

        # Transition to new state given action
        current_state = transition(state=current_state, action=action)
        state_of_charge_history.append(current_state.state_of_charge)

    return SimulationResults(
        grid_flow=grid_flow_history,
        state_of_charge=state_of_charge_history,
        cost=cost_history,
    )


def simulate_mpc(
    scenario: SimulationScenario,
    parameters: MpcSimulationParameters,
) -> SimulationResults:

    n_timesteps = scenario.timesteps
    horizon = parameters.horizon
    assert n_timesteps + horizon <= len(scenario.forecasts.demand)
    assert n_timesteps + horizon <= len(scenario.forecasts.price)

    initial_soc = scenario.initial_conditions.state_of_charge
    grid_flow_history: list[float] = [0.0]
    state_of_charge_history: list[float] = [initial_soc]
    cost_history: list[float] = [0.0]

    current_state = State(
        timestep=0,
        state_of_charge=initial_soc,
        boundary_conditions=scenario.boundary_conditions,
        forecasts=scenario.forecasts,
    )

    for t in range(n_timesteps):
        # Take action given current state
        action = mpc(state=current_state, parameters=parameters)
        grid_flow_history.append(action.grid_flow)

        # Calculate objective value
        cost_of_charge = compute_cost_of_charge(current_state, action)
        cost_history.append(cost_of_charge)

        # Transition to new state given action
        current_state = transition(state=current_state, action=action)
        state_of_charge_history.append(current_state.state_of_charge)

    return SimulationResults(
        grid_flow=grid_flow_history,
        state_of_charge=state_of_charge_history,
        cost=cost_history,
    )
