from rhapsody.backends import DragonExecutionBackend
from rhapsody.backends.data import DragonDataBackend

import rhapsody_plugins.openfoam as rof
from dragon.data.ddict import DDict
from radex.clients.core import DragonClient as Client

import skopt

from pitzdaily_case import KEpsilonParameters, pitzDailyCase

import asyncio
import multiprocessing as mp
from typing import cast

TARGET_AVGINLETS = -1.9

default_params = KEpsilonParameters(14.855, 0.09, 1.44, 1.92)


def initialize_optimizer():
    lower_bound = KEpsilonParameters(default_params.epsilon * 0.2, 0.05, 1.0, 1.5)
    upper_bound = KEpsilonParameters(default_params.epsilon * 4, 0.15, 1.5, 3.0)
    bounds = list(zip(lower_bound.as_optimizer_list(), upper_bound.as_optimizer_list()))
    return skopt.Optimizer(dimensions=bounds, random_state=10, base_estimator="gp")


def create_case(registry, radex_store_descriptor, parameter_values, case_index):
    parameters = KEpsilonParameters(*parameter_values)
    return pitzDailyCase(
        registry,
        parameters,
        f"pD-{case_index:04d}",
        radex_store_descriptor,
        TARGET_AVGINLETS,
    )


def update_optimizer(optimizer, case):
    if case.results.converged:
        optimizer.tell(case.parameters.as_optimizer_list(), case.results.loss)


async def main(max_concurrent_cases=32, max_cases=128, convergence=1e-5):
    if max_concurrent_cases < 1:
        raise ValueError("max_concurrent_cases must be at least 1")

    mp.set_start_method("dragon", force=True)

    backend = await DragonExecutionBackend()
    radex_store = await DragonDataBackend(
        managers_per_node=1,
        n_nodes=1,
        total_mem=512 * 1024 * 1024,
        working_set_size=2 + 2,
    )
    radex_store_descriptor = radex_store.endpoints[0].serialize()


    client = Client(radex_store_descriptor, timeout=30)
    registry = rof.OFExecutableRegistry()
    optimizer = initialize_optimizer()
    next_case_index = 0

    best_case_loss = 1e10
    best_case = None
    completed_cases = 0

    async def submit_case():
        nonlocal next_case_index

        parameter_values = optimizer.ask()
        case = create_case(
            registry, radex_store_descriptor, parameter_values, next_case_index
        )
        next_case_index += 1
        futures = await session.submit_tasks(case.stages["solve"].to_tasks())
        return case, asyncio.gather(*futures)

    async with rof.OFSession(backends=[backend]) as session:
        active_cases = {}
        while len(active_cases) < max_concurrent_cases and next_case_index < max_cases:
            case, completion = await submit_case()
            active_cases[completion] = case

        while active_cases:
            completed, _ = await asyncio.wait(
                active_cases, return_when=asyncio.FIRST_COMPLETED
            )

            for completion in completed:
                case = active_cases.pop(completion)
                await completion
                case.gather_results(client)
                if case.results.converged:
                    print(case.results.pretty_print())
                update_optimizer(optimizer, case)
                completed_cases += 1
                print()
                if (
                    case.results.converged
                    and case.results.loss is not None
                    and case.results.loss < best_case_loss
                ):
                    best_case = case
                    best_case_loss = case.results.loss

                if best_case is not None:
                    print(
                        f"Completed {completed_cases} cases: best parameters "
                        f"{best_case.parameters.pretty_print()}; "
                        f"{best_case.results.pretty_print()}",
                        flush=True,
                    )

                if next_case_index < max_cases and best_case_loss > convergence:
                    next_case, next_completion = await submit_case()
                    active_cases[next_completion] = next_case

    if best_case is None:
        print("No converged cases completed.", flush=True)
    else:
        print(
            f"Final optimal parameters {best_case.parameters.pretty_print()}; "
            f"{best_case.results.pretty_print()}",
            flush=True,
        )

if __name__ == "__main__":
    asyncio.run(main())
