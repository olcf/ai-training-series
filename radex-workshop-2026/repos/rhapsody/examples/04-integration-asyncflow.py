import asyncio
import multiprocessing as mp
import time

from radical.asyncflow import WorkflowEngine

from rhapsody.backends import DragonExecutionBackend


async def main():
    mp.set_start_method("dragon")

    backend = await DragonExecutionBackend()
    flow = await WorkflowEngine.create(backend=backend)

    @flow.function_task
    async def task1(*args):
        # Simulate lightweight data generation (e.g., creating a list of numbers)
        print("Task 1: Generating data")

        data = list(range(1000))
        return sum(data)  # Simple computation: sum of numbers

    @flow.function_task
    async def task2(*args):
        # Simulate processing data from task1 (e.g., filtering even numbers)
        input_data = args[0]
        print(f"Task 2: Processing data from Task 1, input sum: {input_data}")

        return [x for x in range(1000) if x % 2 == 0]  # Return list of even numbers

    @flow.function_task
    async def task3(*args):
        # Simulate aggregating results from task1 and task2
        sum_data, even_numbers = args
        print(f"Task 3: Aggregating, sum: {sum_data},even count: {len(even_numbers)}")
        # Aggregate results
        return {"total_sum": sum_data, "even_count": len(even_numbers)}

    async def run_wf(wf_id):
        print(f"Starting workflow {wf_id} at {time.time()}")
        t1 = task1()
        t2 = task2(t1)
        t3 = task3(t1, t2)
        result = await t3  # Await the final task
        print(f"Workflow {wf_id} completed at {time.time()}, result: {result}")

    # Run workflows concurrently
    await asyncio.gather(*[run_wf(i) for i in range(1024)])

    await flow.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
