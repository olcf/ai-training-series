import asyncio
import multiprocessing as mp

from rhapsody import Session
from rhapsody.api import ComputeTask
from rhapsody.backends import DragonExecutionBackendV3


async def main():

    mp.set_start_method("dragon", force=True)

    backend = await DragonExecutionBackendV3(
        name="dragon"
    )

    async with Session(backends=[backend]) as session:

        ###############################################################
        # GPU APPLICATION
        #
        # Starts immediately and remains active for the duration
        ###############################################################

        gpu_task = ComputeTask(
            executable="python",
            arguments=[
                "gpu_service.py",
                "--store-host", "redis.service",
                "--store-port", "6379",
                "--input-queue", "simulation_features",
                "--output-queue", "predictions"
            ],
            backend=backend.name,

            task_backend_specific_kwargs={
                "partition": "gpu_partition",
                "process_template": {
                    "env": {
                        "CUDA_VISIBLE_DEVICES": "0"
                    }
                }
            }
        )        #
        await session.submit_tasks([gpu_task])
        ###############################################################
        # MPI PREPROCESSING PIPELINE
        ###############################################################

        preprocess_mesh = ComputeTask(
            executable="python",
            arguments=[
                "generate_mesh.py",
                "--output", "mesh.h5"
            ],
            backend=backend.name,

            task_backend_specific_kwargs={
                "partition": "preprocess_partition"
            }
        )

        await session.submit_tasks([preprocess_mesh])

        await preprocess_mesh

        if preprocess_mesh.state != "DONE":
            raise RuntimeError("Mesh generation failed")

        print("Mesh generation complete")

        ###############################################################

        partition_mesh = ComputeTask(
            executable="python",
            arguments=[
                "partition_mesh.py",
                "mesh.h5",
                "--output",
                "mesh.part"
            ],
            backend=backend.name,

            task_backend_specific_kwargs={
                "partition": "preprocess_partition"
            }
        )

        await session.submit_tasks([partition_mesh])

        await partition_mesh

        if partition_mesh.state != "DONE":
            raise RuntimeError("Mesh partitioning failed")

        print("Mesh partitioning complete")

        ###############################################################

        generate_inputs = ComputeTask(
            executable="python",
            arguments=[
                "generate_solver_inputs.py",
                "mesh.part",
                "--output",
                "solver.input"
            ],
            backend=backend.name,

            task_backend_specific_kwargs={
                "partition": "preprocess_partition"
            }
        )

        await session.submit_tasks([generate_inputs])

        await generate_inputs

        if generate_inputs.state != "DONE":
            raise RuntimeError("Input generation failed")

        print("Solver inputs ready")

        ###############################################################
        # MPI APPLICATION
        #
        # Starts only after preprocessing succeeds
        ###############################################################

        mpi_task = ComputeTask(
            executable="mpirun",
            arguments=[
                "-n", "128",

                "./solver",

                "solver.input",

                "--store-host", "redis.service",
                "--store-port", "6379",

                "--send-queue",
                "simulation_features",

                "--recv-queue",
                "predictions"
            ],
