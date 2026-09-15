import asyncio
import multiprocessing as mp

from rhapsody.backends import DragonExecutionBackend

import rhapsody_plugins.openfoam as rof

def create_openfoam_case():
    case = rof.CaseDefinition("../openfoam-cases/airFoil2D", "./run", clean=True)
    reg = rof.OFExecutableRegistry()

    # Preprocessing
    decompose = reg.decomposePar()
    case.add_stage("preprocessing", decompose)

    # Solvers
    simplefoam = reg.simpleFoam(num_ranks=8)
    case.add_stage("solve", simplefoam)

    # Postprocessing
    reconstruct = reg.reconstructPar()
    case.add_stage("postprocessing", reconstruct)

    return case

async def main():

    mp.set_start_method("dragon", force=True)

    backend = await DragonExecutionBackend()
    case = create_openfoam_case()

    async with rof.OFSession(backends=[backend]) as session:
        print("Preprocessing stage")
        await case.stages["preprocessing"].execute(session)

        print("Solver stage")
        await case.stages["solve"].execute(session)

        print("Postprocessing stage")
        await case.stages["postprocessing"].execute(session)

if __name__ == "__main__":
    asyncio.run(main())
