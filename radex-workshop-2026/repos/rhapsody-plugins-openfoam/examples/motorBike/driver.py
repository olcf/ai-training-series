import asyncio
import multiprocessing as mp
import shutil

from pathlib import Path

from rhapsody.backends import DragonExecutionBackend

import rhapsody_plugins.openfoam as rof

CASE_DIR = Path("../openfoam-cases/motorbike")
GEOMETRY = Path("../openfoam-cases/resources/motorBike.obj.gz")
DECOMPOSE_PAR_DICT_6 = "system/decomposeParDict.6"

def define_stages(case, reg):

    num_subdomains, _ = reg.foamDictionary(
        args=DECOMPOSE_PAR_DICT_6,
        options={
            "value": None,
            "entry": "numberOfSubdomains",
        }
    ).run_local(cwd=case.run_path)
    num_subdomains = int(num_subdomains)

    decompose_option = {
        "decomposeParDict": DECOMPOSE_PAR_DICT_6
    }

    # Mesh generation: surfaceExtract and initial decomposition
    mesh_stage_1 = rof.OFStage()
    mesh_stage_1.add_tasks(
        [
            reg.surfaceFeatureExtract(),
            reg.blockMesh(),
            reg.decomposePar(options=decompose_option),
        ]
    )
    case.add_stage("mesh-generation-1", mesh_stage_1)

    mesh_stage_2 = rof.OFStage()
    mesh_stage_2.add_tasks(
        [
            reg.snappyHexMesh(
                options=decompose_option | {"overwrite": None},
                num_ranks=num_subdomains
            ),
            reg.topoSet(
                options=decompose_option,
                num_ranks=num_subdomains
            )
        ]
    )
    case.add_stage("mesh-generation-2", mesh_stage_2)

    init_stage = rof.OFStage()
    init_stage.add_tasks(
        [
            reg.RunFunctions(
                "restore0Dir",
                options={"processor": None}
            ),
            reg.patchSummary(
                num_ranks=num_subdomains,
                options=decompose_option,
            ),
            reg.potentialFoam(
                num_ranks=num_subdomains,
                options={"writePhi": None} | decompose_option,
            ),
            reg.checkMesh(
                num_ranks=num_subdomains,
                options={
                    "writeFields": "(nonOrthoAngle)",
                    "constant": None,
                }
            )
        ]
    )
    case.add_stage("initialization", init_stage)
    case.add_stage(
        "Solver",
        reg.simpleFoam(
            num_ranks=num_subdomains,
            options=decompose_option
        )
    )
    postprocess = rof.OFStage()
    postprocess.add_tasks(
        [
            reg.reconstructParMesh(options={"constant": None}),
            reg.reconstructPar(options={"latestTime": None})
        ]
    )
    case.add_stage("reconstruct", postprocess)
    return case

async def main():

    reg = rof.OFExecutableRegistry()
    case = rof.CaseDefinition("../openfoam-cases/motorBike", "./run", clean=True)
    dest = case.run_path / "constant" / "triSurface"
    dest.mkdir()
    shutil.copy(GEOMETRY, dest)
    case = define_stages(case, reg)

    mp.set_start_method("dragon", force=True)
    backend = await DragonExecutionBackend()

    async with rof.OFSession(backends=[backend]) as session:
        for stage_name, stage in case.stages.items():
            print(f"Running stage: {stage_name}")
            await stage.execute(session)

if __name__ == "__main__":
    asyncio.run(main())
