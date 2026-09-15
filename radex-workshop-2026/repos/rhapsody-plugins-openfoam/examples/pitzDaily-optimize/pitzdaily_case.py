import os
from dataclasses import dataclass
from math import sqrt
from typing import TYPE_CHECKING, Optional

import numpy as np

from rhapsody_plugins.openfoam import CaseDefinition
from rhapsody_plugins.openfoam.utils import OFKey

from radex.handles.handles import IncomingHandle

if TYPE_CHECKING:
    from radex.clients.core import DragonClient


OPENFOAM_CASES=os.getenv("OPENFOAM_CASE_DIR", default="../openfoam-cases")


@dataclass
class KEpsilonParameters:
    epsilon: float
    Cmu: float
    C1: float
    C2: float
    kappa: float = 0.375

    @property
    def sigma_epsilon(self):
        return sqrt(self.kappa) / (sqrt(self.Cmu) * (self.C2 - self.C1))

    def as_list(self):
        return [
            self.epsilon,
            self.Cmu,
            self.C1,
            self.C2,
            self.sigma_epsilon,
        ]

    def as_array(self):
        return np.array(self.as_list())

    def as_optimizer_list(self):
        return [self.epsilon, self.Cmu, self.C1, self.C2]

    def pretty_print(self):
        return (
            f"epsilon={self.epsilon:.6g}, Cmu={self.Cmu:.6g}, "
            f"C1={self.C1:.6g}, C2={self.C2:.6g}"
        )


@dataclass
class pitzDailyResults:
    final_step: Optional[int]
    avg_inlets: Optional[float]
    loss: Optional[float]
    converged: bool

    def pretty_print(self):
        if self.avg_inlets is None:
            return "converged=False, results unavailable"
        return (
            f"converged={self.converged}, final_step={self.final_step}, "
            f"avg_inlets={self.avg_inlets:.6g}, loss={self.loss:.6g}"
        )


class pitzDailyCase(CaseDefinition):
    def __init__(
        self,
        registry,
        parameters,
        identifier,
        radex_store,
        target_avg_inlets,
        clean=True,
    ):
        self.parameters = parameters
        self.identifier = identifier
        self.target_avg_inlets = target_avg_inlets
        super().__init__(
            f"{OPENFOAM_CASES}/pitzDaily",
            f"./run-{identifier}",
            clean=clean,
        )
        self._modify_dictionaries(registry)
        simplefoam = registry.simpleFoam(execute_async=True, radex_store=radex_store)
        self.add_stage("solve", simplefoam)

    def _modify_dictionaries(self, registry):
        epsilon_dict = "0/epsilon"
        turbulence_dict = "constant/turbulenceProperties"
        control_dict = "system/controlDict"

        registry.foamDictionary(
            args=[epsilon_dict],
            options={
                "entry": "internalField",
                "set": f"uniform {self.parameters.epsilon}",
            },
        ).run_local(cwd=self.run_path)

        for patch in ["inlet", "upperWall", "lowerWall"]:
            registry.foamDictionary(
                args=[epsilon_dict],
                options={
                    "entry": f"boundaryField.{patch}.value",
                    "set": f"uniform {self.parameters.epsilon}",
                },
            ).run_local(cwd=self.run_path)

        for coefficient, value in {
            "Cmu": self.parameters.Cmu,
            "C1": self.parameters.C1,
            "C2": self.parameters.C2,
            "sigmaEps": self.parameters.sigma_epsilon,
        }.items():
            registry.foamDictionary(
                args=[turbulence_dict],
                options={"entry": f"RAS.{coefficient}", "set": value},
            ).run_local(cwd=self.run_path)

        registry.foamDictionary(
            args=[control_dict],
            options={
                "entry": "functions.radexWrite.identifier",
                "set": self.identifier,
            },
        ).run_local(cwd=self.run_path)

    def gather_results(
        self,
        client: "DragonClient",
        max_iter: int = 2000,
    ):
        try:
            final_step = client.get_scalar(
                IncomingHandle(OFKey(self.identifier, "final_step")))
            avg_inlets = client.get_scalar(
                IncomingHandle(OFKey(self.identifier, "avgInlets", time=final_step, rank=0))
            )
            results = pitzDailyResults(
                final_step=int(final_step),
                avg_inlets=avg_inlets,
                loss=(avg_inlets - self.target_avg_inlets) ** 2,
                converged=final_step < max_iter,
            )
        except:
            print(f"WARNING: {self.identifier} failed", flush=True)
            results = pitzDailyResults(
                final_step=None,
                avg_inlets=None,
                loss=None,
                converged=False,
            )
        self.results = results
