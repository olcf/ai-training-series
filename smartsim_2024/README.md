# SmartSim Workshop 2024

Advancements in AI and AI accelerators are opening up a new avenue of research which combines data-driven techniques and scientific simulations. These hybrid applications often are more easily described by a loosely-coupled set of components (of which the simulation may be one) instead of a pipeline-like workflow common in traditional scientific computations. The ability to describe and execute these workflows along with the need to share data among the components can be a significant barrier to entry for domain scientists. HPE’s SmartSim is an open-source software library that seeks to address these challenges by providing an in-memory, distributed feature store for data storage and exchange, lightweight communication clients embeddable in C, C++, Fortran, and Python, and a Python library allowing users to compose complex workflows.

This workshop will showcase recent collaborations using SmartSim that are representative of emerging AI/HPC applications:

* applying Bayesian optimization to an ensemble of simulations to tune the parameters of a turbulence model
* on-the-fly training and inference to replace the mesh motion solver in OpenFOAM, a widely-used, open-source computational fluid dynamics solver and
* intelligently sampling data streamed from a simulation to enable online-training at-scale for a high-resolution simulation of particle fluidization using MFIX-Exa, a Department of Energy, exascale-capable multiphase flow solver.

In addition to showcasing the capabilities, this presentation will also highlight the challenges and considerations unique to AI/HPC and how they compare to other common AI applications. Immediately following the presentations, interested users will have the opportunity to run a trimmed down version of the data-selection and online-training workflow and learn how to adapt it for their own simulation codes.

Specific examples of how to use SmartSim on Frontier (OLCF) and Perlmutter (NERSC) will be provided. 

* Tutorial link: https://github.com/CrayLabs/smartsim_workshops/tree/nersc_olcf_2024
* Recording: https://vimeo.com/1013517492
* Slides: https://www.olcf.ornl.gov/wp-content/uploads/SmartSim_NERSC_OLCF_Workshop2024.pdf
