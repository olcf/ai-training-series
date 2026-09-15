## Installing OpenFOAM

OpenFOAM is a real world, free, and open source CFD modeling software. We will
be using this to show how RHAPSODY can be integrated into an actual, industry
standard application and highlight a plugin package (to be installed at a later
date) to assist with this integration.

For tdoay, we simiply need to install OpenFOAM and some third party software.

To start, we will begin by cloning the necessary repositories

```sh
$ git clone -b OpenFOAM-v2506 \
      --single-branch https://gitlab.com/openfoam/core/openfoam.git \
      OpenFOAM-v2506
$ git clone -b v2506 \
      --single-branch https://gitlab.com/openfoam/core/thirdparty-common.git \
      OpenFOAM-v2506/ThirdParty
$ pushd OpenFOAM-v2506
```

We will then need to create and/or edit the `etc/prefs.sh` in the OpenFOAM
repository to contain the following:

```
# If you get an error from this command, try using
# `module swap PrgEnv-cray PrgEnv-gnu` instead
module load PrgEnv-gnu

export WM_COMPILER=Gcc
export WM_MPLIB=CRAY-MPICH
export I_MPI_CC=cc
export I_MPI_CXX=CC
```

Finally, simple source the `etc/bashrc` file (this is where you could see the
potential module conflict commented on above) and run the `Allwmake` script.

> [!CAUTION]
> This step may take a while. If you attempting to install this on your own, it
> might be worth setting this up in a screen so that you do not need to baby
> sit it. 

```sh
$ source etc/bashrc
$ ./Allwmake -j 64  # or some other number of cores CPU cores
```

Once it is done, we can return to the installation root.

```sh
$ popd
```
