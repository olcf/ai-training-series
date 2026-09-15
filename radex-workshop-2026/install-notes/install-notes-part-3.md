## Installing Rhapsody Plugins (OpenFOAM)

This is the last piece of software we need to install for the RHAPSODY/radex
demo. To start, navigate to the shared install directory we used the previous
two days and set up your build environment.

```sh
$ cd <your-install-dir>
$ source ./venv/bin/activate
$ source OpenFoam-2506/etc/bashrc
```

To check that the environment is setup correctly, make sure that you can import
rhapsody and that `wmake` is on your path.

```sh
$ python -c 'import rhapsody; print("RHAPSODY is installed")'
$ which wmake
```

Next we'll need to clone down the RHAPSODY plugin for OpenFOAM:

```sh
$ git clone https://github.com/CrayLabs/rhapsody-plugins-openfoam.git rhapsody-plugins-openfoam
```

Next we need to build the plugin extension modules. To do this, we need to tell
the plugin where we installed we previously installed the `radex` and
`smartredis` libraries. We do this by setting the `RADEX_DIR` and
`SMARTREDIS_DIR` environment variables (note that this only needs to be done
for building the extension, not for running).

Once that is done, we simply need to move to the cloned repository and run the
`Allwmake` script.

```sh
$ export RADEX_DIR="$(pwd)/radex/install"
$ export SMARTREDIS_DIR="$(pwd)/smartredis/install"

$ cd rhapsody-plugins-openfoam
$ ./Allwmake
```

At this point you should be be able to see the following libraries installed in
your `FOAM_USER_LIBBIN` directory and have the following python packages
available in your environment.

```sh
$ ls "${FOAM_USER_LIBBIN}"
libradexBase.so  libradexIO.so  libradexRead.so  libradexWrite.so
$ pip list | grep rhapsody
rhapsody-plugins.openfoam 0.1.0
rhapsody-py               0.4.0
```

Finally, to confirm that everything is installed correctly, we will run the
`pitzDaily-optimize` example. We will begin by installing one more dependency
needed for that example, a python package named `scikit-optimize`.

```sh
$ pip install scikit-optimize
```

Next simply navigate to the example and locate the driver script,
`driver-staged.py`. This script will run `pitzDaily` OpenFOAM tutorial using
dragon and radex for any communication necessary. Like the other examples that
we planned to share, this should be runnable with a single node without HSTA or
global services.

```sh
cd example/pitzDaily-optimize
dragon -s -- ./driver-staged.py
```

This should iterate for 4 or 5 iterations and then exit. As long as no
obviously incorrect errors arise, you should be fully set up!!
