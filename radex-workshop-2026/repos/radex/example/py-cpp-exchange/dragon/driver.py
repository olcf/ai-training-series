import dataclasses
import os
import pathlib
import textwrap
import time

import dragon
import numpy as np
from dragon.data.ddict import DDict
from dragon.native.process import Process, ProcessTemplate

from radex.clients.core import DragonClient as Client
from radex.handles.handles import IncomingHandle, OutgoingHandle

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
EXAMPLES_BIN_DIR = ROOT / "install" / "bin" / "examples"
DELAY_SET_TIME = 1


def main() -> int:
    dd = DDict(
        managers_per_node=1,
        n_nodes=1,
        trace=False,
        wait_for_keys=True,
        working_set_size=3,
    )
    serial_dd = dd.serialize()
    app_tmpl = ProcessTemplate(
        target=os.fspath(EXAMPLES_BIN_DIR / "dragon-cpp-with-py"),
        env={"SERIALIZED_DDICT": serial_dd},
    )
    app = Process.from_template(app_tmpl)

    print(f"Driver: Making client", flush=True)
    client = Client(serial_dd, 5)

    print(f"Driver: Starting app", flush=True)
    app.start()
    try:
        time.sleep(DELAY_SET_TIME)
        print("Driver: Setting Int", flush=True)
        client.put_scalar(OutgoingHandle("py-int"), 123)

        time.sleep(DELAY_SET_TIME)
        print("Driver: Setting Double", flush=True)
        client.put_scalar(OutgoingHandle("py-double"), 9.87)

        time.sleep(DELAY_SET_TIME)
        print("Driver: Setting Numpy Int", flush=True)
        client.put_scalar(OutgoingHandle("py-np-float"), np.float32(45.6))

        time.sleep(DELAY_SET_TIME)
        print("Driver: Setting Int Tensor", flush=True)
        client.put_tensor(OutgoingHandle("py-int-tensor"), np.arange(4, dtype=np.int32))

        time.sleep(DELAY_SET_TIME)
        print("Driver: Setting Float Tensor", flush=True)
        client.put_tensor(
            OutgoingHandle("py-float-tensor"),
            np.arange(12, dtype=np.float64).reshape((6, 2)),
        )

        print(f"Driver: Looking for keys", flush=True)
        print_scalar(client, "cpp-double")
        print_scalar(client, "cpp-int")
        print_tensor(client, "cpp-double-tensor")
        print_tensor(client, "cpp-long-tensor")

        print("Driver: Setting a py object")
        py_obj_key = "my-py-obj"
        obj = C("spam-and-eggs")
        client.put_picklable(py_obj_key, obj)
        print("Driver: Getting a py object", flush=True)
        recv = client.get_picklable(py_obj_key)
        print(f"Driver: Got object `{recv}`", flush=True)
    finally:
        app.join()

    return 0


def print_scalar(client, key):
    print(f"Driver: Waiting for scalar key `{key}`", flush=True)
    scalar = client.wait_for_scalar(IncomingHandle(key), 10)
    print(
        textwrap.dedent(f"""\
        Driver: Got scalar:
                |- Type: {scalar.dtype}
                \\- Value: {scalar}
        """),
        flush=True,
    )


def print_tensor(client, key):
    print(f"Driver: Waiting for tensor key `{key}`", flush=True)
    tensor = client.wait_for_tensor(IncomingHandle(key), 10)
    print(
        textwrap.dedent(f"""\
        Driver: Got tensor:
                |- Type: {tensor.dtype}
                |- Dims: {tensor.shape}
                \\- Data: {tensor.ravel()}
        """),
        flush=True,
    )


@dataclasses.dataclass(frozen=True)
class C:
    msg: str


if __name__ == "__main__":
    raise SystemExit(main())
