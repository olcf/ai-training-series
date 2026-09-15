import os
import pathlib

import dragon
from dragon.data.ddict import DDict
from dragon.native.process import Process, ProcessTemplate

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
EXAMPLES_BIN_DIR = ROOT / "install" / "bin" / "examples"


def main() -> int:
    dd = DDict(
        managers_per_node=1,
        n_nodes=1,
        trace=False,
        wait_for_keys=True,
        working_set_size=4,
    )
    serial_dd = dd.serialize()
    producer_tmpl = ProcessTemplate(
        target=os.fspath(EXAMPLES_BIN_DIR / "dragon-cpp-producer"),
        env={"RADEX_STORE": serial_dd},
    )
    consumer_tmpl = ProcessTemplate(
        target=os.fspath(EXAMPLES_BIN_DIR / "dragon-cpp-consumer"),
        env={"RADEX_STORE": serial_dd},
    )

    print("==> Running Producer...", flush=True)
    producer = Process.from_template(producer_tmpl)
    producer.start()
    producer.join()
    print("==> Producer Joined", flush=True)

    print("==> Running Consumer...", flush=True)
    consumer = Process.from_template(consumer_tmpl)
    consumer.start()
    consumer.join()
    print("==> Consumer Joined", flush=True)
    dd.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
