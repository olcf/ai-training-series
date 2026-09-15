import os
import pathlib
import shutil
import textwrap
import time

from smartsim import Experiment

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
EXAMPLES_BIN_DIR = ROOT / "install" / "bin" / "examples"


def main() -> int:
    exp_name = "cpp-sr-client-exchange"
    producer_name = "producer"
    consumer_name = "consumer"

    os.environ["SR_LOG_LEVEL"] = "QUIET"
    if (HERE / exp_name).exists():
        shutil.rmtree(HERE / exp_name)

    exp = Experiment(exp_name, launcher="local")

    db = exp.create_database(db_nodes=1, interface="lo")

    producer_settings = exp.create_run_settings(
        os.fspath(EXAMPLES_BIN_DIR / "redis-cpp-producer")
    )
    producer = exp.create_model("producer", producer_settings)

    consumer_settings = exp.create_run_settings(
        os.fspath(EXAMPLES_BIN_DIR / "redis-cpp-consumer")
    )
    consumer = exp.create_model("consumer", consumer_settings)

    exp.generate(db, producer, consumer)
    exp.start(db, block=False)
    time.sleep(3)

    try:
        print("SmartSim is running the producer")
        exp.start(producer, block=True, monitor=False)
        print("SmartSim is running the consumer")
        exp.start(consumer, block=True, monitor=False)
    finally:
        exp.stop(db)

    print("PRODUCER OUTPUT:")
    with open(
        HERE / exp_name / producer_name / f"{producer_name}.out", "r", encoding="utf-8"
    ) as f:
        print(textwrap.indent(f.read(), "    "))

    print("CONSUMER OUTOUT:")
    with open(
        HERE / exp_name / consumer_name / f"{consumer_name}.out", "r", encoding="utf-8"
    ) as f:
        print(textwrap.indent(f.read(), "    "))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
