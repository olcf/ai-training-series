import os
import pathlib
import sys


def test_run_example(example, tmp_path, monkeypatch):
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # FIXME: Ideally we can remove this env futzing when we figure out how to
    #        properly set the rpath on the examples such that `libdragon.so`
    #        does not need to be present on
    #        `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH`. For now we can just add it
    #        based on the install site of the dragon package. There is likely a
    #        similar error with `libsmartredis.so` as well, but our CI appends
    #        the path to the library look up env var already, so we do not need
    #        to do that here.
    # =================================================================================
    ld_lib_path = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    try:
        import dragon
    except ImportError:
        pytest.xfail(
            reason=(
                f"Dragon library `libdragon.so` must be on {ld_lib_path} "
                "to run the examples due to unresolved linking errors"
            )
        )
    else:
        dragon_path, *_ = dragon.__path__
        drg_libs_paths = ":".join(
            os.fspath(p) for p in pathlib.Path(dragon_path).absolute().glob("lib*")
        )

        prev_ld_lib_path = os.environ.get(ld_lib_path, "")
        new_ld_lib_path = (
            f"{drg_libs_paths}:{prev_ld_lib_path}" if prev_ld_lib_path else dragon_libs
        )
        monkeypatch.setenv(ld_lib_path, new_ld_lib_path)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    returncode, out, err = example.run(where=tmp_path)
    assert returncode == 0
    with (
        open(out, "r", encoding="utf-8") as fh,
        open(example.expected_stdout, "r", encoding="utf-8") as xfh,
    ):
        for line, xline in zip(fh, xfh):
            assert line.strip() == xline.strip()
