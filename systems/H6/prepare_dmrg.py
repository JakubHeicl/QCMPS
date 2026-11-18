from pathlib import Path
import sys


SYSTEMS_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEMS_DIRECTORY))

from dmrg_common import run_dmrg_preparation


if __name__ == "__main__":
    run_dmrg_preparation(
        system_name="H6",
        system_directory=Path(__file__).resolve().parent,
        fcidump_filename="H6_FCIDUMP",
        reference_filename="H6_reference.json",
        output_filename="H6_dmrg_reference.json",
    )

