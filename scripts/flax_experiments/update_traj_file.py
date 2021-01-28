"""Copies a dataset; for migrating to newer hdf5 library versions.
"""

import dataclasses

import datargs
import fannypack
from tqdm.auto import tqdm


@dataclasses.dataclass
class Args:
    input_path: str
    output_path: str


args: Args = datargs.parse(Args)

with fannypack.data.TrajectoriesFile(args.input_path) as input_file:
    with fannypack.data.TrajectoriesFile(
        args.output_path, read_only=False
    ) as output_file:
        output_file.resize(len(input_file))
        for i, traj in enumerate(tqdm(input_file)):
            output_file[i] = traj
