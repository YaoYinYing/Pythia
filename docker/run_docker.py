# yinying edit this file from deepmind alphafold repo

"""Docker launch script for Pythia docker image."""

import os
import pathlib
import signal

import docker
from absl import app
from absl import flags
from docker import types
from typing import Tuple

flags.DEFINE_string("input_dir", None, "Input directory path.")


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Try to use cpu")

flags.DEFINE_integer("n_jobs", os.cpu_count(), "Number of parallel jobs")


flags.DEFINE_string("pdb_filename", None, "Path to a specific PDB filename.")


flags.DEFINE_bool(
    "check_plddt", True, "Generate a human-friendly report file in xlsx format"
)

flags.DEFINE_float("plddt_cutoff", 95, "pLDDT cutoff value.")

flags.DEFINE_string(
    "docker_image_name", "pythia-wubianlab", "Name of the Pythia Docker image."
)

flags.DEFINE_string(
    "docker_user",
    f"{os.geteuid()}:{os.getegid()}",
    "UID:GID with which to run the Docker container. The output directories "
    "will be owned by this user:group. By default, this is the current user. "
    "Valid options are: uid or uid:gid, non-numeric values are not recognised "
    "by Docker unless that user has been created within the container.",
)

FLAGS = flags.FLAGS

try:
    _ROOT_MOUNT_DIRECTORY = f"/home/{os.getlogin()}"
except:
    _ROOT_MOUNT_DIRECTORY = pathlib.Path("/tmp/").resolve()
    os.makedirs(_ROOT_MOUNT_DIRECTORY, exist_ok=True)


def _create_mount(mount_name: str, path: str) -> Tuple[types.Mount, str]:
    """Create a mount point for each file and directory used by the model."""
    path = pathlib.Path(path).absolute()
    target_path = pathlib.Path(_ROOT_MOUNT_DIRECTORY, mount_name)

    if path.is_dir():
        source_path = path
        mounted_path = target_path
    else:
        source_path = path.parent
        mounted_path = pathlib.Path(target_path, path.name)
    if not source_path.exists():
        raise ValueError(
            f'Failed to find source directory "{source_path}" to '
            "mount in Docker container."
        )
    print("Mounting %s -> %s", source_path, target_path)
    mount = types.Mount(
        target=str(target_path), source=str(source_path), type="bind", read_only=True
    )
    return mount, str(mounted_path)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    mounts = []
    command_args = []

    # os.makedirs(save_dir, exist_ok=True)
    if not FLAGS.input_dir and not FLAGS.pdb_filename:
        raise app.UsageError("Please specify input pdb file or path!")
    if FLAGS.input_dir:
        input_dir = pathlib.Path(FLAGS.input_dir).resolve()
        input_target_path = os.path.join(_ROOT_MOUNT_DIRECTORY, "input")
        mounts.append(types.Mount(input_target_path, str(input_dir), type="bind"))
        command_args.append(f"--input_dir={input_target_path}")
    elif FLAGS.pdb_filename:
        pdb_filename = pathlib.Path(FLAGS.pdb_filename).resolve()
        input_target_path = os.path.join(_ROOT_MOUNT_DIRECTORY, "input")
        mounts.append(types.Mount(input_target_path, str(pdb_filename), type="bind"))
        command_args.append(f"--pdb_filename={input_target_path}")

    command_args.extend(
        [
            f"--device={FLAGS.device}",
            f"--n_jobs={FLAGS.n_jobs}",
        ]
    )
    if FLAGS.check_plddt:
        command_args.extend(
            [
                f"--check_plddt",
                f"--plddt_cutoff={FLAGS.plddt_cutoff}",
            ]
        )

    print(command_args)

    client = docker.from_env()

    container = client.containers.run(
        image=FLAGS.docker_image_name,
        command=command_args,
        remove=True,
        detach=True,
        mounts=mounts,
        user=FLAGS.docker_user,
    )

    # Add signal handler to ensure CTRL+C also stops the running container.
    signal.signal(signal.SIGINT, lambda unused_sig, unused_frame: container.kill())

    for line in container.logs(stream=True):
        print(line.strip().decode("utf-8"))


if __name__ == "__main__":
    app.run(main)
