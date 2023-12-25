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

flags.DEFINE_string("pdb_filename", None, "Path to a specific PDB filename.")
flags.DEFINE_string("save_dir", "./pythia_predictions", "Saving directory path.")


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Try to use cpu")

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


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    mounts = []
    command_args = []

    # os.makedirs(save_dir, exist_ok=True)
    if FLAGS.pdb_filename:
        pdb_filename = pathlib.Path(FLAGS.pdb_filename).resolve()
        input_target_path = os.path.join(_ROOT_MOUNT_DIRECTORY, "input", os.path.basename(pdb_filename))
        mounts.append(types.Mount(input_target_path, str(pdb_filename), type="bind"))
        command_args.append(f"--pdb_filename={input_target_path}")

    save_dir = pathlib.Path(FLAGS.save_dir).resolve()

    os.makedirs(save_dir, exist_ok=True)
    output_target_path = os.path.join(_ROOT_MOUNT_DIRECTORY, "output")
    mounts.append(types.Mount(output_target_path, str(save_dir), type="bind"))
    command_args.append(f"--save_dir={output_target_path}")

    command_args.extend(
        [
            f"--device={FLAGS.device}",
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
    flags.mark_flags_as_required([
        'pdb_filename',
    ])
    app.run(main)
