# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to generate SLURM job scripts from Jinja2 templates.
"""

import argparse
import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional

from jinja2 import Template


@dataclass
class SlurmStep:
    """Represents a step in the interactive workflow."""

    job_id: str
    description: str
    command: Optional[tuple[str, ...]] = None
    print_command: bool = False

    def describe(self) -> str:
        logging.info(self.description)
        if self.print_command and self.command:
            logging.info(f"    {' '.join(self.command)}")

    def execute(self) -> None:
        if self.command:
            result = subprocess.run(
                self.command, capture_output=True, text=True, check=True
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            logging.info(f"stdout: {stdout}")
            logging.info(f"stderr: {stderr}")
        else:
            logging.info("Nothing to execute.")


class WaitForAllocation(SlurmStep):
    """Wait for the nodes to be allocated."""

    wait_time: int = 10  # seconds

    def execute(self) -> None:
        while True:
            result = subprocess.run(
                ["squeue", "-j", self.job_id, "-h", "-o", "%t"],
                capture_output=True,
                text=True,
                check=True,
            )
            output_lines = result.stdout.strip().split("\n")
            if len(output_lines) < 1:
                raise RuntimeError("No output from squeue. Exiting.")
            elif output_lines[0] == "R":
                logging.info("Nodes allocated. Exiting.")
                break
            elif output_lines[0].startswith("P"):
                logging.info("Waiting for nodes to be allocated...")
            elif output_lines[0].startswith("C"):
                raise RuntimeError("Job was cancelled")
            else:
                raise RuntimeError(f"Unknown state: {output_lines[0]}")
            time.sleep(self.wait_time)


class WaitForSetup(SlurmStep):
    """Wait for the setup to complete and for GPUs to be idle."""

    def execute(self) -> None:
        logging.info("Waiting for setup to complete and for GPUs to be idle...")
        logging.info("Not implemented yet.")


@dataclass
class SlurmConfig:
    """Configuration class for SLURM job management."""

    job_id: str
    timeout: int = 3600  # seconds # TODO

    def create_execution_steps(self) -> List[SlurmStep]:
        """Create the sequence of SLURM steps for the interactive workflow."""
        return [
            WaitForAllocation(
                job_id=self.job_id,
                description="1. Wait for the nodes to be allocated",
            ),
            WaitForSetup(
                job_id=self.job_id,
                description="2. Wait for setup to complete and for GPUs to be idle",
            ),
            SlurmStep(
                job_id=self.job_id,
                description="3. Run warmup on the prefill host:",
                command=(
                    "srun",
                    "--jobid",
                    self.job_id,
                    "--overlap",
                    "bash",
                    "utils/bench.sh",
                    "$PREFILL_HOST_IP",
                    "--type",
                    "warmup",
                ),
                print_command=True,
            ),
        ]

    def create_cleanup_steps(self) -> List[SlurmStep]:
        """Create the cleanup steps for the interactive workflow."""
        return [
            SlurmStep(
                job_id=self.job_id,
                description="Cancel the job:",
                command=("scancel", self.job_id),
                print_command=True,
            ),
        ]

    def execute_interactive_workflow(self) -> None:
        """Execute the complete interactive workflow."""
        logging.info("Entering interactive mode")
        logging.info("If the connection is lost, make sure to cancel the job manually:")
        logging.info(f"    scancel {self.job_id}")

        steps = self.create_execution_steps()
        cleanup_steps = self.create_cleanup_steps()

        try:
            logging.info("--------------------------------")
            logging.info("The following steps will be executed:")
            for step in steps:
                step.describe()
            logging.info("The following steps will be executed when the job is done:")
            for step in cleanup_steps:
                step.describe()
            logging.info("--------------------------------")

            for step in steps:
                step.execute()
        except Exception as e:
            logging.error(f"Error executing step: {e}")
            raise
        finally:
            logging.info("--------------------------------")
            logging.info("Running cleanup steps:")
            for step in cleanup_steps:
                step.execute()
            logging.info("--------------------------------")
            logging.info("Interactive mode completed")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s| %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def generate_job_script(template_path, output_path, **kwargs):
    """Generate a job script from template with given parameters."""
    with open(template_path, "r") as f:
        template = Template(f.read())

    rendered_script = template.render(**kwargs)
    with open(output_path, "w") as f:
        f.write(rendered_script)

    return output_path


def submit_job(job_script_path):
    """
    Submit the job script to SLURM and extract the job ID from the output.

    Returns:
        The job ID of the submitted job.
    """
    try:
        result = subprocess.run(
            ["sbatch", job_script_path], capture_output=True, text=True, check=True
        )
        output_lines = result.stdout.strip().split("\n")

        # sbatch typically outputs: "Submitted batch job JOBID"
        job_id = output_lines[-1].split()[-1]
        logging.info(f"Job submitted successfully with ID: {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        logging.error(f"Error submitting job: {e}")
        logging.error(f"stderr: {e.stderr}")
        raise
    except (IndexError, ValueError):
        logging.error(f"Error parsing job ID from sbatch output: {result.stdout}")
        raise


def _parse_command_line_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and submit SLURM job scripts"
    )
    parser.add_argument(
        "--template", required=True, help="Path to Jinja2 template file"
    )

    # Template parameters
    parser.add_argument("--job-name", default="dynamo_setup", help="SLURM job name")
    parser.add_argument("--account", required=True, help="SLURM account")
    parser.add_argument("--model-dir", required=True, help="Model directory path")
    parser.add_argument("--config-dir", required=True, help="Config directory path")
    parser.add_argument("--container-image", required=True, help="Container image")
    parser.add_argument(
        "--time-limit", default="01:00:00", help="Time limit (HH:MM:SS)"
    )
    parser.add_argument(
        "--prefill-nodes", type=int, default=2, help="Number of prefill nodes"
    )
    parser.add_argument(
        "--decode-nodes", type=int, default=2, help="Number of decode nodes"
    )
    parser.add_argument(
        "--gpus-per-node", type=int, default=8, help="Number of GPUs per node"
    )
    parser.add_argument(
        "--network-interface", default="eth3", help="Network interface to use"
    )
    parser.add_argument(
        "--gpu-type", choices=["h100", "gb200"], default="h100", help="GPU type to use"
    )
    parser.add_argument(
        "--use-sglang-commands",
        action="store_true",
        default=False,
        help="Use SGLang commands instead of Dynamo",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Run the job interactively",
    )
    return parser.parse_args(args)


def main(input_args: list[str] | None = None):
    setup_logging()
    args = _parse_command_line_args(input_args)

    total_nodes = args.prefill_nodes + args.decode_nodes
    template_vars = {
        "job_name": args.job_name,
        "total_nodes": total_nodes,
        "account": args.account,
        "time_limit": args.time_limit,
        "prefill_nodes": args.prefill_nodes,
        "decode_nodes": args.decode_nodes,
        "model_dir": args.model_dir,
        "config_dir": args.config_dir,
        "container_image": args.container_image,
        "gpus_per_node": args.gpus_per_node,
        "network_interface": args.network_interface,
        "gpu_type": args.gpu_type,
        "use_sglang_commands": args.use_sglang_commands,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh") as temp_file:
        generate_job_script(args.template, temp_file.name, **template_vars)
        job_id = submit_job(temp_file.name)
        logging.info(f"Job logs will be available in: logs/{job_id}/")

    if args.interactive:
        slurm_config = SlurmConfig(
            job_id=job_id,
        )

        slurm_config.execute_interactive_workflow()


if __name__ == "__main__":
    main()
