"""Utility for generating a BackpropKF-style disc tracking dataset.

Based on code from:
> https://github.com/akloss/differentiable_filters
"""
import dataclasses
import random
from typing import List, Tuple

import cv2
import fannypack as fp
import numpy as onp
from matplotlib import pyplot as plt

RgbColor = Tuple[int, int, int]


@dataclasses.dataclass(frozen=True)
class Disk:
    radius: int
    position: onp.ndarray
    """Position of disk. Origin at center of image."""

    velocity: onp.ndarray
    color: RgbColor

    def __post_init__(self):
        assert self.position.shape == self.velocity.shape == (2,)


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    sequences_per_file: int
    sequence_length: int
    num_files: int


@dataclasses.dataclass(frozen=True)
class ToySystem:
    image_size: int
    num_distractors: int

    spring_constant: float
    drag_constant: float

    position_noise_std: float
    velocity_noise_std: float

    def initialize_disks(self) -> List[Disk]:
        """Helper for initializing a list of disks. First one will be red."""

        # Red disk -- this is the one we care about!
        red_disk = Disk(
            radius=7,
            position=onp.random.uniform(
                -self.image_size / 2.0, self.image_size / 2.0, size=(2,)
            ),
            velocity=onp.random.normal(loc=0.0, scale=3.0, size=(2,)),
            color=(255, 0, 0),
        )
        color_options: List[RgbColor] = [
            (0, 255, 0),
            (0, 0, 255),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
            (255, 255, 255),
        ]

        # Create list of disks
        disks: List[Disk] = [red_disk]
        for i in range(self.num_distractors):
            disks.append(
                Disk(
                    radius=onp.random.randint(3, 10),
                    position=onp.random.uniform(
                        -self.image_size / 2.0, self.image_size / 2.0, size=(2,)
                    ),
                    velocity=onp.random.normal(loc=0.0, scale=3.0, size=(2,)),
                    color=random.choice(color_options),
                )
            )
        return disks

    def observation_model(self, disks: List[Disk]) -> Tuple[onp.ndarray, int]:
        """Helper for rendering disks. Returns 2D image array + # of visible red pixels."""

        image = onp.zeros((self.image_size, self.image_size, 3), dtype=onp.uint8)

        for disk in disks:
            cv2.circle(
                img=image,
                center=tuple((disk.position + self.image_size / 2.0).astype(onp.int32)),
                radius=disk.radius,
                color=disk.color,
                thickness=-1,
            )

        return image, onp.count_nonzero((image == (255, 0, 0)).all(axis=2))

    def process_model(self, disk: Disk) -> Disk:
        """Process model for a single disk."""

        # Compute external forces
        spring_force = -self.spring_constant * disk.position
        drag_force = (
            -self.drag_constant * onp.sign(disk.velocity) * (disk.velocity ** 2)
        )

        # Sample process noises
        position_noise = onp.random.normal(
            loc=0.0, scale=self.position_noise_std, size=(2,)
        )
        velocity_noise = onp.random.normal(
            loc=0.0, scale=self.velocity_noise_std, size=(2,)
        )

        # Update positions, velocities & return
        position_updated = disk.position + disk.velocity + velocity_noise
        velocity_updated = disk.velocity + spring_force + drag_force + position_noise

        return dataclasses.replace(
            disk, position=position_updated, velocity=velocity_updated
        )


config = DatasetConfig(
    sequences_per_file=1,
    sequence_length=10,
    num_files=5,
)
system = ToySystem(
    image_size=120,
    num_distractors=5,
    spring_constant=0.05,
    drag_constant=0.0075,
    position_noise_std=0.1,
    velocity_noise_std=2.0,
)
output_dir = "data"

## Code for previewing generated dataset
#
# from celluloid import Camera
#
# fig = plt.figure()
# camera = Camera(fig)
#
# disks = system.initialize_disks()
# for _ in range(config.sequence_length):
#     plt.imshow(system.observation_model(disks)[0])
#     disks = list(map(system.process_model, disks))
#     camera.snap()
#
# print("Animate!")
# animation = camera.animate(interval=50)
# animation.save("animation.mp4")

for i in range(config.num_files):
    with fp.data.TrajectoriesFile(
        f"{output_dir}/toy_{i}.hdf5", read_only=False
    ) as traj_file:

        # Add timesteps in trajectory
        for _ in range(config.sequences_per_file):
            # Record some metadata; this is duplicated for each trajectory!
            traj_file.add_meta(dataclasses.asdict(system))

            # Initialize system and iterate
            disks = system.initialize_disks()
            for _ in range(config.sequence_length):
                image, visible_pixels_count = system.observation_model(disks)

                # Write timestep
                traj_file.add_timestep(
                    {
                        "image": image,
                        "visible_pixels_count": visible_pixels_count,
                        "position": disks[0].position,
                        "velocity": disks[0].velocity,
                    }
                )

                # Propagate disks through process model
                disks = list(map(system.process_model, disks))

            # Done with this trajectory!
            traj_file.complete_trajectory()
