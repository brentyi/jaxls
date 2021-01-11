import dataclasses
import pathlib
from typing import Any, Dict, Optional, cast

import flax
import yaml
from flax.training import checkpoints

XXX = 5

import jaxfg


@jaxfg.utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)
class CheckpointContents:
    step: int
    optimizer: flax.optim.Optimizer
    metadata_yaml: str


flax.optim.Adam


@dataclasses.dataclass
class Trainer:
    experiment_name: str
    optimizer: flax.optim.Optimizer
    step: int = 0
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        assert "_" not in self.experiment_name

    def __repr__(self) -> str:
        x = dataclasses.asdict(self)
        x["optimizer"] = repr(self.optimizer.optimizer_def.hyper_params)
        return f"Trainer{repr(x)}"

    def load_checkpoint(
        self,
        step: Optional[int] = None,
    ) -> None:
        checkpoint_template = CheckpointContents(
            step=0,
            optimizer=self.optimizer,
            metadata_yaml="",
        )
        checkpoint: CheckpointContents = checkpoints.restore_checkpoint(
            ckpt_dir="./checkpoints",
            target=checkpoint_template,
            step=step,
            prefix=f"{self.experiment_name}_",
        )
        self.step = checkpoint.step
        self.optimizer = checkpoint.optimizer
        self.metadata = yaml.safe_load(checkpoint.metadata_yaml)

    def save_checkpoint(self, keep: int = 5) -> None:
        contents = CheckpointContents(
            step=self.step,
            optimizer=self.optimizer,
            metadata_yaml=yaml.safe_dump(self.metadata),
        )
        checkpoints.save_checkpoint(
            ckpt_dir="./checkpoints",
            target=contents,
            step=self.step,
            prefix=f"{self.experiment_name}_",
            keep=keep,
        )

    def apply_gradient(self, grad: jaxfg.types.PyTree, **hyper_param_overrides: Any):
        self.optimizer = self.optimizer.apply_gradient(grad, **hyper_param_overrides)
        self.step += 1
