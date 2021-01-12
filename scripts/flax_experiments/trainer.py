import dataclasses
from typing import Any, Dict, Optional, TypeVar

import flax
import jaxfg
import yaml
from flax.training import checkpoints


@jaxfg.utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)
class CheckpointContents:
    optimizer: flax.optim.Optimizer
    metadata_yaml: str


OptimizerType = TypeVar("OptimizerType", bound=flax.optim.Optimizer)


@dataclasses.dataclass
class Trainer:
    """Struct for bundling common things for running experiments."""

    experiment_name: str
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    verbose: bool = True

    def __post_init__(self):
        assert "_" not in self.experiment_name

    def load_checkpoint(
        self,
        optimizer_template: OptimizerType,
        step: Optional[int] = None,
    ) -> OptimizerType:
        checkpoint_template = CheckpointContents(
            optimizer=optimizer_template,
            metadata_yaml="",
        )
        checkpoint: CheckpointContents = checkpoints.restore_checkpoint(
            ckpt_dir="./checkpoints",
            target=checkpoint_template,
            step=step,
            prefix=f"{self.experiment_name}_",
        )
        self.metadata = yaml.safe_load(checkpoint.metadata_yaml)
        self._print(
            f"Loaded checkpoint: was at step {optimizer_template.state.step}, now at {checkpoint.optimizer.state.step}"
        )
        return checkpoint.optimizer

    def save_checkpoint(self, optimizer: flax.optim.Optimizer, keep: int = 5) -> None:
        contents = CheckpointContents(
            optimizer=optimizer,
            metadata_yaml=yaml.safe_dump(self.metadata),
        )
        checkpoints.save_checkpoint(
            ckpt_dir="./checkpoints",
            target=contents,
            step=optimizer.state.step,
            prefix=f"{self.experiment_name}_",
            keep=keep,
        )
        self._print(f"Saved checkpoint at step {optimizer.state.step}")

    def _print(self, *args, **kwargs):
        """Prefixed printing helper. No-op if `verbose` is set to `False`."""
        if self.verbose:
            print(f"[{type(self).__name__}]", *args, **kwargs)
