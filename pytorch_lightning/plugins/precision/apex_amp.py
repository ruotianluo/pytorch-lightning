# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, List, Tuple

import torch
from torch.optim import Optimizer
import types

from pytorch_lightning.core import LightningModule
from pytorch_lightning.plugins.precision.mixed import MixedPrecisionPlugin
from pytorch_lightning.utilities import _APEX_AVAILABLE, AMPType, rank_zero_warn

if _APEX_AVAILABLE:
    from apex import amp


def _patch_optimizer(optimizer):
    if not hasattr(optimizer, '_amp_stash'):
        # No need for patch
        return
    old_lazy_init_with_master_weights = optimizer._lazy_init_maybe_master_weights
    def new_lazy_init_with_master_weights(self):
        old_lazy_init_with_master_weights()
        self._amp_stash.lazy_init_called = True  # nasty, do this, because in load_state_dict we need it load the master param
        if hasattr(self, '_lazy_saved_state_dict'):
            self.load_state_dict(self._lazy_saved_state_dict)
            del self._lazy_saved_state_dict
    optimizer._lazy_init_maybe_master_weights = types.MethodType(new_lazy_init_with_master_weights, optimizer)

    old_state_dict = optimizer.state_dict
    def new_state_dict(self):
        state_dict = old_state_dict()
        if self._amp_stash.lazy_init_called:
            # We also need to save the master_params
            state_dict['master_params'] = [p.data.clone() for group in optimizer.param_groups for p in group['params']]
        return state_dict
    optimizer.state_dict = types.MethodType(new_state_dict, optimizer)

    old_load_state_dict = optimizer.load_state_dict
    def new_load_state_dict(self, state_dict):
        if not self._amp_stash.lazy_init_called and not hasattr(self, '_lazy_saved_state_dict'):
            """
            Note: here we assume the first time optimizer call load_state_dict is the checkpoint;
            We assume this is because there is a load_state_dict in lazy_init, and it's not trivial to remove it.
            So we add this assumption to make sure that we are resuming from real checkpoint, not the self.state_dict() from lazy_init.
            """

            # save it, load it after lazy init.
            self._lazy_saved_state_dict = state_dict
            old_load_state_dict(state_dict)
            return

        if self._amp_stash.lazy_init_called and 'master_params' in state_dict:
            # initilized already
            master_params = state_dict.pop('master_params')
            for group in optimizer.param_groups:
                for p in group['params']:
                    p.data.copy_(master_params.pop(0))
        old_load_state_dict(state_dict)
    optimizer.load_state_dict = types.MethodType(new_load_state_dict, optimizer)


class ApexMixedPrecisionPlugin(MixedPrecisionPlugin):
    """Mixed Precision Plugin based on Nvidia/Apex (https://github.com/NVIDIA/apex)"""

    def __init__(self, amp_level: str):
        self.backend = AMPType.APEX
        self.amp_level = amp_level

    def master_params(self, optimizer: torch.optim.Optimizer):
        return amp.master_params(optimizer)

    def connect(self, model: torch.nn.Module, optimizers, lr_schedulers):
        """Connects the precision plugin to the training process,
        configures apex and reinits the schedulers
        """
        if model.device.type != "cuda":
            return model, optimizers, lr_schedulers
        model, optimizers = self.configure_apex(amp, model, optimizers, self.amp_level)
        self.reinit_scheduler_properties(optimizers, lr_schedulers)
        return model, optimizers, lr_schedulers

    def backward(
        self,
        model: LightningModule,
        closure_loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        opt_idx: int,
        should_accumulate: bool,
        *args,
        **kwargs,
    ):
        """performs the actual backpropagation

        Args:
            model: the model to be optimized
            closure_loss: the loss value obtained from the closure
            optimizer: the optimizer to perform the step lateron
            opt_idx: the optimizer's index
            should_accumulate: whether to accumulate gradients or not

        """
        closure_loss = amp.scale_loss(closure_loss, model.trainer.optimizers if optimizer is None else optimizer)

        # enter apex context
        context = closure_loss
        closure_loss = closure_loss.__enter__()

        # do backward pass
        # TODO: not entirely sure, why we need this
        if model is not None and isinstance(model, LightningModule):
            model.backward(closure_loss, optimizer, opt_idx, **kwargs)

            # TODO: avoid dev_debugger and track these calls with mock
            model.trainer.dev_debugger.track_event('AMP', str(AMPType.APEX))

        else:
            closure_loss.backward(*args, **kwargs)

        # exit amp context
        a, b, c = None, None, None
        error = context.__exit__(a, b, c)
        if error:
            rank_zero_warn(a, b, c)
            raise Exception("apex unscale error")

        # once backward has been applied, release graph
        closure_loss = closure_loss.detach()
        return closure_loss

    def configure_apex(
        self,
        amp: object,
        model: LightningModule,
        optimizers: List[Optimizer],
        amp_level: str,
    ) -> Tuple[LightningModule, List[Optimizer]]:
        r"""
        Override to init AMP your own way.
        Must return a model and list of optimizers.

        Args:
            amp: pointer to amp library object.
            model: pointer to current :class:`LightningModule`.
            optimizers: list of optimizers passed in :meth:`configure_optimizers`.
            amp_level: AMP mode chosen ('O1', 'O2', etc...)

        Return:
            Apex wrapped model and optimizers

        Examples:
            .. code-block:: python

                # Default implementation used by Trainer.
                def configure_apex(self, amp, model, optimizers, amp_level):
                    model, optimizers = amp.initialize(
                        model, optimizers, opt_level=amp_level,
                    )

                    return model, optimizers
        """
        model, optimizers = amp.initialize(model, optimizers, opt_level=amp_level)
        for optimizer in optimizers:
            _patch_optimizer(optimizer)
        return model, optimizers

    @staticmethod
    def reinit_scheduler_properties(optimizers: list, schedulers: list):
        """Reinitializes schedulers with correct properties"""
        # Reinitialize optimizer.step properties added by schedulers
        for scheduler in schedulers:
            scheduler = scheduler['scheduler']
            state = None

            for optimizer in optimizers:
                # check that we dont mix users optimizers and schedulers
                if scheduler.optimizer == optimizer:
                    # Find the mro belonging to the base lr scheduler class
                    for i, mro in enumerate(scheduler.__class__.__mro__):
                        if mro in (torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            state = scheduler.state_dict()
                            scheduler.__class__.__mro__[i].__init__(scheduler, optimizer)
                            scheduler.load_state_dict(state)
                            break

                if state is not None:
                    break

    def pre_optimizer_step(
        self, pl_module: LightningModule, optimizer: Optimizer, optimizer_idx: int, lambda_closure: Callable, **kwargs
    ) -> bool:
        """
        always called before the optimizer step.
        """
        # apex amp does not support closures.
        lambda_closure()

        if not pl_module.automatic_optimization:
            pl_module.trainer.call_hook("on_after_backward")

        optimizer.step()

        return False
