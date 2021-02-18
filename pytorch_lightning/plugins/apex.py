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
import math
from typing import List, Tuple, Union
import types

import torch
from torch.optim.optimizer import Optimizer

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.plugins.precision_plugin import PrecisionPlugin
from pytorch_lightning.utilities import APEX_AVAILABLE, AMPType
from pytorch_lightning.utilities.distributed import rank_zero_warn

if APEX_AVAILABLE:
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


class ApexPlugin(PrecisionPlugin):

    def __init__(self, trainer=None):
        self.trainer = trainer

    def connect(self, model, optimizers):
        model, optimizers = self.configure_apex(amp, model, optimizers, self.trainer.amp_level)
        self.trainer.reinit_scheduler_properties(optimizers, self.trainer.lr_schedulers)
        return model, optimizers

    def training_step(self, fx, args):
        output = fx(args)
        return output

    def backward(self, closure_loss, optimizer, opt_idx, *args, **kwargs):
        closure_loss = amp.scale_loss(closure_loss, optimizer)

        # enter apex context
        self.trainer.dev_debugger.track_event('AMP', str(AMPType.APEX))
        context = closure_loss
        closure_loss = closure_loss.__enter__()

        # do backward pass
        if self.trainer.train_loop.automatic_optimization:
            model = self.trainer.get_model()
            model.backward(closure_loss, optimizer, opt_idx)
        else:
            closure_loss.backward(*args, **kwargs)

        # exit amp context
        a, b, c = None, None, None
        error = context.__exit__(a, b, c)
        if error:
            rank_zero_warn(a, b, c)
            raise Exception('apex unscale error')

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

    def clip_gradients(self, grad_clip_val: Union[int, float], optimizer: Optimizer, norm_type: float):
        """
        This code is a modification of :meth:`torch.nn.utils.clip_grad_norm_` using a higher epsilon for fp16 weights.
        This is important when setting amp_level to O2, and the master weights are in fp16.
        Args:
            grad_clip_val: Maximum norm of gradients.
            optimizer: Optimizer with gradients that will be clipped.
            norm_type: (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        """
        model = self.trainer.get_model()
        parameters = model.parameters()
        max_norm = float(grad_clip_val)

        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]

        if len(parameters) == 0:
            return torch.tensor(0.)
        device = parameters[0].grad.device
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
        clip_coef = max_norm / (total_norm + self.norm_clipping_epsilon)
        if clip_coef < 1:
            for p in parameters:
                p.grad.detach().mul_(clip_coef.to(p.grad.device))

    @property
    def norm_clipping_epsilon(self):
        return 1e-5

    def optimizer_step(self, trainer, optimizer, closure):
        # apex amp does not yet support closures.
        # TODO: pass the closure to the step ASAP
        with trainer.profiler.profile("closure"):
            closure()

        if not self.trainer.train_loop.automatic_optimization:
            trainer.call_hook("on_after_backward")

        with trainer.profiler.profile("optimizer_step"):
            optimizer.step()
