import gc
import importlib
import torch
import accelerate
import math
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import time
from math import inf
import logging
from termcolor import colored


qbits_available = importlib.util.find_spec("intel_extension_for_transformers") is not None


def get_module_by_name_suffix(model, module_name: str):
    for name, module in model.named_modules():
        if name.endswith(module_name):
            return module


def simple_dispatch_model(model, device_map):
    from accelerate.hooks import add_hook_to_module, AlignDevicesHook

    if "" in device_map:
        d = device_map[""]
        model = model.to(torch.device(d))
        model.hf_device_map = device_map
        return model

    tied_params = accelerate.utils.modeling.find_tied_parameters(model)
    if set(device_map.values()) == {"cpu"} or set(device_map.values()) == {
        "cpu",
        "disk",
    }:
        main_device = "cpu"
    else:
        main_device = [d for d in device_map.values() if d not in ["cpu", "disk"]][0]

    cpu_offload_group = [(n, d) for n, d in device_map.items() if d == "cpu"]
    prev_hook = None
    for idx, (n, d) in enumerate(cpu_offload_group):
        m = get_module_by_name_suffix(model, n)
        _, prev_hook = accelerate.cpu_offload_with_hook(
            m, execution_device=main_device, prev_module_hook=prev_hook
        )
    # set first cpu offload module's prev_module_hook to the last cpu offload module's hook
    if len(cpu_offload_group) > 1:
        get_module_by_name_suffix(
            model, cpu_offload_group[0][0]
        )._hf_hook.prev_module_hook = prev_hook

    for n, d in device_map.items():
        m = get_module_by_name_suffix(model, n)
        if d != "cpu":
            d = torch.device(d)
            hook = AlignDevicesHook(d, io_same_device=True, place_submodules=True)
            add_hook_to_module(m, hook)
    accelerate.utils.modeling.retie_parameters(model, tied_params)
    model.hf_device_map = device_map

    return model


def set_module_name(model, name, value):
    if "." in name:
        parent_name = name.rsplit(".", 1)[0]
        child_name = name[len(parent_name) + 1 :]
        parent = model.get_submodule(parent_name)
    else:
        parent_name = ""
        parent = model
        child_name = name

    setattr(parent, child_name, value)


def clear_memory(weight=None):
    if weight is not None:
        del weight
    gc.collect()
    torch.cuda.empty_cache()


def compute_memory_used_pct(device):
    memory_used = torch.cuda.max_memory_allocated(device) / (1024**3)
    memory_pct = (
        memory_used
        / (torch.cuda.get_device_properties(device).total_memory / (1024**3))
        * 100
    )
    return memory_pct


def get_best_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def get_lowest_memory_device_index():
    device = None
    curr_device_memory_pct = 0
    for device_index in range(torch.cuda.device_count()):
        device_memory_pct = compute_memory_used_pct(device_index)
        if device is None or device_memory_pct < curr_device_memory_pct:
            device = device_index
            curr_device_memory_pct = device_memory_pct

    return device


@torch.no_grad()
def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True,retain_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph, retain_graph=retain_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def create_logger(output_dir='/mnt/public/zhangfengzhao/log/awq_logs', dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ':' + colored('(%(levelname)s %(message)s)', 'red')

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}_{int(time.time())}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        shape=None,
        group_size=None,
        zero_point=False,
        init_value=4.0,
        w_bit=4,
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        assert len(shape) == 2
        assert group_size > 0
        self.group_size = group_size
        self.zero_point = zero_point
        self.shape = shape
        self.init_value = init_value
        self.sigmoid = nn.Sigmoid()
        if group_size:
            dim1 = int(shape[0]*math.ceil(shape[1]/group_size))
        else:
            dim1 = shape[0]
        self.upbound_factor = nn.Parameter(torch.ones((dim1,1))*self.init_value)
        self.lowbound_factor = nn.Parameter(torch.ones((dim1,1))*self.init_value)
        self.w_bit = w_bit
        if self.zero_point:
            self.max_int = 2**self.w_bit - 1
            self.min_int = 0
        else:
            self.max_int = 2 ** (self.w_bit - 1) - 1
            self.min_int = -(2 ** (self.w_bit - 1))
        self.scales = None
        self.zeros = None
    
    def get_trainable_params(self):
        return [self.upbound_factor, self.lowbound_factor]
    
    @staticmethod
    def _round_ste(x:torch.Tensor):
            return (x.round() - x).detach() + x
    
    @torch.no_grad()
    def pseudo_dequantize_tensor(
        self, w: torch.Tensor):
        # get repeated count
        repeat_count = w.data.shape[-1] // self.scales.shape[-1]
        scales = self.scales.repeat(1, repeat_count).reshape(w.data.shape)

        # dequantize
        if self.zero_point:
            zeros = self.zeros.repeat(1, repeat_count).reshape(w.data.shape)
            w = (w.data - zeros) * scales
        else:
            w = w.data * scales
        return w

    def forward(self, w:torch.Tensor):
        assert torch.isnan(w).sum() == 0
        org_w_shape = w.shape
        assert org_w_shape[-1] % self.group_size == 0
        w = w.reshape(-1, self.group_size)
        
        reduce_shape = [-1]
        wmin = w.amin(reduce_shape, keepdim=True)
        wmax =  w.amax(reduce_shape, keepdim=True)

        wmax = self.sigmoid(self.upbound_factor)*wmax
        wmin = self.sigmoid(self.lowbound_factor)*wmin

        if self.zero_point:
            scales = (wmax - wmin).clamp(min=1e-5) / self.max_int
            zeros = (-torch.round(wmin / scales)).clamp(self.min_int, self.max_int)
            w_dequant = (
                    torch.clamp(self._round_ste(w / scales) + zeros, self.min_int, self.max_int) - zeros
                ) * scales
            zeros = zeros.view(org_w_shape[0], -1)
        else: 
            scales = wmax / self.max_int
            zeros = None
            w_dequant = torch.clamp(self._round_ste(w / scales), self.min_int, self.max_int) * scales
        
        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w_dequant).sum() == 0

        scales = scales.view(org_w_shape[0], -1)
        w_dequant = w_dequant.reshape(org_w_shape)
        self.scales = scales
        self.zeros = zeros
        return w_dequant
    

class FakeLinear(nn.Module):
    def __init__(self, 
        ori_layer:nn.Linear,
        group_size=None,
        zero_point=False,
        init_value=4.0,
        w_bit=4,
        use_tmp_weight=False,
        quant_module=UniformAffineQuantizer):
        super(FakeLinear, self).__init__()
        self.ori_layer = ori_layer
        self.use_tmp_weight = use_tmp_weight
        shape = self.ori_layer.weight.shape
        self.quant_module = quant_module(shape=shape, group_size=group_size, 
                                         zero_point=zero_point, init_value=init_value, w_bit=w_bit)
        self.register_buffer('tmp_weight', self.ori_layer.weight.data.clone())
    
    def forward(self, x):
        if self.use_tmp_weight:
            weight = self.quant_module(self.tmp_weight)
            self.tmp_weight = weight.data.clone()
        else:
            weight = self.quant_module(self.ori_layer.weight)
        return F.linear(x, weight, self.ori_layer.bias)