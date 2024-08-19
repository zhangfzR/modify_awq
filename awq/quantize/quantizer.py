import torch
import inspect
import logging
import functools
import torch.nn as nn
from torch import float16, Tensor
from typing import Union
import numpy as np
import math
import pdb
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, List, Optional
from collections import defaultdict
from awq.utils.calib_data import get_calib_dataset
from awq.quantize.scale import apply_scale, apply_clip, apply_lwc
from awq.utils.utils import (clear_memory, get_best_device, NativeScalerWithGradNormCount, 
                             create_logger, FakeLinear)
from awq.modules.linear import (
    WQLinear_GEMM,
    WQLinear_GEMV,
    WQLinear_Marlin,
    WQLinear_GEMVFast,
)
from awq.utils.module import (
    append_str_prefix,
    get_op_name,
    get_named_linears,
    set_op_by_name,
    exclude_layers_to_not_quantize,
    set_op_by_name_module_fakelinear,
)


class AwqQuantizer:
    def __init__(
        self,
        awq_model,
        model,
        tokenizer,
        w_bit,
        group_size,
        zero_point,
        version,
        calib_data,
        split,
        text_column,
        duo_scaling,
        modules_to_not_convert=None,
        export_compatible=False,
        apply_clip=True,
        n_parallel_calib_samples=None,
        max_calib_samples=128,
        max_calib_seq_len=512,
        max_chunk_memory=1024 * 1024 * 1024,
        n_grid = 20,
        n_grid_clip = 20,
        disable_hqq=True,
        epochs = 20,
        lwc_lr = 1e-2,
        disable_lwc=True,
        use_adaround=False,
    ) -> None:
        self.awq_model = awq_model
        self.model = model
        self.tokenizer = tokenizer
        self.w_bit = w_bit
        self.group_size = group_size
        self.zero_point = zero_point
        self.version = version
        self.calib_data = calib_data
        self.split = split
        self.text_column = text_column
        self.duo_scaling = duo_scaling
        self.export_compatible = export_compatible
        self.apply_clip = apply_clip
        self.n_parallel_calib_samples = n_parallel_calib_samples
        self.max_calib_samples = max_calib_samples
        self.max_calib_seq_len = max_calib_seq_len
        self.max_chunk_memory = max_chunk_memory
        self.modules_to_not_convert = (
            modules_to_not_convert if modules_to_not_convert is not None else []
        )
        self.modules, self.module_kwargs, self.inps = self.init_quant(
            n_samples=self.max_calib_samples, max_seq_len=self.max_calib_seq_len
        )
        self.n_grid = n_grid
        self.n_grid_clip = n_grid_clip
        self.disable_hqq = disable_hqq
        self.init_value = 4.    # inti value of learnable weight clipping
        self.epochs = epochs
        self.lec_lr = lwc_lr
        
        self.logger = create_logger()
        self.disable_lwc = disable_lwc
        self.use_adaround = use_adaround
        if not self.disable_hqq:
            self.logger.info("Using HQQ optimization")

    def pseudo_quantize_tensor(self, w: torch.Tensor):
        org_w_shape = w.shape
        if self.group_size > 0:
            assert org_w_shape[-1] % self.group_size == 0
            w = w.reshape(-1, self.group_size)
        assert w.dim() == 2
        assert torch.isnan(w).sum() == 0

        # zero point quantization
        if self.zero_point:
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            max_int = 2**self.w_bit - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
            if not self.disable_hqq:
                w, scales, zeros = self.optimize_weights_proximal_legacy(
                    w, scales, zeros, [min_int, max_int], axis=1, device=w.device, verbose=False)
                zeros = zeros.view(org_w_shape[0], -1)
            else:
                w = (
                    torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
                ) * scales
                zeros = zeros.view(org_w_shape[0], -1)
        else:
            max_val = w.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (self.w_bit - 1) - 1
            min_int = -(2 ** (self.w_bit - 1))
            scales = max_val / max_int
            zeros = None
            w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        scales = scales.view(org_w_shape[0], -1)
        w = w.reshape(org_w_shape)

        return w, scales, zeros
    
    def shrink_lp_op(self, x: Tensor, beta: float, lp_norm: float) -> Tensor:
        if lp_norm == 1:
            return torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - 1.0 / beta)
        else:
            return torch.sign(x) * torch.nn.functional.relu(
                torch.abs(x) - (1.0 / beta) * torch.pow(torch.abs(x), lp_norm - 1)
            )
    
    @torch.inference_mode()
    def optimize_weights_proximal_legacy(
        self,
        tensor: Tensor,
        scale: Tensor,
        zero: Tensor,
        min_max: list,
        axis: int = 0,
        device: Union[str, None] = None,
        opt_params: dict = {"lp_norm": 0.7, "beta": 1e1, "kappa": 1.01, "iters": 20},
        verbose: bool = False,
    ) -> tuple:
        lp_norm, beta, kappa, iters = (
            opt_params["lp_norm"],
            opt_params["beta"],
            opt_params["kappa"],
            opt_params["iters"],
        )

        if device is None:
            device = tensor.device
        else:
            device = torch.device(device)

        dtype = float16
        W_f = tensor.to(dtype=dtype, device=device)
        scale = scale.to(dtype=dtype, device=device)
        zero = zero.to(dtype=dtype, device=device)

        best_error = 1e4
        for i in range(iters):
            W_q = torch.round(W_f / scale + zero).clamp(min_max[0], min_max[1])
            W_r = (W_q - zero) * scale
            W_e = self.shrink_lp_op(W_f - W_r, beta, lp_norm)
            zero = torch.mean(W_q - (W_f - W_e) / scale, axis=axis, keepdim=True)
            beta *= kappa

            current_error = float(torch.abs(W_f - W_r).mean())
            if verbose:
                print(i, np.round(current_error, 6))
            if current_error < best_error:
                best_error = current_error
            else:
                break

        scale = scale.to(tensor.device)
        zero = zero.to(tensor.device)
        del W_f, W_q, W_r, W_e
        torch.cuda.empty_cache()

        W = (torch.round(tensor / scale + zero).clamp(min_max[0], min_max[1]) - zero) * scale
        zero = torch.clamp(torch.round(zero), min_max[0], min_max[1])
        return W, scale, zero

    def pseudo_dequantize_tensor(
        self, w: nn.Linear, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None
    ):
        # get repeated count
        repeat_count = w.weight.data.shape[-1] // scales.shape[-1]
        scales = scales.repeat(1, repeat_count).reshape(w.weight.data.shape)

        # dequantize
        if self.zero_point:
            zeros = zeros.repeat(1, repeat_count).reshape(w.weight.data.shape)
            w = (w.weight.data - zeros) * scales
        else:
            w = w.weight.data * scales

        return w


    def learnable_weight_clipping(self, i, idx, prev_op, layers, inp:Tensor, module2inspect=None, kwargs={}):
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]
        loss_scaler = NativeScalerWithGradNormCount()
        module2inspect.to(get_best_device())
        # module2inspect_copy = copy.deepcopy(module2inspect)

        inp = inp.to(next(module2inspect.parameters()).device)
        device = inp.device
        
        if "use_cache" in kwargs:
            kwargs.pop("use_cache")
        
        with torch.no_grad():
            module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)
            module_kwargs = {name: value.to(device) for name, value in module_kwargs.items() if isinstance(value, Tensor)}
            fp16_output = self._module_forward(inp, module2inspect, module_kwargs)
        param_list = []
        fake_linear_list_tmp = {}

        with torch.cuda.amp.autocast():
            for fc in layers:
                fake_linear = FakeLinear(ori_layer=fc, group_size=self.group_size, 
                                         zero_point=self.zero_point, init_value=self.init_value, 
                                         w_bit=self.w_bit).to(device)
                param_list += fake_linear.quant_module.get_trainable_params()
                fake_linear_list_tmp[id(fc)] = fake_linear
        if isinstance(module2inspect, nn.Linear):
            module2inspect = fake_linear
        else:
            set_op_by_name_module_fakelinear(module2inspect, fake_linear_list_tmp)
        optimizer = torch.optim.AdamW(param_list, lr=self.lec_lr, weight_decay=0.0)
        del fake_linear_list_tmp

        for epochs in tqdm(range(self.epochs)):
            loss_list = []
            norm_list = []
            with torch.cuda.amp.autocast():
                int_w_output = self._module_forward(inp, module2inspect, module_kwargs)
                loss = F.mse_loss(int_w_output, fp16_output)
            if not math.isfinite(loss):
                self.logger.info("Loss is NAN, stopping training")
                pdb.set_trace()
            
            loss_list.append(loss.detach().cpu())
            optimizer.zero_grad()
            norm = loss_scaler(loss, optimizer,parameters=param_list).cpu()
            norm_list.append(norm.data)

            loss_mean = torch.stack(loss_list).mean()
            norm_mean = torch.stack(norm_list).mean()
            self.logger.info(f"layer {i} module {idx} iter {epochs} loss:{loss_mean} norm:{norm_mean} \
                             max memory_allocated {torch.cuda.max_memory_allocated(device) / 1024**2} ")
        apply_lwc(module2inspect)
        clear_memory()


    def quantize(self):
        for i in tqdm(range(len(self.modules)), desc="AWQ"):
            with torch.no_grad():
                # Move module and inputs to correct device
                common_device = next(self.modules[i].parameters()).device
                if common_device is None or str(common_device) == "cpu":
                    if torch.cuda.is_available():
                        best_device = "cuda:" + str(i % torch.cuda.device_count())
                    else:
                        best_device = get_best_device()

                    self.modules[i] = self.modules[i].to(best_device)
                    common_device = next(self.modules[i].parameters()).device

                if self.module_kwargs.get("position_ids") is not None:
                    self.module_kwargs["position_ids"] = self.module_kwargs[
                        "position_ids"
                    ].to(common_device)

                if self.module_kwargs.get("attention_mask") is not None:
                    self.module_kwargs["attention_mask"] = self.module_kwargs[
                        "attention_mask"
                    ].to(common_device)

                self.inps = self.inps.to(common_device)

                # [STEP 1]: Get layer, extract linear modules, extract input features
                named_linears = get_named_linears(self.modules[i])

                # Filter out the linear layers we don't want to exclude
                named_linears = exclude_layers_to_not_quantize(
                    named_linears, self.modules_to_not_convert
                )

                input_feat = self._get_input_feat(self.modules[i], named_linears)
                clear_memory()

                # [STEP 2]: Compute and apply scale list
                module_config: List[Dict] = self.awq_model.get_layers_for_scaling(
                    self.modules[i], input_feat, self.module_kwargs
                )
                scales_list = [
                    self._search_best_scale(self.modules[i], **layer)
                    for layer in tqdm(module_config, desc="Best Scales", leave=False)
                ]
                apply_scale(self.modules[i], scales_list, input_feat_dict=input_feat)
                scales_list = append_str_prefix(
                    scales_list, get_op_name(self.model, self.modules[i]) + "."
                )

            # [STEP 3]: Compute and apply clipping list
                self.logger.info("Starting AWQ Clipping optimization")
                if self.apply_clip:
                    clip_list = self._search_best_clip(
                        self.modules[i], named_linears, input_feat
                    )
                    apply_clip(self.modules[i], clip_list)
                    clip_list = append_str_prefix(
                        clip_list, get_op_name(self.model, self.modules[i]) + "."
                    )
            if not self.disable_lwc:       
                self.logger.info("Starting LWC optimization")
                for idx, layer in enumerate(tqdm(module_config, desc="LWC", leave=False)):
                    self.learnable_weight_clipping(i, idx, **layer)

            # # [STEP 4]: Quantize weights2
            if not self.export_compatible:
                self._apply_quant(self.modules[i], named_linears)

            clear_memory()

    def pack(self):
        for i in tqdm(range(len(self.modules)), desc="Packing"):
            named_linears = get_named_linears(self.modules[i])
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert
            )
            self._apply_quant(self.modules[i], named_linears)
            clear_memory()

    
    @torch.no_grad()
    def _apply_quant(self, module, named_linears: Dict[str, nn.Linear]):
        for name, linear_layer in named_linears.items():
            # NOTE: small regression in perplexity if linear layer uses .cpu().float()
            linear_layer = linear_layer.to(get_best_device()).half()

            if self.disable_lwc:
                linear_layer.weight.data, scales, zeros = self.pseudo_quantize_tensor(
                    linear_layer.weight.data
                )
            else:
                scales, zeros = linear_layer.scales, linear_layer.zeros

            if self.version == "gemm":
                scales = scales.t().contiguous()
                if zeros is not None:
                    zeros = zeros.t().contiguous()
                q_linear_module = WQLinear_GEMM

            elif self.version == "gemv":
                q_linear_module = WQLinear_GEMV

            elif self.version == "marlin":
                q_linear_module = WQLinear_Marlin

            elif self.version == "gemv_fast":
                q_linear_module = WQLinear_GEMVFast

            else:
                raise ValueError(f"Unknown version {self.version}")

            q_linear = q_linear_module.from_linear(
                linear=linear_layer,
                w_bit=self.w_bit,
                group_size=self.group_size,
                init_only=False,
                scales=scales,
                zeros=zeros,
                use_adaround=self.use_adaround,
            )

            linear_layer.cpu()
            q_linear.to(next(module.parameters()).device)
            set_op_by_name(module, name, q_linear)
            clear_memory()

    # @torch.no_grad()
    def _module_forward(
        self, x: torch.Tensor, module: torch.nn.Module, module_kwargs: Dict
    ) -> torch.Tensor:
        if self.n_parallel_calib_samples is None:
            # runs through all samples at once
            module_output = module(x, **module_kwargs)
            if isinstance(module_output, tuple):
                module_output = module_output[0]
        else:
            # memory efficiently runs through all calibration samples
            # but only n_parallel_calib_samples at a time
            module_output = []
            partitioned_inputs = torch.split(x, self.n_parallel_calib_samples)
            for x_partial in tqdm(
                partitioned_inputs, desc="Module forward", leave=False
            ):
                partial_output = module(x_partial, **module_kwargs)

                if isinstance(partial_output, tuple):
                    partial_output = partial_output[0]

                module_output.append(partial_output.cpu())

            module_output = torch.cat(module_output, dim=0)

        return module_output

    @torch.no_grad()
    def _search_best_scale(
        self,
        module,
        prev_op,
        layers: List[nn.Linear],
        inp: torch.Tensor,
        module2inspect=None,
        kwargs={},
    ):
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        if "use_cache" in kwargs:
            kwargs.pop("use_cache")

        # Put x on the right device
        inp = inp.to(next(module2inspect.parameters()).device)

        # [STEP 1]: Compute per-channel mean of normalised weights
        # All layer weights are concatted together
        weight = torch.cat([_m.weight for _m in layers], dim=0)
        org_shape = weight.shape
        # The weights are reshaped to be organised by quantization group
        weight = weight.view(-1, self.group_size)
        # Calculates the relative magnitude of the weights within each of the quantization groups,
        # and rescales each group individually so that each group has weights on a 0-1 scale.
        w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)
        # Resizes the rescaled weight matrix back up to its original dimensions
        w_scale = w_scale.view(org_shape)
        # Gets the average rescaled magnitude for each output channel
        w_mean = w_scale.mean(0)
        clear_memory(weight)

        # [STEP 2]: Compute per-channel mean of the input activation with chunking
        # move inp to cpu to avoid memory leak
        inp_flat = inp.cpu().abs().view(-1, inp.shape[-1])
        num_elements = inp_flat.size(0)
        num_channels = inp_flat.size(1)
        element_size_bytes = inp_flat.element_size() * 2 # multiplied by 2 for FP32

        # Calculate chunk size dynamically based on max_chunk_memory
        chunk_size = int(self.max_chunk_memory // (element_size_bytes * num_channels))
        chunk_size = min(chunk_size, num_elements)

        # Use float32 for sum calculation
        x_sum = torch.zeros(num_channels, dtype=torch.float32, device=inp.device)
        
        for i in range(0, num_elements, chunk_size):
            end = min(i + chunk_size, num_elements)
            chunk_sum = inp_flat[i:end].to(torch.float32).sum(dim=0)
            x_sum += chunk_sum.to(inp.device)

        x_mean = (x_sum / num_elements).to(inp.dtype)
        clear_memory(x_sum)


        # [STEP 3]: Compute output of module
        with torch.no_grad():
            module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)
            fp16_output = self._module_forward(inp, module2inspect, module_kwargs)

        # [STEP 4]: Compute loss
        best_scales = self._compute_best_scale(
            inp, w_mean, x_mean, module2inspect, layers, fp16_output, module_kwargs, n_grid=self.n_grid
        )

        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            best_scales,
        )

    def _compute_best_scale(
        self,
        x: torch.Tensor,
        w_mean: torch.Tensor,
        x_mean: torch.Tensor,
        module2inspect: torch.nn.Module,
        linears2scale: List[nn.Linear],
        fp16_output: torch.Tensor,
        kwargs: Dict={},
        n_grid = 20,
    ):
        """
        Compute loss and select best scales

        L(s) = || Q(W * s) (s^-1 * X) - W * X ||
        Q: weight quantization function | pseudo_quantize_tensor(W * s)
        X: inputs from calib dataset    | X
        W: original weights in FP16     | layer
        s: per channel scaling factor   | s^-1 * X
        """
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")

        org_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}

        device = x.device
        x_mean = x_mean.view(-1).to(device)
        w_mean = w_mean.view(-1).to(device)

        with tqdm(range(n_grid), desc="Grid Search", leave=False) as pbar:
            for ratio in pbar:
                # create new scales
                ratio = ratio / n_grid

                # NOTE: s^-1 * x is fused here, according to paper
                if self.duo_scaling:
                    scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
                else:
                    scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
                scales = scales / (scales.max() * scales.min()).sqrt()
                scales_view = scales.view(1, -1).to(device)

                # avoid scaling values that overflow
                scales[torch.isinf(scales)] = 1
                scales[torch.isnan(scales)] = 1

                # Q(W * s)
                for fc in linears2scale:
                    fc.weight.mul_(scales_view)
                    fc.weight.data = (
                        self.pseudo_quantize_tensor(fc.weight.data)[0] / scales_view
                    )

                # W * X
                int_w_output = self._module_forward(x, module2inspect, kwargs)

                # compute mean squared error (L2 norm)
                loss = self._compute_loss(fp16_output, int_w_output, device)

                history.append(loss)
                if loss < best_error:
                    best_error = loss
                    best_ratio = ratio
                    best_scales = scales.clone()
                module2inspect.load_state_dict(org_sd)
                pbar.set_description(f"Grid Search (Best: {best_ratio})")

        if best_ratio == -1:
            logging.debug(history)
            raise Exception

        assert torch.isnan(best_scales).sum() == 0, best_scales

        return best_scales.detach().cpu()

    @torch.no_grad()
    def _compute_loss(
        self,
        fp16_output: torch.Tensor,
        int_w_output: torch.Tensor,
        device: torch.device,
    ):
        loss = 0.0
        fp16_output_flat = fp16_output.view(-1)
        int_w_output_flat = int_w_output.view(-1)
        num_elements = fp16_output_flat.size(0)
        element_size_bytes = fp16_output.element_size()

        # Calculate chunk size dynamically based on max_chunk_memory
        # Divide the max_chunk_memory by twice the element size
        chunk_size = self.max_chunk_memory // (element_size_bytes * 2)
        chunk_size = min(chunk_size, num_elements)

        # Split the computation into chunks
        fp16_chunks = torch.split(fp16_output_flat, chunk_size)
        int_w_chunks = torch.split(int_w_output_flat, chunk_size)

        # Compute the loss for each chunk
        with tqdm(
            zip(fp16_chunks, int_w_chunks),
            total=len(fp16_chunks),
            desc="Computing Loss",
            leave=False,
        ) as pbar:
            for fp16_chunk, int_w_chunk in pbar:
                chunk_loss = (fp16_chunk.to(device) - int_w_chunk.to(device)).float().pow(2).sum().item()
                loss += chunk_loss
                pbar.set_description(f"Computing Loss (loss: {loss:.2f})")

        # Normalize the loss by the total number of elements
        loss /= num_elements

        return loss

    @torch.no_grad()
    def _search_best_clip(self, layer, named_linears, input_feat):
        clip_list = []
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

        for name in tqdm(named_linears, desc="Computing Best Clip", leave=False):
            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in avoid_clipping]):
                continue

            named_linears[name].to(get_best_device())
            max_val = self._compute_best_clip(
                named_linears[name].weight, input_feat[name],n_grid=self.n_grid_clip,
            )
            clip_list.append((name, max_val))
            named_linears[name].cpu()

        return clip_list

    @torch.no_grad()
    def _compute_best_clip(
        self,
        w: torch.Tensor,
        input_feat: torch.Tensor,
        n_grid=20,
        max_shrink=0.5,
        n_sample_token=512,
    ):
        assert w.dim() == 2
        org_w_shape = w.shape
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        group_size = self.group_size if self.group_size > 0 else org_w_shape[1]
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)

        # Compute input feature step size (minimum 1)
        step_size = max(1, input_feat.shape[1] // n_sample_token)
        input_feat = input_feat[:, ::step_size]
        
        w = w.reshape(org_w_shape[0], 1, -1, group_size)

        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64  # prevent OOM
        assert org_w_shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []

        for i_b in range(org_w_shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = -max_val
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = self.pseudo_quantize_tensor(cur_w)[0]
                cur_out = (input_feat * q_w).sum(dim=-1)

                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)

        clear_memory(input_feat)
        clear_memory(org_out)

        return best_max_val.squeeze(1)

    def init_quant(self, n_samples=128, max_seq_len=512):
        modules = self.awq_model.get_model_layers(self.model)
        samples = get_calib_dataset(
            data=self.calib_data,
            tokenizer=self.tokenizer,
            n_samples=n_samples,
            max_seq_len=max_seq_len,
            split=self.split,
            text_column=self.text_column,
        )
        samples = torch.cat(samples, dim=0)

        inps = []
        layer_kwargs = {}

        best_device = get_best_device()
        modules[0] = modules[0].to(best_device)
        self.awq_model.move_embed(self.model, best_device)

        # get input and kwargs to layer 0
        # with_kwargs is only supported in PyTorch 2.0
        # use this Catcher hack for now
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                # assume first input to forward is hidden states
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)

                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference

        # patch layer 0 to catch input and kwargs
        modules[0] = Catcher(modules[0])
        try:
            self.model(samples.to(next(self.model.parameters()).device))
        except ValueError:  # work with early exit
            pass
        modules[0] = modules[0].module  # restore

        # Update the layer kwargs with `prepare_inputs_for_generation` method
        # that takes care of everything to avoid unexpected errors.
        layer_kwargs = self.model.prepare_inputs_for_generation(samples, **layer_kwargs)
        # Pop the input_ids as they are not needed at all.
        layer_kwargs.pop("input_ids")

        del samples
        inps = inps[0]

        modules[0] = modules[0].cpu()
        self.awq_model.move_embed(self.model, "cpu")

        clear_memory()

        if layer_kwargs.get("attention_mask") is not None:
            layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to(
                best_device
            )

        return modules, layer_kwargs, inps

    def _get_input_feat(self, layer, named_linears):
        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []

        # FIXME: Workaround for Mixtral to use block_sparse_moe input features
        if self.awq_model.model_type == "mixtral":
            named_linears = {
                **named_linears,
                "block_sparse_moe": layer.block_sparse_moe,
            }

        if self.awq_model.model_type == "deepseek_v2":
            named_linears = {
                **named_linears,
                "mlp": layer.mlp,
            }

        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        self.inps = self.inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input

        # Sanitize the kwargs in case we use transformers version that contains
        # kwargs that are not handled by the module.
        # Useful for trust_remote_code models.
        module_kwargs = self._sanitize_kwargs(self.module_kwargs, layer)

        self.inps = self._module_forward(self.inps, layer, module_kwargs)
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        return input_feat

    def _sanitize_kwargs(self, inputs_kwargs, module):
        """
        Remove the arguments that are not supported in the module's
        forward pass to avoid breaking behaviour between different versions
        of transformers.

        Args:
            inputs_kwargs (`dict`):
                The input dictionary to pass to the model layer
            module (`torch.nn.Module`):
                Target module to quantize.
        """
        module_signature = inspect.signature(module.forward).parameters
        sanitized_kwargs = {}
        for k, v in inputs_kwargs.items():
            if k in module_signature:
                sanitized_kwargs[k] = v
        return sanitized_kwargs
