import torch
import torch.nn as nn

from .nn.Conv2d_uint8 import uint8_qparams


def calibrate_int8(model, data_loader, num_pictures, save_path, weight_bits=8):
    """
    对 CNN 进行 min-max (absmax) 校准,生成 INT8 量化 scale。
    - 跳过第一个 Conv2d 层
    - 不做层融合
    - 激活: per-tensor absmax
    - 权重: per-channel absmax (沿输出通道 O)
    - scale_x: 激活 scale, scale_w: 权重 scale (shape = (O,))
    """
    if not 3 <= weight_bits <= 8:
        raise ValueError(f"weight_bits must be between 3 and 8, got {weight_bits}")
    model.eval()
    device = next(model.parameters()).device

    # ---------- 1. 找出需要校准的层 ----------
    target_layers = {}   # name -> module
    first_conv_skipped = False
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if not first_conv_skipped:
                first_conv_skipped = True
                print(f"[Skip] 第一个 Conv2d: {name}")
                continue
            target_layers[name] = module
        # elif isinstance(module, nn.Linear):
        #     target_layers[name] = module

    # ---------- 2. 注册 hook 收集每层输入 absmax ----------
    absmax_record = {name: 0.0 for name in target_layers}
    hooks = []

    def make_hook(layer_name):
        def hook(module, inputs, output):
            x = inputs[0]
            cur = x.detach().abs().max().item()
            if cur > absmax_record[layer_name]:
                absmax_record[layer_name] = cur
        return hook

    qmax_w = 2 ** (weight_bits - 1) - 1
    for name, module in target_layers.items():
        hooks.append(module.register_forward_hook(make_hook(name)))

    # ---------- 3. 跑校准数据 ----------
    seen = 0
    with torch.no_grad():
        for batch in data_loader:
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            imgs = imgs.to(device)
            model(imgs)
            seen += imgs.size(0)
            if seen >= num_pictures:
                break

    for h in hooks:
        h.remove()

    # ---------- 4. 计算激活 scale_x (per-tensor) ----------
    scale_x_dict = {}
    for name, absmax in absmax_record.items():
        scale = absmax / 127.0 if absmax > 0 else 1.0
        scale_x_dict[name] = torch.tensor(scale)
        print(f"[scale_x] {name}: absmax={absmax:.6f}, scale={scale:.8f}")

    # ---------- 5. 计算权重 scale_w (per-channel, 沿 O) ----------
    scale_w_dict = {}
    for name, module in target_layers.items():
        w = module.weight.detach()
        # Conv2d: (O, I, kH, kW) -> reduce over (1,2,3)
        # Linear: (O, I)         -> reduce over (1,)
        reduce_dims = tuple(range(1, w.dim()))
        absmax_per_oc = w.abs().amax(dim=reduce_dims)        # shape: (O,)
        scale_w = absmax_per_oc / qmax_w
        scale_w = torch.where(scale_w > 0, scale_w, torch.ones_like(scale_w))
        scale_w_dict[name] = scale_w
        print(f"[scale_w] {name}: shape={tuple(scale_w.shape)}, "
              f"max={scale_w.max().item():.8f}, min={scale_w.min().item():.8f}")

    # ---------- 6. 构建新的 state_dict ----------
    # 依次保存所有原参数,并在 target 层旁边插入 scale_x / scale_w
    new_state_dict = {}
    orig_state = model.state_dict()

    # 先建一个 "层名 -> 该层所有参数 key" 的索引,方便插入 scale
    # 对每个 target 层,我们在它的参数后面紧跟着插入 scale_x / scale_w
    target_param_prefixes = set(target_layers.keys())
    inserted = set()

    for k, v in orig_state.items():
        new_state_dict[k] = v
        # 判断这个 key 属于哪个 target 层 (例如 "features.3.weight" -> "features.3")
        layer_name = k.rsplit(".", 1)[0]
        param_name = k.rsplit(".", 1)[-1]
        if layer_name in target_param_prefixes and layer_name not in inserted:
            # 在该层第一个参数(通常是 weight)之后插入 scale
            if param_name == "weight":
                # 注意:scale 也要紧跟 weight 之后,但 bias 还没写入
                # 这里先不插,等 bias 也写完(或确认没 bias)再插
                pass

    # 重新来一遍,采用更稳的策略:遍历 orig_state,记录每个 target 层最后一个参数的位置,在其后插入
    new_state_dict = {}
    # 找到每个 target 层的"最后一个参数 key"
    last_key_of_layer = {}
    for k in orig_state.keys():
        layer_name = k.rsplit(".", 1)[0]
        if layer_name in target_param_prefixes:
            last_key_of_layer[layer_name] = k  # 不断覆盖,最终是最后一个

    for k, v in orig_state.items():
        new_state_dict[k] = v
        layer_name = k.rsplit(".", 1)[0]
        if layer_name in target_param_prefixes and k == last_key_of_layer[layer_name]:
            new_state_dict[f"{layer_name}.scale_x"] = scale_x_dict[layer_name]
            new_state_dict[f"{layer_name}.scale_w"] = scale_w_dict[layer_name]

    # ---------- 7. 保存 --------------
    torch.save(new_state_dict, save_path)
    print(f"[Save] 已保存到 {save_path}")

    return new_state_dict


def calibrate_uint8(
        model,
        data_loader,
        num_pictures,
        save_path=None,
        ignore_first_conv=True,
):
    """校准 CNN 的 uint8 非对称量化参数并返回增强后的 ``state_dict``。

    量化粒度与 :class:`approxtorch.nn.Conv2d_uint8` 一致：

    - 激活：per-tensor min/max；
    - 权重：沿输出通道的 per-channel min/max；
    - 量化范围固定为 ``[0, 255]``；
    - min/max 范围强制包含真实 0，使 0 能被 zero point 精确表示；
    - zero point 数值严格为整数，但以 ``torch.float32`` 保存，以兼容
      ``Conv2d_uint8`` 的运算和 CUDA fake-quant 融合接口。

    每个目标卷积会写入两组字段：``x_min/x_max/w_min/w_max`` 可直接加载到
    当前 ``Conv2d_uint8`` 的统计 buffer；``scale_x/zero_x/scale_w/zero_w``
    是显式量化参数，便于模型转换、导出或检查。二者由同一组有效范围导出，
    不会出现 checkpoint 中的统计量和量化参数不一致。

    Args:
        model: 待校准的浮点模型。
        data_loader: 校准数据迭代器；batch 可以是输入 tensor，或首元素为
            输入 tensor 的 list/tuple（例如 ``(images, labels)``）。
        num_pictures: 最多使用的样本数；``None`` 表示遍历完整个迭代器。
        save_path: 非 ``None`` 时保存增强后的 ``state_dict``。
        ignore_first_conv: 是否跳过网络中的第一个 ``nn.Conv2d``，默认与
            ``convert_model`` 的项目约定一致。

    Returns:
        包含原模型参数和 uint8 校准参数的 ``state_dict``。
    """
    if num_pictures is not None:
        if isinstance(num_pictures, bool) or not isinstance(num_pictures, int):
            raise TypeError("num_pictures must be a positive int or None")
        if num_pictures <= 0:
            raise ValueError("num_pictures must be positive")

    # 参数模型取第一个参数的 device；无参数模型再尝试 buffer，最后退回 CPU。
    first_parameter = next(model.parameters(), None)
    first_buffer = next(model.buffers(), None)
    if first_parameter is not None:
        device = first_parameter.device
    elif first_buffer is not None:
        device = first_buffer.device
    else:
        device = torch.device("cpu")

    target_layers = {}
    first_conv_seen = False
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if ignore_first_conv and not first_conv_seen:
            first_conv_seen = True
            print(f"[Skip] 第一个 Conv2d: {name or '<root>'}")
            continue
        first_conv_seen = True
        target_layers[name] = module

    if not target_layers:
        raise ValueError(
            "no Conv2d layers selected for uint8 calibration; if the model "
            "contains only one Conv2d, pass ignore_first_conv=False")

    # 标量 tensor 留在层所在 device 上，hook 中不做逐 batch .item() 同步。
    x_min_record = {
        name: torch.tensor(
            float("inf"), dtype=torch.float32, device=module.weight.device)
        for name, module in target_layers.items()
    }
    x_max_record = {
        name: torch.tensor(
            float("-inf"), dtype=torch.float32, device=module.weight.device)
        for name, module in target_layers.items()
    }
    hooks = []

    def make_hook(layer_name):
        def hook(module, inputs, output):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError(
                    f"input of Conv2d '{layer_name}' must be a torch.Tensor")
            x = inputs[0].detach()
            # qparams 固定以 fp32 保存；标量归约后再转型，不额外物化一份完整
            # fp32 activation（兼顾 fp16/bf16 模型与校准显存）。
            current_min = x.amin().to(dtype=torch.float32)
            current_max = x.amax().to(dtype=torch.float32)
            if not (torch.isfinite(current_min) and torch.isfinite(current_max)):
                raise ValueError(
                    f"non-finite activation observed at Conv2d '{layer_name}'")
            x_min_record[layer_name] = torch.minimum(
                x_min_record[layer_name], current_min)
            x_max_record[layer_name] = torch.maximum(
                x_max_record[layer_name], current_max)
        return hook

    for name, module in target_layers.items():
        hooks.append(module.register_forward_hook(make_hook(name)))

    training_states = {module: module.training for module in model.modules()}
    seen = 0
    model.eval()
    try:
        with torch.no_grad():
            for batch in data_loader:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                if not isinstance(images, torch.Tensor):
                    raise TypeError(
                        "calibration batch must be a Tensor or a list/tuple "
                        "whose first element is a Tensor")
                if images.dim() == 0:
                    raise ValueError("calibration input must have a batch dimension")

                # 不让最后一个 batch 超出 num_pictures；极值统计严格对应用户
                # 请求的样本数，而不是向上取整到整个 batch。
                if num_pictures is not None:
                    remaining = num_pictures - seen
                    if remaining <= 0:
                        break
                    images = images[:remaining]
                if images.shape[0] == 0:
                    continue

                images = images.to(device)
                model(images)
                seen += images.shape[0]
                if num_pictures is not None and seen >= num_pictures:
                    break
    finally:
        for hook in hooks:
            hook.remove()
        # 精确保留校准前可能混合存在的 train/eval 状态（例如冻结的 BN）。
        for module, training in training_states.items():
            module.training = training

    if seen == 0:
        raise ValueError("data_loader yielded no calibration samples")

    qparams = {}
    for name, module in target_layers.items():
        observed_x_min = x_min_record[name]
        observed_x_max = x_max_record[name]
        if not (torch.isfinite(observed_x_min) and torch.isfinite(observed_x_max)):
            raise RuntimeError(
                f"Conv2d '{name}' did not run during calibration")

        # affine quantization must include real zero.  Store these effective
        # ranges so Conv2d_uint8 can reproduce the exact same qparams later.
        x_min = torch.minimum(observed_x_min, torch.zeros_like(observed_x_min))
        x_max = torch.maximum(observed_x_max, torch.zeros_like(observed_x_max))

        weight = module.weight.detach()
        if not torch.isfinite(weight).all():
            raise ValueError(f"non-finite weight observed at Conv2d '{name}'")
        reduce_dims = tuple(range(1, weight.dim()))
        w_min = weight.amin(dim=reduce_dims).to(dtype=torch.float32)
        w_max = weight.amax(dim=reduce_dims).to(dtype=torch.float32)
        w_min = torch.minimum(w_min, torch.zeros_like(w_min))
        w_max = torch.maximum(w_max, torch.zeros_like(w_max))

        scale_x, zero_x_float = uint8_qparams(x_min, x_max)
        scale_w, zero_w_float = uint8_qparams(w_min, w_max)
        # uint8_qparams 已执行 round；zero point 是整数值，但这里必须保留
        # float32 dtype。Conv2d_uint8 的 qparam buffer 和融合 CUDA fake-quant
        # 都使用 float32，直接保存 int32 会导致类型转换或绕过融合路径。
        zero_x = zero_x_float.to(dtype=torch.float32)
        zero_w = zero_w_float.to(dtype=torch.float32)

        qparams[name] = {
            "x_min": x_min.detach().clone(),
            "x_max": x_max.detach().clone(),
            "w_min": w_min.detach().clone(),
            "w_max": w_max.detach().clone(),
            "scale_x": scale_x.detach().clone(),
            "zero_x": zero_x.detach().clone(),
            "scale_w": scale_w.detach().clone(),
            "zero_w": zero_w.detach().clone(),
        }
        print(
            f"[uint8 qparams] {name}: "
            f"x=[{x_min.item():.6g}, {x_max.item():.6g}], "
            f"scale_x={scale_x.item():.8g}, zero_x={zero_x.item()}, "
            f"weight_channels={weight.shape[0]}, "
            f"zero_w=[{zero_w.min().item()}, {zero_w.max().item()}]"
        )

    # state_dict() 本身已经是独立映射；直接扩充可保留 PyTorch 附带的
    # version metadata，避免某些版本化模块 load_state_dict 时丢信息。
    new_state_dict = model.state_dict()

    def state_key(layer_name, field):
        return f"{layer_name}.{field}" if layer_name else field

    for name, layer_qparams in qparams.items():
        for field, value in layer_qparams.items():
            new_state_dict[state_key(name, field)] = value

    if save_path is not None:
        torch.save(new_state_dict, save_path)
        print(f"[Save] uint8 calibration parameters saved to {save_path}")

    return new_state_dict
