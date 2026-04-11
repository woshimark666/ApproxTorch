import torch
import torch.nn as nn


def calibrate_int8(model, data_loader, num_pictures, save_path):
    """
    对 CNN 进行 min-max (absmax) 校准,生成 INT8 量化 scale。
    - 跳过第一个 Conv2d 层
    - 不做层融合
    - 激活: per-tensor absmax
    - 权重: per-channel absmax (沿输出通道 O)
    - scale_x: 激活 scale, scale_w: 权重 scale (shape = (O,))
    """
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
        scale_w = absmax_per_oc / 127.0
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