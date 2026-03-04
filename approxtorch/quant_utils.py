import torch
from .nn import Conv2d_int8, Conv2d_uint8
import copy


# collect the min/max for each layer's activation and weight
def collect_minmax(model, dataloader, w_quantizer, num_batches=None):
    model.eval()
    x_min = {}
    a_max = {}
    w_min = {}
    w_max = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            x = input[0]
            min_val = torch.min(x)
            max_val = torch.max(x)
            
            if name not in a_max:
                a_max[name] = max_val
            else:
                a_max[name] = max(a_max[name], max_val)
                
            if name not in x_min:
                x_min[name] = min_val
            else:
                x_min[name] = min(x_min[name], min_val)
                
        return hook
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if num_batches is not None and i > num_batches:
                break
            inputs = inputs.cuda()
            model(inputs)
    
    for hook in hooks:
        hook.remove()
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            weight = module.weight.data
            
            if w_quantizer[2] == 'channel':
                w_max[name] = torch.amax(weight, dim=(1,2,3), keepdim=False)
                w_min[name] = torch.amin(weight, dim=(1,2,3), keepdim=False)
            elif w_quantizer[2] == 'tensor':
                w_max[name] = torch.max(weight)
                w_min[name] = torch.min(weight)
            else:
                raise ValueError(f"Invalid w_quantizer: {w_quantizer}")

    return a_max, x_min, w_max, w_min


# collect the absmax for each layer's activation and weight
# absmax is for int type
def collect_absmax(model, dataloader, qmethod, num_batches=None):
    model.eval()
    activation_absmax = {}
    weight_absmax = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            x = input[0]
            max_abs = x.abs().max()
            if name not in activation_absmax:
                activation_absmax[name] = max_abs
            else:
                activation_absmax[name] = max(activation_absmax[name], max_abs)
        return hook
    
    # Register hooks for all Conv2d except the first one
    # collect the absmax for activation
    first_conv_found = False
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if not first_conv_found:
                # Skip the first Conv2d layer
                first_conv_found = True
                continue
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Process data
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if num_batches is not None and i > num_batches:
                break
            inputs = inputs.cuda()
            model(inputs)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # collect the absmax for weight
    first_conv_found = False
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if not first_conv_found:
                # Skip the first Conv2d layer
                first_conv_found = True
                continue
            weight = module.weight.data
            
            if qmethod[2] == 'channel':
                weight_reshaped = weight.view(weight.size(0), -1)
                maxabs = weight_reshaped.abs().max(dim=1)[0].cpu()
                weight_absmax[name] = maxabs
                
            elif qmethod[2] == 'tensor':
                weight_absmax[name] = weight.abs().max().cpu()
            else:
                raise ValueError(f"Invalid qmethod: {qmethod}")

    return activation_absmax, weight_absmax

# calibrate for uint8 model
def calibrate_uint8(model, train_loader, data_percentage, ignore_first_conv=True,
                    x_quantizer=('static', 'asymmetric', 'tensor'),
                    w_quantizer=('static', 'asymmetric', 'tensor'),
                    save_path=None):
    
    # Calculate number of batches to process based on percentage
    num_batches = int(len(train_loader) * data_percentage) if data_percentage < 1.0 else None
    
    # Collect min/max values
    x_max, x_min, w_max, w_min = collect_minmax(model, train_loader, 
                                    w_quantizer, num_batches)
    
    # 创建新的state_dict,先复制原有的所有参数
    new_state_dict = model.state_dict().copy()
    
    first_conv_found = False
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if ignore_first_conv and not first_conv_found:
                # Skip the first Conv2d layer
                first_conv_found = True
                continue
            
            # 计算量化参数
            w_scale = (w_max[name] - w_min[name]) / 255.
            w_zero = -torch.round(w_min[name] / w_scale)
            x_scale = (x_max[name] - x_min[name]) / 255.
            x_zero = -torch.round(x_min[name] / x_scale)
            
            # 添加量化参数到新的state_dict
            new_state_dict[f'{name}.scale_x'] = x_scale.detach().clone()
            new_state_dict[f'{name}.zero_x'] = x_zero.detach().clone()
            new_state_dict[f'{name}.scale_w'] = w_scale.detach().clone()
            new_state_dict[f'{name}.zero_w'] = w_zero.detach().clone()
    
    if save_path is not None:
        torch.save(new_state_dict, save_path)
        print(f"State dict with scales saved to {save_path}")
    
    return new_state_dict



def calibrate_int8(model, train_loader, data_precentage, 
                   x_quantizer=('static', 'asymmetric', 'tensor'), 
                   w_quantizer=('static', 'asymmetric', 'tensor'), 
                   save_path=None):
    
    # Calculate number of batches to process based on percentage
    num_batches = int(len(train_loader) * data_precentage) if data_precentage < 1.0 else None
    
    # Collect min/max values
    activation_absmax, weight_absmax = collect_absmax(model, train_loader, qmethod, num_batches)

    first_conv_found = False
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if not first_conv_found:
                # Skip the first Conv2d layer
                first_conv_found = True
                continue
            # 权重scale: channel级
            w_scale = weight_absmax[name].detach().clone() / 127.0
            a_scale = activation_absmax[name].detach().clone() / 127.0
            # 注册buffer（如果已存在则覆盖）
            if hasattr(module, 'scale_feature'):
                module.activation_scale.copy_(torch.tensor(a_scale))
            else:
                module.register_buffer('scale_feature', a_scale)
                
            if hasattr(module, 'scale_weight'):
                module.weight_scale.copy_(w_scale)
            else:
                module.register_buffer('scale_weight', w_scale)
            

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"State dict with scales saved to {save_path}")


def calibrate_int4(model, train_loader, data_precentage, qmethod=('static', 'tensor', 'tensor'), save_path=None):
    
    # Calculate number of batches to process based on percentage
    num_batches = int(len(train_loader) * data_precentage) if data_precentage < 1.0 else None
    
    # Collect min/max values
    activation_absmax, weight_absmax = collect_absmax(model, train_loader, qmethod, num_batches)

    first_conv_found = False
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if not first_conv_found:
                # Skip the first Conv2d layer
                first_conv_found = True
                continue
            # 权重scale: channel级
            w_scale = weight_absmax[name].detach().clone() / 7.5
            a_scale = activation_absmax[name].detach().clone() / 7.5
            # 注册buffer（如果已存在则覆盖）
            if hasattr(module, 'scale_feature'):
                module.activation_scale.copy_(torch.tensor(a_scale))
            else:
                module.register_buffer('scale_feature', a_scale)
                
            if hasattr(module, 'scale_weight'):
                module.weight_scale.copy_(w_scale)
            else:
                module.register_buffer('scale_weight', w_scale)
            

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"State dict with scales saved to {save_path}")





all_conv = (Conv2d_int8, Conv2d_uint8)


def forze_scale(model):
    """
    input the model and forze all the scale parameters
    """
    for name, module in model.named_modules():
        if type(module) in all_conv:
            module.disable_update_qparams()

def unforze_scale(model):
    """
    input the model and unforze all the scale parameters
    """
    for name, module in model.named_modules():
        if type(module) in all_conv:
            module.enable_update_qparams()