import torch
from .nn import Conv2d_int8, Conv2d_uint8
import copy

# collect the min/max for each layer's activation and weight
# min/max is for uint type
def collect_minmax(model, dataloader, qmethod, num_batches=None):
    model.eval()
    activation_min = {}
    activation_max = {}
    weight_min = {}
    weight_max = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            x = input[0]
            min_val = torch.min(x)
            max_val = torch.max(x)
            
            if name not in activation_max:
                activation_max[name] = max_val
            else:
                activation_max[name] = max(activation_max[name], max_val)
                
                
            if name not in activation_min:
                activation_min[name] = min_val
            else:
                activation_min[name] = min(activation_min[name], min_val)
                
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
                weight_max[name] = torch.amax(weight, dim=(1,2,3), keepdim=False)
                weight_min[name] = torch.amin(weight, dim=(1,2,3), keepdim=False)
            elif qmethod[2] == 'tensor':
                weight_max[name] = torch.max(weight, keepdim=False)
                weight_min[name] = torch.min(weight, keepdim=False)
            else:
                raise ValueError(f"Invalid qmethod: {qmethod}")

    return activation_max, activation_min, weight_max, weight_min


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
def calibrate_uint8(model, train_loader, data_precentage, qmethod=('static', 'tensor', 'tensor'), save_path=None):
    
    # Calculate number of batches to process based on percentage
    num_batches = int(len(train_loader) * data_precentage) if data_precentage < 1.0 else None
    
    # Collect min/max values
    activation_max, activation_min, weight_max, weight_min = collect_minmax(model, train_loader, qmethod, num_batches)
    first_conv_found = False
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if not first_conv_found:
                # Skip the first Conv2d layer
                first_conv_found = True
                continue
            # 权重scale: channel级
            w_scale = (weight_max[name] - weight_min[name]) / 255.
            w_zero = - torch.round(weight_min[name] / w_scale)
            a_scale = (activation_max[name] - activation_min[name]) / 255.
            a_zero = - torch.round(activation_min[name] / a_scale)
            
            w_scale = w_scale.detach().clone()
            w_zero = w_zero.detach().clone()
            a_scale = a_scale.detach().clone()
            a_zero = a_zero.detach().clone()
            # 注册buffer（如果已存在则覆盖）
            if hasattr(module, 'scale_feature'):
                module.activation_scale.copy_(a_scale)
            else:
                module.register_buffer('scale_feature', a_scale)
                
            if hasattr(module, 'zero_feature'):
                module.zero_feature.copy_(a_zero)
            else:
                module.register_buffer('zero_feature', a_zero)
                
            if hasattr(module, 'scale_weight'):
                module.weight_scale.copy_(w_scale)
            else:
                module.register_buffer('scale_weight', w_scale)
                
            if hasattr(module, 'zero_weight'):
                module.zero_weight.copy_(w_zero)
            else:
                module.register_buffer('zero_weight', w_zero)
            

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"State dict with scales saved to {save_path}")



def calibrate_int8(model, train_loader, data_precentage, qmethod=('static', 'tensor', 'tensor'), save_path=None):
    
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
        if isinstance(module, all_conv):
            module.freeze_scale()

def unforze_scale(model):
    """
    input the model and unforze all the scale parameters
    """
    for name, module in model.named_modules():
        if isinstance(module, all_conv):
            module.unfreeze_scale()