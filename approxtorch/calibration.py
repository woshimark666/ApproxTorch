import torch
import json


def collect_min_max_values(model, dataloader, num_batches=None):
    """
    Collect min and max values of inputs for all Conv2d and Linear layers
    Args:
        model: The model to analyze
        dataloader: DataLoader for collecting statistics
        num_batches: Number of batches to process (None for all batches)
    Returns:
        Dictionary containing min/max values for each layer
    """
    model.eval()
    feature_stats = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            if name not in feature_stats:
                feature_stats[name] = {
                    'min': float('inf'),
                    'max': float('-inf'),
                    'abs_max': float('-inf')
                }
            # Get the input tensor
            x = input[0]
            # Update min and max values
            feature_stats[name]['min'] = min(feature_stats[name]['min'], x.min().item())
            feature_stats[name]['max'] = max(feature_stats[name]['max'], x.max().item())
            feature_stats[name]['abs_max'] = max(feature_stats[name]['abs_max'], x.abs().max().item())
        return hook
    
    # Register hooks for all Conv2d and Linear layers
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Process data
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if num_batches is not None and i >= num_batches:
                break
            inputs = inputs.cuda()
            model(inputs)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    
    # get T for weights
    weight_stats = {}
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            layer_weight = getattr(module, 'weight', None)

            if layer_weight is not None:
                weight_stats[name] = {}
                weight_stats[name]['min'] = layer_weight.min().item()
                weight_stats[name]['max'] = layer_weight.max().item()
                weight_stats[name]['abs_max'] = layer_weight.abs().max().item()
                
    return feature_stats, weight_stats


def calibrate(model, train_loader, data_precentage, save_path=None):
    
    batch_size = train_loader.batch_size
    # Calculate number of batches to process based on percentage
    num_batches = int(len(train_loader) * data_precentage) if data_precentage < 1.0 else None
    
    # Collect min/max values
    feature_stats, weight_stats = collect_min_max_values(model, train_loader, num_batches)
    all_stats = {}
    all_stats['feature'] = feature_stats
    all_stats['weight'] = weight_stats
    
    # save into json file
    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump(all_stats, f, indent=4)
    
    return all_stats