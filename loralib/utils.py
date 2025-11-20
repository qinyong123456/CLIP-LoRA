import os

import torch
import torch.nn as nn

from typing import Dict

from .layers import LoRALayer, PlainMultiheadAttentionLoRA

INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}


INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},

    'ViT-L/14': {
        'half-up': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'half-bottom': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}
}


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def get_lora_parameters(model, bias='none'):
    params = []
    for name, param in model.named_parameters():
        if bias == 'none':
            if 'lora_' in name:
                params.append(param)
        elif bias == 'all':
            if 'lora_' in name or 'bias' in name:
                params.append(param)
        elif bias == 'lora_only':
            if 'lora_' in name:
                params.append(param)
                bias_name = name.split('lora_')[0] + 'bias'
                if bias_name in model.state_dict():
                    bias_param = dict(model.named_parameters())[bias_name]
                    params.append(bias_param)
        else:
            raise NotImplementedError
    return params


def apply_lora(args, clip_model):
    list_lora_layers = []
    per_layer_r = []

    def assign_r_for_index(encoder_type, idx):
        if args.rank_strategy == 'uniform':
            return args.r
        pos = args.position
        dataset = getattr(args, 'dataset', '')
        is_fine = dataset in ['stanford_cars', 'fgvc', 'oxford_pets']
        r_h = args.rank_high
        r_m = args.rank_mid
        r_l = args.rank_low
        if pos == 'all':
            if idx <= 3:
                base = r_l
            elif idx <= 7:
                base = r_m
            else:
                base = r_h
        else:
            if encoder_type == 'vision':
                if pos in ['top', 'top3', 'up']:
                    base = r_h
                elif pos in ['mid', 'half-up']:
                    base = r_m
                else:
                    base = r_l
            else:
                if pos in ['top1', 'top3', 'up']:
                    base = r_m
                elif pos in ['mid', 'half-up']:
                    base = r_m
                else:
                    base = r_l
        if args.dataset_rank_bias == 'auto' and is_fine:
            base = min(r_h, base + 2)
        return base

    text_indices = []
    vision_indices = []
    if args.encoder == 'text' or args.encoder == 'both':
        text_indices = INDEX_POSITIONS_TEXT[args.position]
    if args.encoder == 'vision' or args.encoder == 'both':
        vision_indices = INDEX_POSITIONS_VISION[args.backbone][args.position]

    assigned_rs_text = [assign_r_for_index('text', i) for i in text_indices]
    assigned_rs_vision = [assign_r_for_index('vision', i) for i in vision_indices]

    all_assigned = assigned_rs_text + assigned_rs_vision
    if len(all_assigned) > 0 and args.rank_strategy == 'static_map':
        total = sum(all_assigned) * len(args.params)
        budget = args.rank_budget
        if total > budget:
            scale = budget / float(total)
            scaled = [max(1, int(round(r * scale))) for r in all_assigned]
            diff = budget - sum(scaled) * len(args.params)
            if diff < 0:
                for k in range(len(scaled)):
                    if diff == 0:
                        break
                    if scaled[k] > 1:
                        scaled[k] -= 1
                        diff += len(args.params)
            ptr = 0
            assigned_rs_text = scaled[:len(assigned_rs_text)]
            assigned_rs_vision = scaled[len(assigned_rs_text):]

    if args.encoder == 'text' or args.encoder == 'both':
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            print(f"Residual Attention Block {i}: {block}")
            if i in text_indices:
                r_i = assigned_rs_text[text_indices.index(i)] if args.rank_strategy == 'static_map' else args.r
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        print(f"Assigned r={r_i} for text block {i}")
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=r_i, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
                        per_layer_r.append(r_i)

    if args.encoder == 'vision' or args.encoder == 'both':
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            print(f"Residual Attention Block {i}: {block}")
            if i in vision_indices:
                r_i = assigned_rs_vision[vision_indices.index(i)] if args.rank_strategy == 'static_map' else args.r
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        print(f"Assigned r={r_i} for vision block {i}")
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=r_i, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
                        per_layer_r.append(r_i)
    setattr(args, 'per_layer_r', per_layer_r)
    return list_lora_layers


def save_lora(args, list_lora_layers):
    weights = {}
    for i, layer in enumerate(list_lora_layers):
        layer_weights = {}
        if 'q' in args.params:
            layer_weights['q_proj'] = {
                'w_lora_A': layer.q_proj.w_lora_A.data,
                'w_lora_B': layer.q_proj.w_lora_B.data
            }
        if 'k' in args.params:
            layer_weights['k_proj'] = {
                'w_lora_A': layer.k_proj.w_lora_A.data,
                'w_lora_B': layer.k_proj.w_lora_B.data
            }
        if 'v' in args.params:
            layer_weights['v_proj'] = {
                'w_lora_A': layer.v_proj.w_lora_A.data,
                'w_lora_B': layer.v_proj.w_lora_B.data
            }
        if 'o' in args.params:
            layer_weights['proj'] = {
                'w_lora_A': layer.proj.w_lora_A.data,
                'w_lora_B': layer.proj.w_lora_B.data
            }

        weights[f'layer_{i}'] = layer_weights

    metadata = {
        'r': args.r,
        'alpha': args.alpha,
        'encoder': args.encoder,
        'params': args.params,
        'position': args.position,
        'rank_strategy': getattr(args, 'rank_strategy', 'uniform'),
        'rank_budget': getattr(args, 'rank_budget', None),
        'per_layer_r': getattr(args, 'per_layer_r', None)
    }

    save_data = {
        'weights': weights,
        'metadata': metadata
    }

    # to manage names like ViT-B/16
    backbone = args.backbone.replace('/', '').replace('-', '').lower()
    save_dir = f'{args.save_path}/{backbone}/{args.dataset}/{args.shots}shots/seed{args.seed}'
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/{args.filename}.pt'
    torch.save(save_data, save_path)
    print(f'LoRA weights saved to {save_path}')


def load_lora(args, list_lora_layers):
    # to manage names like ViT-B/16
    backbone = args.backbone.replace('/', '').replace('-', '').lower()
    load_path = f'{args.save_path}/{backbone}/{args.dataset}/{args.shots}shots/seed{args.seed}/{args.filename}.pt'

    if not os.path.exists(load_path):
        raise FileNotFoundError(f'File {load_path} does not exist.')

    loaded_data = torch.load(load_path)

    metadata = loaded_data['metadata']
    if 'per_layer_r' in metadata and metadata['per_layer_r'] is not None:
        current = getattr(args, 'per_layer_r', None)
        if current is None:
            raise ValueError('per_layer_r missing in current args')
        if len(current) != len(metadata['per_layer_r']):
            raise ValueError('per_layer_r length mismatch')
        for i in range(len(current)):
            if current[i] != metadata['per_layer_r'][i]:
                found = metadata['per_layer_r'][i]
                raise ValueError(f"per_layer_r mismatch at layer {i}: expected {current[i]}, found {found}")
    else:
        if metadata['r'] != args.r:
            raise ValueError(
                f"r mismatch: expected {args.r}, found {metadata['r']}")
    if metadata['alpha'] != args.alpha:
        raise ValueError(
            f"alpha mismatch: expected {args.alpha}, found {metadata['alpha']}")
    if metadata['encoder'] != args.encoder:
        raise ValueError(
            f"Encoder mismatch: expected {args.encoder}, found {metadata['encoder']}")
    if metadata['params'] != args.params:
        raise ValueError(
            f"Params mismatch: expected {args.params}, found {metadata['params']}")
    if metadata['position'] != args.position:
        raise ValueError(
            f"Position mismatch: expected {args.position}, found {metadata['position']}")

    weights = loaded_data['weights']
    for i, layer in enumerate(list_lora_layers):
        layer_weights = weights[f'layer_{i}']
        if 'q' in args.params and 'q_proj' in layer_weights:
            layer.q_proj.w_lora_A.data.copy_(
                layer_weights['q_proj']['w_lora_A'])
            layer.q_proj.w_lora_B.data.copy_(
                layer_weights['q_proj']['w_lora_B'])
        if 'k' in args.params and 'k_proj' in layer_weights:
            layer.k_proj.w_lora_A.data.copy_(
                layer_weights['k_proj']['w_lora_A'])
            layer.k_proj.w_lora_B.data.copy_(
                layer_weights['k_proj']['w_lora_B'])
        if 'v' in args.params and 'v_proj' in layer_weights:
            layer.v_proj.w_lora_A.data.copy_(
                layer_weights['v_proj']['w_lora_A'])
            layer.v_proj.w_lora_B.data.copy_(
                layer_weights['v_proj']['w_lora_B'])
        if 'o' in args.params and 'proj' in layer_weights:
            layer.proj.w_lora_A.data.copy_(layer_weights['proj']['w_lora_A'])
            layer.proj.w_lora_B.data.copy_(layer_weights['proj']['w_lora_B'])

    print(f'LoRA weights loaded from {load_path}')
