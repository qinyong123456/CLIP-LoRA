import torch
import torch.nn.functional as F

from utils import *

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers

def evaluate_lora(args, clip_model, loader, dataset):
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0] 
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    acc /= tot_samples

    return acc


def run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    
    VALIDATION = False
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)
    
    test_features = test_features.cuda()
    test_labels = test_labels.cuda()
 
    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))
    
    test_features = test_features.cpu()
    test_labels = test_labels.cpu()
    
    
    original_r = args.r
    if hasattr(args, 'rank_strategy') and args.rank_strategy == 'warmup_dynamic':
        args.r = getattr(args, 'rank_max', original_r)
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda() 
    
    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots
    
    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    
    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0
    
    # Dynamic warmup and rank allocation
    if hasattr(args, 'rank_strategy') and args.rank_strategy == 'warmup_dynamic':
        warmup_iters = getattr(args, 'rank_warmup_iters', 0)
        scores = [0.0 for _ in range(len(list_lora_layers))]
        # count matrices per layer once
        mats_count = []
        for li, layer in enumerate(list_lora_layers):
            m = 0
            for attr in ['q_proj', 'k_proj', 'v_proj', 'proj']:
                if hasattr(layer, attr):
                    sub = getattr(layer, attr)
                    if hasattr(sub, 'w_lora_A') and hasattr(sub, 'w_lora_B'):
                        m += 1
            mats_count.append(max(1, m))
        scaler = torch.cuda.amp.GradScaler()
        count_warm = 0
        while count_warm < warmup_iters:
            clip_model.train()
            for i, (images, target) in enumerate(tqdm(train_loader)):
                template = dataset.template[0]
                texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
                images, target = images.cuda(), target.cuda()
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts_ids = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts_ids)
                    text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
                    image_features = clip_model.encode_image(images)
                image_features = image_features/image_features.norm(dim=-1, keepdim=True)
                cosine_similarity = logit_scale * image_features @ text_features.t()
                loss = F.cross_entropy(cosine_similarity, target)
                optimizer.zero_grad()
                scaler.scale(loss).backward()

                # accumulate grad norms per layer
                for li, layer in enumerate(list_lora_layers):
                    layer_score = 0.0
                    for attr in ['q_proj', 'k_proj', 'v_proj', 'proj']:
                        if hasattr(layer, attr):
                            sub = getattr(layer, attr)
                            if hasattr(sub, 'w_lora_A') and hasattr(sub, 'w_lora_B'):
                                gA = getattr(sub, 'w_lora_A').grad
                                gB = getattr(sub, 'w_lora_B').grad
                                if gA is not None:
                                    layer_score += gA.norm().item()
                                if gB is not None:
                                    layer_score += gB.norm().item()
                    scores[li] += layer_score
                scaler.step(optimizer)
                scaler.update()

                count_warm += 1
                if count_warm >= warmup_iters:
                    break

        # allocate effective ranks per layer under budget
        B = getattr(args, 'rank_budget', len(list_lora_layers))
        r_min = getattr(args, 'rank_min', 0)
        r_max = getattr(args, 'rank_max', original_r)
        sum_scores = sum(scores)
        eff_r = [r_min for _ in range(len(list_lora_layers))]
        if sum_scores > 0:
            for i in range(len(list_lora_layers)):
                w = scores[i]
                m = mats_count[i]
                ri = int(round((B * (w / sum_scores)) / m))
                ri = max(r_min, min(r_max, ri))
                eff_r[i] = ri
        # adjust to budget precisely
        def total_units(eff):
            return sum((eff[i] * (mats_count[i] if mats_count[i] > 0 else 1)) for i in range(len(eff)))
        # reduce if over budget
        while total_units(eff_r) > B:
            # reduce on smallest scores first
            order = sorted(range(len(list_lora_layers)), key=lambda i: scores[i])
            for i in order:
                if eff_r[i] > r_min:
                    eff_r[i] -= 1
                    if total_units(eff_r) <= B:
                        break
            else:
                break
        # increase if under budget
        while total_units(eff_r) < B:
            order = sorted(range(len(list_lora_layers)), key=lambda i: -scores[i])
            for i in order:
                if eff_r[i] < r_max:
                    eff_r[i] += 1
                    if total_units(eff_r) >= B:
                        break
            else:
                break
        # apply effective_r to each submodule
        for li, layer in enumerate(list_lora_layers):
            ri = eff_r[li]
            for attr in ['q_proj', 'k_proj', 'v_proj', 'proj']:
                if hasattr(layer, attr):
                    sub = getattr(layer, attr)
                    if hasattr(sub, 'effective_r'):
                        setattr(sub, 'effective_r', ri)
        # optional: print allocation summary
        print("Dynamic rank allocation under budget:")
        for li, layer in enumerate(list_lora_layers):
            print(f"Layer {li}: effective_r={eff_r[li]}, mats={mats_count[li]}, score={scores[li]:.4f}")
        # restore r to original if needed
        args.r = original_r

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision': 
            text_features = textual_features.t().half()
        for i, (images, target) in enumerate(tqdm(train_loader)):
            
            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
                
            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            
            cosine_similarity = logit_scale * image_features @ text_features.t()
            loss = F.cross_entropy(cosine_similarity, target)
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()
            
            count_iters += 1
            
            if count_iters == total_iters:
                break
            
        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(current_lr, acc_train, loss_epoch))

        
        # Eval
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))
        
    
    acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))
    
    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return
            
    
            
