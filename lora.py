import torch
import torch.nn.functional as F
import math
from utils import *

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers

# 添加C-AdamW优化器实现
class CAdamW(torch.optim.AdamW):
    """
    C-AdamW优化器：AdamW的变种，增加了一个额外的校正项
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, c=0.2):
        super(CAdamW, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.c = c  # 校正系数
        
    def step(self, closure=None):
        """执行单步优化"""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 获取梯度
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('C-AdamW does not support sparse gradients')
                
                # 应用原始AdamW的更新逻辑
                state = self.state[p]
                
                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    # 指数移动平均
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # 指数移动平均的平方
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # 应用权重衰减
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # 更新偏置校正的指数平均
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 偏置校正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # 计算自适应学习率
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                # 应用C-AdamW的校正项
                step_size = group['lr'] / bias_correction1
                
                # 添加校正项
                correction = self.c * torch.sign(exp_avg) * torch.sqrt(exp_avg_sq)
                
                # 更新参数
                p.data.addcdiv_(exp_avg + correction, denom, value=-step_size)
                
        return loss

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
    
    
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda() 
    
    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots
    
    # 将AdamW替换为C-AdamW
    optimizer = CAdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr, c=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    
    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0
    
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
            
    
            
