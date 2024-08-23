import argparse
import math
import os
import sys
from operator import truediv
import random
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from tqdm import tqdm
import torch.nn.functional as F
from utils.loss import ce_loss, consistency_loss, BarlowTwinsLoss
from utils.misc import AverageMeter, Accuracy
from utils.ema import ModelEMA
from model.resnest.torch.models.resnest import resnest50 as create_model
from dataset.plaque import get_plaque



def getMixSampleLabel(inputs_x, targets_x,inputs_u_w,inputs_u_s, mean, alpha_one, mu):

    extended_la = 0.5 + 1 / (1 / torch.exp(-5 * mean) + 1)
    duplicated_labels = torch.cat([targets_x for _ in range(mu*2)], dim=0)
    duplicated_samples = torch.cat([inputs_x for _ in range(mu*2)], dim=0)

    duplicated_input_u = torch.cat((inputs_u_s,inputs_u_w),dim=0)

    extended_la = extended_la[duplicated_labels]
    new_sample_all = extended_la.view(-1, 1, 1, 1) * duplicated_samples.cuda() + (1 - extended_la).view(-1, 1, 1, 1) \
                 *duplicated_input_u.cuda()
    new_sample_w,new_sample_s, new_label = balance_classes(duplicated_samples, new_sample_all, duplicated_labels, alpha_one)

    return new_sample_w, new_sample_s, new_label

def balance_classes(new_sample_w, new_sample_s, labels_all, alpha):
    new_sample_w = new_sample_w.cpu()
    new_sample_s = new_sample_s.cpu()
    labels_all = labels_all.cpu()
    num_classes = (1 - alpha) * len(labels_all)
    num_classes = num_classes.to(torch.int)
    max_class_count = max(num_classes)
    balanced_sample_s = []
    balanced_sample_w = []
    balanced_labels = []

    if max_class_count > 0:
        for i, count in enumerate(num_classes):
            indices = np.where(labels_all == i)[0]
            if len(indices) > 0:
                num_to_add = max_class_count - count
                if num_to_add > 0:
                    selected_indices = np.random.choice(indices, size=int(num_to_add), replace=True)
                    balanced_sample_w.extend(new_sample_w[selected_indices])
                    balanced_sample_s.extend(new_sample_s[selected_indices])
                    balanced_labels.extend([i] * num_to_add)
        if len(balanced_labels) > 0:
            balanced_sample_w = torch.stack(balanced_sample_w)
            balanced_sample_s = torch.stack(balanced_sample_s)
            balanced_labels = torch.tensor(balanced_labels)
            if balanced_labels.size(0) == 1:
                balanced_sample_w = torch.cat((balanced_sample_w, balanced_sample_w), dim=0)
                balanced_sample_s = torch.cat((balanced_sample_s, balanced_sample_s), dim=0)
                balanced_labels = torch.cat((balanced_labels, balanced_labels), dim=0)

    return balanced_sample_w, balanced_sample_s, balanced_labels


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def interleave(x, size):
    s = list(x.shape)
    p = x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])
    return p


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def probs_adjust(sim, probs):
    all = sim * probs
    return truediv(all, all.sum(dim=1).reshape(-1, 1))

def split_data(args, data):
    s = list(data.shape)
    data = data.reshape([2 * args.mu + 1, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])
    data_x_w = data[:args.batch_size]
    data_u_w, data_u_s = data[args.batch_size:].chunk(2)
    return data_x_w, data_u_w, data_u_s


def cosine_similarity(feats, feat_u):
    num = torch.mm(feats, feat_u.T)
    denom = torch.norm(feats) * torch.norm(feat_u)
    return (num / denom).T


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    if args.seed is not None:
        set_seed(args.seed)
    labeled_dataset, unlabeled_dataset, test_dataset = get_plaque(args)

    labeled_loader = DataLoader(
        labeled_dataset,
        sampler=RandomSampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        sampler=RandomSampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    model = create_model(args=args, num_classes=args.num_classes, feature=args.feature_dim).to(args.device)
    parameter = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameter, lr=args.lr)
    args.epochs = math.ceil(args.total_steps / args.eval_step)
    model.zero_grad()
    if args.use_ema:
        ema_model = ModelEMA(args, model, args.ema_decay)
    train(args, labeled_loader, unlabeled_loader, test_loader, model, optimizer, ema_model)


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader, model, optimizer, ema_model):
    p_t = 1 / args.num_classes
    tau_t = 1 / args.num_classes
    device = args.device
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_u = AverageMeter()
        loss_u = BarlowTwinsLoss()
        features_dict = {}
        label_dict = {}
        count_bal = 0
        p_bar = tqdm(range(args.eval_step), file=sys.stdout)
        num_classes = torch.zeros((args.num_classes,), dtype=torch.long).to(device)
        alpha_one = torch.ones((args.num_classes,), dtype=torch.long).to(device)

        for batch_idx in range(args.eval_step):
            labeled_iter = iter(labeled_trainloader)
            index_x, inputs_x, targets_x = next(labeled_iter)
            targets_x = targets_x.to(device)
            unlabeled_iter = iter(unlabeled_trainloader)
            index_u, (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).to(device)
            targets_x = targets_x.to(args.device)
            logits, feats = model(inputs)
            logits_x_w, logits_u_w, logits_u_s = split_data(args, logits)
            feats_x_w, feats_u_w, feats_u_s = split_data(args, feats)

            del logits, feats

            feature_tensors = [item.cpu().detach() for item in features_dict.values()]
            feats_all = torch.stack(feature_tensors).to(device)
            labels_all = torch.tensor(list(label_dict.values())).to(device)
            all_feats = torch.zeros((args.num_classes, 128)).to(device)
            all_nums = torch.zeros((args.num_classes,)).to(device)
            for i in range(len(labels_all)):
                all_feats[labels_all[i]] = all_feats[labels_all[i]] + feats_all[i]
                all_nums[labels_all[i]] = all_nums[labels_all[i]] + 1

            per_feats = torch.nan_to_num(all_feats / all_nums.reshape(-1, 1))

            sim_u_feat = cosine_similarity(per_feats, feats_u_w)

            Lx = ce_loss(logits_x_w, targets_x.type(torch.LongTensor).to(args.device), reduction='mean')
            pseudo_label_w = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
            pseudo_label_w_adjust = probs_adjust(sim_u_feat, pseudo_label_w)

            max_probs_w_m, targets_u_w_m = torch.max(pseudo_label_w_adjust, dim=-1)
            max_probs_w, targets_u_w = torch.max(pseudo_label_w, dim=-1)

            p_t = p_t * args.ema_decay + (1. - args.ema_decay) * pseudo_label_w.mean(dim=0)
            tau_t = tau_t * args.ema_decay + (1. - args.ema_decay) * max_probs_w.mean()
            tau_t_c = (p_t / torch.max(p_t, dim=-1)[0])
            mask = max_probs_w_m.ge(args.threshold_2)


            for step, i in enumerate(targets_u_w_m):
                if max_probs_w_m[step] > args.threshold_2:
                    num_classes[i] = num_classes[i] + 1
            alpha = torch.tensor([(1 - num_classes[0] / (num_classes.sum() + 1e-6)),
                                  (1 - num_classes[1] / (num_classes.sum() + 1e-6)),
                                  (1 - num_classes[2] / (num_classes.sum() + 1e-6))]).to(device)
            alpha_one = alpha_one * args.ema_decay + alpha * (1 - args.ema_decay)
            Lu = consistency_loss(logits_u_s, targets_u_w.type(torch.LongTensor).to(args.device), name='ce',
                                  mask=mask.float(), weight=None, alpha=None)

            new_sample_w, new_sample_s, new_targets = getMixSampleLabel(inputs_x, targets_x, inputs_u_w, inputs_u_s,
                                                                        tau_t * tau_t_c, alpha_one, args.mu)
            if len(new_sample_w) > 0:
                new_sample_w = new_sample_w.to(device)
                new_sample_s = new_sample_s.to(device)
                new_logits, new_feats = model(new_sample_s)

                new_logits_pseudo = torch.softmax(new_logits.detach() / args.T, dim=-1)

                max_probs_new, targets_u_new = torch.max(new_logits_pseudo, dim=-1)
                max_probs_new = max_probs_new.to(device)
                new_targets = new_targets.to(device)
                mask_new = max_probs_new.ge(args.threshold_1)

                new_reliable_sample = new_sample_s * (mask_new)[:, None, None, None]

                new_unreliable_sample = new_sample_w * torch.logical_not(mask_new).bool()[:, None, None, None]

                mix_sample = new_reliable_sample + new_unreliable_sample
                data_augment_sample_logits, _ = model(mix_sample.to(device))
                count_bal += len(new_targets)
                Lu_b = ce_loss(data_augment_sample_logits, new_targets.type(torch.LongTensor).to(args.device),
                               reduction='mean')
                Lu = Lu + Lu_b

            Lu2 = loss_u(feats_u_w, feats_u_s)
            loss = Lx + Lu + 0.005 * Lu2
            loss.backward()
            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            mask_u.update(mask.float().mean().item())
            optimizer.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            optimizer.zero_grad()
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. mask1 {mask:.4f}.count{count:4} ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    mask=mask_u.avg,
                    count=count_bal
                ))
            p_bar.update()
        p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        test_loss, test_acc = acc(args, test_loader, test_model)

        if best_acc < test_acc:
            best_acc = test_acc
            print('best acc is:{:.4f}'.format(best_acc))
            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            filepath = os.path.join(args.save_name + '.pth')
            torch.save(ema_to_save.state_dict() if args.use_ema else model_to_save.state_dict(), filepath)
    print('best acc is:{:.4f}'.format(best_acc))


def acc(args, test_loader, model):
    losses = AverageMeter()
    acces = AverageMeter()
    test_loader = tqdm(test_loader, file=sys.stdout)
    with torch.no_grad():
        for batch_idx, (_, inputs, targets) in enumerate(test_loader):
            model.eval()
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs, _ = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            acc = Accuracy(outputs, targets)

            losses.update(loss.item(), inputs.shape[0])
            acces.update(acc, inputs.shape[0])
    print('val loss is:{:.4f} val acc is:{:.4f}'.format(losses.avg, acces.avg))
    return losses.avg, acces.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BSGP Training')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='number of workers')
    parser.add_argument('--num-labeled', type=int, default=282,
                        help='number of labeled data')
    parser.add_argument('--feature-dim', type=int, default=128,
                        help='number of feature dim')
    parser.add_argument('--total-steps', default=50 * 150, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=50, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--num_classes', default=3, type=int,
                        help='number of classes')
    parser.add_argument('--batch-size', default=8, type=int,
                        help='train batchsize')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        help='initial learning rate')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--mu', default=2, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--lambda-f', default=5e-3, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--seed', default=700, type=int,
                        help="random seed")
    parser.add_argument('--threshold_1', default=0.85, type=float)
    parser.add_argument('--threshold_2', default=0.95, type=float)
    parser.add_argument('--save_name', default='BSGP')
    parser.add_argument('--ave_class', default=True, type=bool)
    parser.add_argument('--k', default=7, type=int)
    parser.add_argument('--train_path', default='')
    parser.add_argument('--val_path', default='')
    args = parser.parse_args()
    main(args)