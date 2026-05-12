import os
import math
import copy
import csv
import warnings
import argparse
from datetime import datetime

import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from cifar_model import *
from dsp_module import *


def warn(*args, **kwargs):
    pass


warnings.warn = warn

benchmark_mode(True)

parser = argparse.ArgumentParser(description="CIFAR-10 ResNet Fine-Tuning")
parser.add_argument("--save_dir", type=str, default="./cifarmodel/", help="Folder to save checkpoints and log.")
parser.add_argument("-l", "--layers", default=20, type=int, metavar="L", help="number of ResNet layers")
parser.add_argument("-d", "--device", default="0", type=str, metavar="D", help="main device (default: 0)")
parser.add_argument("-j", "--workers", default=4, type=int, metavar="J", help="number of data loading workers (default: 4)")
parser.add_argument("--epochs", default=120, type=int, metavar="E", help="number of total epochs to run")
parser.add_argument("-b", "--batch-size", default=128, type=int, metavar="B", help="mini-batch size")
parser.add_argument("--lr", "--learning-rate", default=0.015, type=float, metavar="LR", help="initial learning rate")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument("--weight-decay", "--wd", default=1e-3, type=float, metavar="W", help="weight decay")

# Fine-tuning + G-DSP fusion Hyperparameters
parser.add_argument("-c", "--cycles", default=5, type=int, metavar="C", help="number of cyclic iterations")
parser.add_argument("-g", "--groups", default=4, type=int, metavar="G", help="number of groups")
parser.add_argument("-p", "--prune", default=0.5, type=float, metavar="P", help="pruning rates")
parser.add_argument("--merge-threshold", default=0.95, type=float, metavar="MTH", help="cosine threshold for adaptive filter fusion")
parser.add_argument("--disable-fusion", action="store_true", help="disable adaptive geometric filter fusion")
parser.add_argument("--results-csv", default="./results/gdsp_stage2.csv", type=str, help="CSV file to append experiment results")
parser.add_argument("--exp-name", default="", type=str, help="optional experiment tag")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device


def get_lr(optimizer):
    return [param_group["lr"] for param_group in optimizer.param_groups]


def append_result_row(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def evaluate_topk(cnn, loader, topk=(1, 5)):
    cnn.eval()
    maxk = max(topk)
    total = 0
    correct_k = [0 for _ in topk]
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = cnn(images)
            _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))
            total += labels.size(0)
            for i, k in enumerate(topk):
                correct_k[i] += correct[:k].any(dim=0).float().sum().item()
    return [(100.0 * c / max(1, total)) for c in correct_k]


device = torch.device("cuda")


def evaluate(cnn, test_loader):
    top1, _ = evaluate_topk(cnn, test_loader, topk=(1, 5))
    return top1


def train(network):
    train_dataset = dsets.CIFAR10(
        root="./dataset",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
            ]
        ),
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, drop_last=True
    )
    test_dataset = dsets.CIFAR10(
        root="./dataset",
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
            ]
        ),
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False
    )

    cnn, netname = network
    config = netname
    savepath = args.save_dir + "/" + netname + "_P%sg%dc%.2f.pkl" % (args.device, args.groups, args.prune)
    loadpath = args.save_dir + "/" + netname + "_G%sg%d.pkl" % (args.device, args.groups)
    state_dict, baseacc = torch.load(loadpath)
    print(loadpath)
    print(baseacc)

    pruner = PruneWrapper(cnn, args.groups, 2)
    cnn.load_state_dict(state_dict, strict=False)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(cnn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    epoch_per_cycle = math.ceil(args.epochs / args.cycles)
    scheduler = CosineAnnealingLR(optimizer, epoch_per_cycle)

    flops, params = pruner.initialize(args.prune)
    init_flops = float(flops)
    init_params = float(params)
    bestset = {"acc": 0, "flops": flops, "params": params, "state_dict": copy.deepcopy(cnn.state_dict())}

    bar = tqdm(total=len(train_loader) * args.epochs, ncols=140)
    for epoch in range(args.epochs):
        cnn.train()
        for step, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            gpuimg = images.to(device)
            labels = labels.to(device)

            outputs = cnn(gpuimg)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            pruner.after_step()
            bar.set_description(
                "["
                + config
                + "]LR:%.4f|LOSS:%.2f|ACC:%.2f|PR_F:%.2f|PR_P:%.2f"
                % (get_lr(optimizer)[0], loss.item(), bestset["acc"], bestset["flops"], bestset["params"])
            )
            bar.update()

        scheduler.step()
        acc = evaluate(cnn, test_loader)
        print()
        print(f"Val accuracy: {acc}%")
        cnn.train()

        if bestset["acc"] <= acc:
            bestset = {"acc": acc, "flops": flops, "params": params, "state_dict": copy.deepcopy(cnn.state_dict())}
            torch.save([bestset["state_dict"], bestset["acc"]], savepath)
            bar.set_description(
                "["
                + config
                + "]LR:%.4f|LOSS:%.2f|ACC:%.2f|PR_F:%.2f|PR_P:%.2f"
                % (get_lr(optimizer)[0], loss.item(), bestset["acc"], bestset["flops"], bestset["params"])
            )

        if (epoch < args.epochs - 1) and ((epoch + 1) % epoch_per_cycle == 0):
            cnn.load_state_dict(bestset["state_dict"])
            flops, params = pruner.initialize(bestset["flops"] / 100 + 0.001)
            optimizer = torch.optim.SGD(cnn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, epoch_per_cycle)

    bar.close()

    # G-DSP adaptive intra-group fusion (post-training stage).
    cnn.load_state_dict(bestset["state_dict"])
    pre_fusion_top1 = float(bestset["acc"])
    pre_fusion_top5 = evaluate_topk(cnn, test_loader, topk=(1, 5))[1]
    fused_acc = None
    fused_top5 = None
    merges = []
    if not args.disable_fusion:
        cnn, merges = adaptive_geometric_fusion(cnn, n_groups=args.groups, gamma_merge=args.merge_threshold)
        fused_acc, fused_top5 = evaluate_topk(cnn, test_loader, topk=(1, 5))
        print(f"Adaptive fusion merges: {len(merges)}")
        print(f"Fused model top1/top5 accuracy: {fused_acc:.2f}% / {fused_top5:.2f}%")
        if fused_acc >= bestset["acc"]:
            bestset["acc"] = fused_acc
            bestset["state_dict"] = copy.deepcopy(cnn.state_dict())
            torch.save([bestset["state_dict"], bestset["acc"]], savepath)

    final_top1 = float(bestset["acc"])
    final_top5 = fused_top5 if fused_top5 is not None else pre_fusion_top5
    final_flops = float(bestset["flops"])
    final_params = float(bestset["params"])

    append_result_row(
        args.results_csv,
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "exp_name": args.exp_name,
            "script": "cifar_finetune.py",
            "model": netname,
            "layers": args.layers,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "cycles": args.cycles,
            "groups": args.groups,
            "prune": args.prune,
            "merge_threshold": args.merge_threshold,
            "disable_fusion": int(args.disable_fusion),
            "top1_acc": round(final_top1, 4),
            "top5_acc": round(float(final_top5), 4),
            "best_top1_before_fusion": round(pre_fusion_top1, 4),
            "best_top5_before_fusion": round(float(pre_fusion_top5), 4),
            "fused_top1_acc": "" if fused_acc is None else round(float(fused_acc), 4),
            "fused_top5_acc": "" if fused_top5 is None else round(float(fused_top5), 4),
            "num_merges": len(merges),
            "params_reduced_pct": round(final_params, 4),
            "delta_params_reduced_pct": round(final_params - init_params, 4),
            "flops_reduced_pct": round(final_flops, 4),
            "delta_flops_reduced_pct": round(final_flops - init_flops, 4),
            "checkpoint": savepath,
        },
    )
    print(f"[LOG] Results appended to {args.results_csv}")
    return bestset["acc"]


def resnet(layers):
    return CifarResNet(ResNetBasicblock, layers, 10).to(device), "resnet" + str(layers)


if __name__ == "__main__":
    train(resnet(args.layers))
