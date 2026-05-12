import os
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

parser = argparse.ArgumentParser(description="CIFAR-10 ResNet G-DSP Training")
parser.add_argument("--save_dir", type=str, default="./cifarmodel/", help="Folder to save checkpoints and log.")
parser.add_argument("-l", "--layers", default=20, type=int, metavar="L", help="number of ResNet layers")
parser.add_argument("-d", "--device", default="0", type=str, metavar="D", help="main device (default: 0)")
parser.add_argument("-j", "--workers", default=4, type=int, metavar="J", help="number of data loading workers (default: 4)")
parser.add_argument("--epochs", default=120, type=int, metavar="E", help="number of total epochs to run")
parser.add_argument("-b", "--batch-size", default=128, type=int, metavar="B", help="mini-batch size")
parser.add_argument("--lr", "--learning-rate", default=0.05, type=float, metavar="LR", help="initial learning rate")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument("--weight-decay", "--wd", default=1e-3, type=float, metavar="W", help="weight decay")

# G-DSP Hyperparameters
parser.add_argument("-g", "--groups", default=4, type=int, metavar="G", help="number of groups")
parser.add_argument("--tau-geo", default=0.5, type=float, metavar="T", help="temperature for geometric Gumbel-Softmax")
parser.add_argument("--lambda-geo", default=1e-2, type=float, metavar="LG", help="weight for geometric alignment loss")
parser.add_argument("--lambda-ortho", default=1e-3, type=float, metavar="LO", help="weight for centroid orthogonality loss")
parser.add_argument("--projection-dim", default=0, type=int, metavar="PD", help="optional bottleneck dim before cosine logits (0 disables)")
parser.add_argument("--kmeans-iters", default=20, type=int, metavar="KI", help="iterations for pre-training spherical K-Means")
parser.add_argument("--kmeans-restarts", default=4, type=int, metavar="KR", help="restarts for pre-training spherical K-Means")
parser.add_argument("--results-csv", default="./results/gdsp_stage1.csv", type=str, help="CSV file to append experiment results")
parser.add_argument("--exp-name", default="", type=str, help="optional experiment tag")
# Backward-compatible legacy DSP flags (accepted but not used in G-DSP objective).
parser.add_argument("-r", "--regularize", default=0.0, type=float, metavar="R", help="legacy DSP sparsity coeff (ignored)")
parser.add_argument("-t", "--temparature", default=None, type=float, metavar="T0", help="legacy typo flag; maps to --tau-geo")

args = parser.parse_args()
if args.temparature is not None:
    args.tau_geo = args.temparature
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
    loadpath = args.save_dir + "/" + netname + ".pkl"
    savepath = args.save_dir + "/" + netname + "_G%sg%d.pkl" % (args.device, args.groups)
    state_dict, baseacc = torch.load(loadpath)
    print(loadpath)
    print(baseacc)
    cnn.load_state_dict(state_dict, strict=False)
    criterion = nn.CrossEntropyLoss()
    bestacc = 0
    best_top5 = 0.0
    best_epoch = -1

    # GroupWrapper updates centroids/projectors with its own Adam optimizer.
    # Keep SGD only on base network parameters.
    net_params = [p for n, p in cnn.named_parameters() if (not n.endswith("centroids")) and ("projectors." not in n)]
    optimizer = torch.optim.SGD(net_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epochs)
    group_trainer = GroupWrapper(
        cnn,
        n_groups=args.groups,
        tau_geo=args.tau_geo,
        projection_dim=args.projection_dim,
        kmeans_iters=args.kmeans_iters,
        kmeans_restarts=args.kmeans_restarts,
    )

    bar = tqdm(total=len(train_loader) * args.epochs, ncols=140)
    for epoch in range(args.epochs):
        cnn.train()
        for step, (images, labels) in enumerate(train_loader):
            group_trainer.initialize()
            optimizer.zero_grad()
            group_trainer.zero_grad()
            gpuimg = images.to(device)
            labels = labels.to(device)

            outputs = cnn(gpuimg)
            task_loss = criterion(outputs, labels)
            geo_loss = group_trainer.geometric_alignment_loss()
            ortho_loss = group_trainer.orthogonality_loss()
            loss = task_loss + args.lambda_geo * geo_loss + args.lambda_ortho * ortho_loss

            loss.backward()
            optimizer.step()
            group_trainer.step()
            group_trainer.harden_assignments()
            bar.set_description(
                "["
                + config
                + "]LR:%.4f|LOSS:%.3f|GEO:%.3f|ORTH:%.3f|ACC:%.2f|CONF:%.3f"
                % (get_lr(optimizer)[0], loss.item(), geo_loss.item(), ortho_loss.item(), bestacc, group_trainer.stats())
            )
            bar.update()

        scheduler.step()

        acc, acc5 = evaluate_topk(cnn, test_loader, topk=(1, 5))
        print()
        print(f"Val top1/top5 accuracy: {acc:.2f}% / {acc5:.2f}%")
        cnn.train()

        if (bestacc < acc) and (epoch > 8):
            bestacc = acc
            best_top5 = acc5
            best_epoch = epoch
            torch.save([cnn.state_dict(), bestacc], savepath)
            bar.set_description(
                "["
                + config
                + "]LR:%.4f|LOSS:%.3f|GEO:%.3f|ORTH:%.3f|ACC:%.2f|CONF:%.3f"
                % (get_lr(optimizer)[0], loss.item(), geo_loss.item(), ortho_loss.item(), bestacc, group_trainer.stats())
            )

    bar.close()
    append_result_row(
        args.results_csv,
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "exp_name": args.exp_name,
            "script": "cifar_dsp.py",
            "model": netname,
            "layers": args.layers,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "groups": args.groups,
            "tau_geo": args.tau_geo,
            "lambda_geo": args.lambda_geo,
            "lambda_ortho": args.lambda_ortho,
            "projection_dim": args.projection_dim,
            "kmeans_iters": args.kmeans_iters,
            "kmeans_restarts": args.kmeans_restarts,
            "top1_acc": round(float(bestacc), 4),
            "top5_acc": round(float(best_top5), 4),
            "params_reduced_pct": "",
            "delta_params_reduced_pct": "",
            "flops_reduced_pct": "",
            "delta_flops_reduced_pct": "",
            "best_epoch": best_epoch,
            "checkpoint": savepath,
        },
    )
    print(f"[LOG] Results appended to {args.results_csv}")
    return bestacc


def resnet(layers):
    return CifarResNet(ResNetBasicblock, layers, 10).to(device), "resnet" + str(layers)


if __name__ == "__main__":
    train(resnet(args.layers))
