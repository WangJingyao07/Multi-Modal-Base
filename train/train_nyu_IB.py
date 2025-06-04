import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

import torchvision.transforms as transforms
from data.aligned_conc_dataset import AlignedConcDataset
from utils.utils import *
from models import get_model
from data.helpers_nyu import get_data_loaders

from utils.logger import create_logger
import os
from torch.utils.data import DataLoader
from data.dataset import AddGaussianNoise, AddSaltPepperNoise


def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=16)
    parser.add_argument("--data_path", type=str, default="/data/users/wjy/data/Multi_Modal/QMF/datasets/nyud2_trainvaltest")
    parser.add_argument("--LOAD_SIZE", type=int, default=256)
    parser.add_argument("--FINE_SIZE", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=40)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=512)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4) 
    parser.add_argument("--lr_factor", type=float, default=0.3)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--name", type=str, default="my_model3")
    parser.add_argument("--model", type=str, default="mml")
    parser.add_argument("--num_image_embeds", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--savedir", type=str, default="./saved/NYUD3/5")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--CONTENT_MODEL_PATH", type=str, default="./checkpoint/resnet18_pretrained.pth")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--noise_type", type=str, default="Gaussian")

def get_optimizer(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    return optimizer


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )




def model_forward(i_epoch, model, args, batch):

    rgb, depth, tgt = batch['A'], batch['B'], batch['label']
    idx = batch['idx']

 
    rgb, depth, tgt = rgb.cuda(), depth.cuda(), tgt.cuda()
    depth_rgb_logits, rgb_logits, depth_logits= model(rgb, depth)

    noise = torch.randn(depth_rgb_logits.shape) * np.sqrt(1.0 / 2 * np.pi)
    lmi_mix = torch.mean(torch.linalg.vector_norm(depth_rgb_logits + noise.cuda(), ord=2, dim=1)) 
    # lmi_r = torch.mean(torch.linalg.vector_norm(rgb_logits + noise.cuda(), ord=2, dim=1))
    # lmi_d = torch.mean(torch.linalg.vector_norm(depth_logits + noise.cuda(), ord=2, dim=1))
    # lmi = lmi_mix + lmi_r + lmi_d
    lmi = lmi_mix


    depth_rgb_clf_loss = nn.CrossEntropyLoss()(depth_rgb_logits, tgt)
    depth_clf_loss = nn.CrossEntropyLoss()(depth_logits, tgt)
    rgb_clf_loss = nn.CrossEntropyLoss()(rgb_logits, tgt)
    loss = depth_rgb_clf_loss + depth_clf_loss+ rgb_clf_loss


    return loss, lmi, depth_rgb_logits, rgb_logits, depth_logits, tgt
    # return depth_rgb_clf_loss, lmi, depth_rgb_logits, rgb_logits, depth_logits, tgt


def model_eval(i_epoch, data, model, args, store_preds=False):
    model.eval()
    with torch.no_grad():
        losses, depth_preds, rgb_preds, depthrgb_preds, tgts = [], [], [], [], []
        for batch in data:
            loss, lmi, depth_rgb_logits, rgb_logits, depth_logits, tgt = model_forward(i_epoch, model, args, batch)
            losses.append(loss.item())

            depth_pred = depth_logits.argmax(dim=1).cpu().detach().numpy()
            rgb_pred = rgb_logits.argmax(dim=1).cpu().detach().numpy()
            depth_rgb_pred = depth_rgb_logits.argmax(dim=1).cpu().detach().numpy()

            depth_preds.append(depth_pred)
            rgb_preds.append(rgb_pred)
            depthrgb_preds.append(depth_rgb_pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)

    metrics = {"loss": np.mean(losses)}

    tgts = [l for sl in tgts for l in sl]
    depth_preds = [l for sl in depth_preds for l in sl]
    rgb_preds = [l for sl in rgb_preds for l in sl]
    depthrgb_preds = [l for sl in depthrgb_preds for l in sl]
    metrics["depth_acc"] = accuracy_score(tgts, depth_preds)
    metrics["rgb_acc"] = accuracy_score(tgts, rgb_preds)
    metrics["depthrgb_acc"] = accuracy_score(tgts, depthrgb_preds)
    metrics["f1_socre"] = f1_score(tgts, depthrgb_preds, average='micro')
    return metrics


def train(args):

    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name, str(args.seed))
    os.makedirs(args.savedir, exist_ok=True)


    train_loader, test_loader = get_data_loaders(args)  
        
    model = get_model(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    logger = create_logger("%s/logfile.log" % args.savedir, args)
    logger.info(model)
    model.cuda()

    torch.save(args, os.path.join(args.savedir, "args.pt"))
    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf



    for i_epoch in range(start_epoch, args.max_epochs):
        train_losses = []
        model.train()
        optimizer.zero_grad()
        for batch in tqdm(train_loader, total=len(train_loader)):
            loss, lmi, depth_rgb_logits, rgb_logits, depth_logits, tgt = model_forward(i_epoch, model, args, batch)
            if args.gradient_accumulation_steps > 1:
                 loss = loss / args.gradient_accumulation_steps
                 lmi = lmi / args.gradient_accumulation_steps
                

            train_losses.append(loss.item())
            (loss + lmi).backward()
            # loss.backward()

            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        metrics = model_eval(
            np.inf, test_loader, model, args, store_preds=True
            # i_epoch, val_loader, model, args, store_preds=True

        )
        logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
        # log_metrics("val", metrics, logger)
        logger.info(
            "{}: Loss: {:.5f} | depth_acc: {:.5f}, rgb_acc: {:.5f}, depth rgb acc: {:.5f}, f1_score: {:.5f}".format(
                "val", metrics["loss"], metrics["depth_acc"], metrics["rgb_acc"], metrics["depthrgb_acc"], metrics["f1_socre"]
            )
        )
        tuning_metric = metrics["depthrgb_acc"]

        scheduler.step(tuning_metric)
        
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1

        save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
        )

        # if n_no_improve >= args.patience:
        #     logger.info("No improvement. Breaking out of loop.")
        #     break

    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    model.eval()
    test_metrics = model_eval(
        np.inf, test_loader, model, args, store_preds=True
    )
    logger.info(
        "{}: Loss: {:.5f} | depth_acc: {:.5f}, rgb_acc: {:.5f}, depth rgb acc: {:.5f}".format(
            "Test", test_metrics["loss"], test_metrics["depth_acc"], test_metrics["rgb_acc"],
            test_metrics["depthrgb_acc"]
        )
    )
    # log_metrics(f"Test", test_metrics, logger)


def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    train(args)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    cli_main()
