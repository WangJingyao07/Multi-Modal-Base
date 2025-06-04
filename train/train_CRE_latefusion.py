import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
# import pdb
from utils.utils import *
from pytorch_pretrained_bert import BertAdam
from models import get_model
from data.helpers_CRE import get_data_loaders
from utils.logger import create_logger



def get_args(parser):
    parser.add_argument('--dataset', default="CREMAD", type=str, choices=["CREMAD", "IEMOCAP"])
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument("--data_path", type=str, default='/data/users/wjy/data/Trian_datasets/Affections/CREMAD')
    parser.add_argument("--bert_model", type=str, default="./bert-base-uncased")#
    parser.add_argument('--LOAD_SIZE', default=256, type=int)
    parser.add_argument('--FiNE_SIZE', default=224, type=int)
    parser.add_argument('--model', default='mml_vt', type=str, choices=['mml_vt','mml_avt', 'latefusion', 'tmc'])
    parser.add_argument("--glove_path", type=str, default="./datasets/glove_embeds/glove.840B.300d.txt")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=24)
    parser.add_argument("--n_classes", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--name", type=str, default="nameless")
    parser.add_argument("--savedir", type=str, default="/path/to/save_dir/")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--task", type=str, default="CREMAD", choices= ["CREMAD", "IEMOCAP"])
    parser.add_argument("--task_type", type=str, default="classification")
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)
    # parser.add_argument("--fusion", type=str, default="concat", choices=["concat", "sum"])
    # parser.add_argument("--mask", type=str, default="soft_mask", choices=["soft_mask", "hard_mask"])
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--noise_type", type=str, default="Gaussian")
    parser.add_argument("--n_workers", type=int, default=4)

def get_criterion(args):
    if args.task_type == "multilabel":
        if args.weight_classes:
            freqs = [args.label_freqs[l] for l in args.labels]
            label_weights = (torch.FloatTensor(freqs) / args.train_data_len) ** -1
            criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights.cuda())
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    return criterion

def get_optimizer(model, args):
    if args.model in ["bert", "concatbert", "mmbt"]:
        total_steps = (
            args.train_data_len
            / args.batch_sz
            / args.gradient_accumulation_steps
            * args.max_epochs
        )
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
        ]
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=args.lr,
            warmup=args.warmup,
            t_total=total_steps,
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return optimizer

def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )


def model_forward(i_epoch, model, args, criterion, batch):
    spec, visual, tgt, idx = batch
    spec, visual, tgt = spec.cuda(), visual.cuda(), tgt.cuda()


    fusion_logits, audio_logits, visual_logits = model(spec, visual)

    audio_clf_loss = nn.CrossEntropyLoss()(audio_logits, tgt)
    visaul_loss = nn.CrossEntropyLoss()(visual_logits, tgt)
    fusion_loss = nn.CrossEntropyLoss()(fusion_logits, tgt)

    loss = fusion_loss
    return loss, audio_clf_loss, visaul_loss, audio_logits, visual_logits, tgt





def model_eval(i_epoch, data, model, args, criterion, store_preds=False):
    model.eval()
    with torch.no_grad():
        losses, preds, tgts, preds_audio, preds_visual = [], [], [], [], []
        # accuracy_2 = 0
        for batch in data:
            av_loss, audio_clf_loss, visaul_loss, audio_logits, visual_logits, tgt = model_forward(i_epoch, model, args, criterion, batch)
            loss = av_loss

            losses.append(loss.item())
# 

            pred = nn.functional.softmax((0.5*audio_logits + 0.5*visual_logits), dim=1).argmax(dim=1).cpu().detach().numpy()
            pred_audio = nn.functional.softmax(audio_logits, dim=1).argmax(dim=1).cpu().detach().numpy()
            pred_visual = nn.functional.softmax(visual_logits, dim=1).argmax(dim=1).cpu().detach().numpy()

            preds.append(pred)
            preds_audio.append(pred_audio)
            preds_visual.append(pred_visual)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)
            # accuracy_2 += torch.eq(torch.from_numpy(pred),torch.from_numpy(tgt)).sum().item() 该计算方式和accuracy_score一样

    metrics = {"loss": np.mean(losses)}
  
    tgts = [l for sl in tgts for l in sl]
    preds = [l for sl in preds for l in sl]
    preds_audio = [l for sl in preds_audio for l in sl]
    preds_visual = [l for sl in preds_visual for l in sl]
    
    metrics["acc"] = accuracy_score(tgts, preds)
    metrics["acc_audio"] = accuracy_score(tgts, preds_audio)
    metrics["acc_visual"] = accuracy_score(tgts, preds_visual)
    metrics["micro_f1"] = f1_score(tgts, preds, average="micro")
        
    # if store_preds:
    #     store_preds_to_disk(tgts, preds, args)
    return metrics





def train(args):

    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name,'logs')
    os.makedirs(args.savedir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.savedir, "tensorboard"))

    train_loader, test_loader = get_data_loaders(args)  
    model = get_model(args)

    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 70, 0.1)


    scheduler = get_scheduler(optimizer, args)
    

    logger = create_logger("%s/logfile.log" % args.savedir, args)
    logger.info(model)
    model.cuda()


    torch.save(args, os.path.join(args.savedir, "args.pt"))

    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf
    
    if os.path.exists(os.path.join(args.savedir, "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(args.savedir, "checkpoint.pt"))
        start_epoch = checkpoint["epoch"]
        n_no_improve = checkpoint["n_no_improve"]
        best_metric = checkpoint["best_metric"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])


    logger.info("Training..")


    for i_epoch in range(start_epoch, args.max_epochs):
        train_losses = []
        model.train()
        optimizer.zero_grad()

        for batch in tqdm(train_loader, total=len(train_loader)):
            loss, audio_clf_loss, visaul_loss, audio_logits, visual_logits, tgt = model_forward(i_epoch, model, args, criterion, batch)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            train_losses.append(loss.item())
            loss.backward()
            global_step += 1
            # if global_step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        metrics = model_eval(i_epoch, test_loader, model, args, criterion)
        writer.add_scalar('Loss/train', loss, i_epoch)
        writer.add_scalar('Accuracy/val', metrics["acc"], i_epoch)
        writer.add_scalar('Accuracy/val_audio', metrics["acc_audio"], i_epoch)
        writer.add_scalar('Accuracy/val_visual', metrics["acc_visual"], i_epoch)
        
        logger.info("Train Loss: {:.4f}, epoch:{}".format(np.mean(train_losses), i_epoch+1))

        log_metrics("Val", metrics, args, logger)

        tuning_metric = (
            metrics["micro_f1"] if args.task_type == "multilabel" else metrics["acc"]
        )
        # scheduler.step()
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

    writer.close()
    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    model.eval()
    test_metrics = model_eval(
            np.inf, test_loader, model, args, criterion, store_preds=True
        )
    # logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))

    log_metrics(f"Test loss", test_metrics, args, logger)


def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    train(args)


if __name__ == "__main__":
    cli_main()
