import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
import re

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import pdb
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
    # 训练相关参数
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

    out, a,v  = model(spec, visual)
    weight_size = model.fusion_module.fc_out.weight.size(1)
    out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                     + model.fusion_module.fc_out.bias / 2)

    out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                     + model.fusion_module.fc_out.bias / 2)
    # audio_clf_loss = nn.CrossEntropyLoss()(a, tgt)
    # visaul_loss = nn.CrossEntropyLoss()(v, tgt)
    audio_visual_loss = nn.CrossEntropyLoss()(out, tgt)

    loss = audio_visual_loss 



    return loss, out, out_a, out_v, tgt





def model_eval(i_epoch, data, model, args, criterion, store_preds=False):
    model.eval()
    with torch.no_grad():
        losses, preds, tgts, preds_audio, preds_visual = [], [], [], [], []
        # accuracy_2 = 0
        for batch in data:
            loss, out, a, v, tgt = model_forward(i_epoch, model, args, criterion, batch)
            losses.append(loss.item())

        
            pred = nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()
            pred_audio = nn.functional.softmax(a, dim=1).argmax(dim=1).cpu().detach().numpy()
            pred_visual = nn.functional.softmax(v, dim=1).argmax(dim=1).cpu().detach().numpy()

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
    args.savedir = os.path.join(args.savedir, args.name, str(args.seed))
    os.makedirs(args.savedir, exist_ok=True)


    train_loader, test_loader = get_data_loaders(args)  
    model = get_model(args)

    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 70, 0.1)


    # scheduler = get_scheduler(optimizer, args)
    

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
            loss, out, a, v, tgt = model_forward(i_epoch, model, args, criterion, batch)

            # if args.gradient_accumulation_steps > 1:
            #     loss = loss / args.gradient_accumulation_steps

            train_losses.append(loss.item())
            loss.backward()

            score_v = sum([nn.functional.softmax(v, dim=1)[i][tgt[i]] for i in range(v.size(0))])
            score_a = sum([nn.functional.softmax(a, dim=1)[i][tgt[i]] for i in range(a.size(0))])

            ratio_v = score_v / score_a
            ratio_a = 1 / ratio_v

            """
            Below is the Eq.(10) in our CVPR paper:
                    1 - tanh(alpha * rho_t_u), if rho_t_u > 1
            k_t_u =
                    1,                         else
            coeff_u is k_t_u, where t means iteration steps and u is modality indicator, either a or v.
            """

            if ratio_v > 1:
                coeff_v = 1 - nn.functional.tanh(0.3 * nn.ReLU()(ratio_v))
                coeff_a = 1
            else:
                coeff_a = 1 - nn.functional.tanh(0.3 * nn.ReLU()(ratio_a))
                coeff_v = 1

            # if args.use_tensorboard:
            #     iteration = epoch * len(dataloader) + step
            #     writer.add_scalar('data/ratio v', ratio_v, iteration)
            #     writer.add_scalar('data/coefficient v', coeff_v, iteration)
            #     writer.add_scalar('data/coefficient a', coeff_a, iteration)

            if 0 <= i_epoch <= 50: # bug fixed
                 for name, parms in model.named_parameters():
                        layer = str(name).split('.')[1]

                        if 'audio' in name and len(parms.grad.size()) == 4:
                                # if args.modulation == 'OGM_GE':
                            parms.grad = parms.grad * coeff_a + \
                                                torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                                # elif args.modulation == 'OGM':
                                    # parms.grad *= coeff_a

                        if 'visual' in name and len(parms.grad.size()) == 4:
                                # if args.modulation == 'OGM_GE':
                            parms.grad = parms.grad * coeff_v + \
                                                torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                                # elif args.modulation == 'OGM':
                                    # parms.grad *= coeff_v

            global_step += 1
            # if global_step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        metrics = model_eval(i_epoch, test_loader, model, args, criterion)
        logger.info("Train Loss: {:.4f}, epochs: {}".format(np.mean(train_losses), i_epoch+1))
        log_metrics("Val", metrics, args, logger)

        tuning_metric = (
            metrics["micro_f1"] if args.task_type == "multilabel" else metrics["acc"]
        )
        scheduler.step()
        # scheduler.step(tuning_metric)

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
    # parser = argparse.ArgumentParser(description="Train Models")
    # get_args(parser)
    # args, remaining_args = parser.parse_known_args()
    # model = get_model(args)

    # print(model)
