
import argparse
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_pretrained_bert import BertAdam
from torch.utils.tensorboard import SummaryWriter

from data.helpers import get_data_loaders
from models import get_model
from utils.logger import create_logger
from utils.utils import *

def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=16)
    parser.add_argument("--bert_model", type=str, default="./bert-base-uncased")#, choices=["bert-base-uncased", "bert-large-uncased"])
    parser.add_argument("--data_path", type=str, default="/data/users/wjy/data/Multi_Modal/QMF/text-image-classification/datasets")
    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed_sz", type=int, default=300)
    parser.add_argument("--freeze_img", type=int, default=3)
    parser.add_argument("--freeze_txt", type=int, default=5)
    parser.add_argument("--glove_path", type=str, default="./datasets/glove_embeds/glove.840B.300d.txt")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=40)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=5e-05)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--model", type=str, default="mml_avt", choices=["mutual","mml_avt", "latefusion"])
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--name", type=str, default="mml_VT02_1")
    parser.add_argument("--num_image_embeds", type=int, default=3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--savedir", type=str, default="./saved/MVSA_VAE")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--task", type=str, default="MVSA_Single", choices=["mmimdb", "vsnli", "food101","MVSA_Single"])
    parser.add_argument("--task_type", type=str, default="classification")
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)
    parser.add_argument("--noise_type", type=str, default="Gaussian", choices=["clean", "gaussian", "dropout"])
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--CONTENT_MODEL_PATH", type=str, default="./checkpoint/resnet18_pretrained.pth")
    parser.add_argument("--dataset", type=str, default="MVSA")

    # parser.add_argument("--beta", type=float, default=0.5)
    # parser.add_argument("--unified_hidden_sz", type=int, default=768)  #
    # parser.add_argument("--unified_dim", type=int, default=1024)
    # parser.add_argument("--class_dim", type=int, default=128)
    # parser.add_argument("--cos_dim", type=int, default=768)
    # parser.add_argument("--num_heads", type=int, default=4)
    # parser.add_argument("--alpha1", type=float, default=0.2)
    # parser.add_argument("--alpha2", type=float, default=0.9)




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


def model_eval(i_epoch, data, model, args, criterion, store_preds=False):
    model.eval()
    with torch.no_grad():
        losses, preds, tgts, preds_txt, preds_img = [], [], [], [], []
        for batch in data:
            # loss, txt_img_logits, txt_logits, img_logits, tgt, l_mi = model_forward(i_epoch, model, args, criterion, batch)
            fusion_loss, fusion_logits, txt_clf_loss, img_clf_loss, txt_logits, img_logits, tgt = model_forward(i_epoch, model, args, criterion, batch)
            losses.append(fusion_loss.item())

        
            pred = torch.nn.functional.softmax(fusion_logits, dim=1).argmax(dim=1).cpu().detach().numpy()
            pred_txt = torch.nn.functional.softmax(txt_logits, dim=1).argmax(dim=1).cpu().detach().numpy()
            pred_img = torch.nn.functional.softmax(img_logits, dim=1).argmax(dim=1).cpu().detach().numpy()

            preds.append(pred)
            preds_txt.append(pred_txt)
            preds_img.append(pred_img)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)

    metrics = {"loss": np.mean(losses)}
  
    tgts = [l for sl in tgts for l in sl]
    preds = [l for sl in preds for l in sl]
    preds_txt = [l for sl in preds_txt for l in sl]
    preds_img = [l for sl in preds_img for l in sl]
    metrics["acc"] = accuracy_score(tgts, preds)
    metrics["acc_txt"] = accuracy_score(tgts, preds_txt)
    metrics["acc_img"] = accuracy_score(tgts, preds_img)
    metrics["micro_f1"] = f1_score(tgts, preds, average="micro")
        

    if store_preds:
        store_preds_to_disk(tgts, preds, args)

    return metrics



 
def model_forward(i_epoch, model, args, criterion, batch):
    txt, segment, mask, img, tgt, idx = batch

    txt, mask, segment, img, tgt = txt.cuda(), mask.cuda(), segment.cuda(), img.cuda(), tgt.cuda()
    # txt_img_logits, txt_logits, img_logits = model(txt, mask, segment, img)
    fusion_logits, txt_logits, img_logits = model(txt, mask, segment, img)
   
    fusion_loss = nn.CrossEntropyLoss()(fusion_logits, tgt)
    txt_clf_loss = nn.CrossEntropyLoss()(txt_logits, tgt)
    img_clf_loss = nn.CrossEntropyLoss()(img_logits, tgt)


    return fusion_loss, fusion_logits, txt_clf_loss, img_clf_loss, txt_logits, img_logits, tgt



def train(args):

    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name,'logs')
    os.makedirs(args.savedir, exist_ok=True)

    train_loader, val_loader, test_loaders = get_data_loaders(args)
    writer = SummaryWriter(log_dir=os.path.join(args.savedir, "tensorboard"))

    model = get_model(args)

    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)
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
            fusion_loss, fusion_logits, txt_clf_loss, img_clf_loss, txt_logits, img_logits, tgt = model_forward(i_epoch, model, args, criterion, batch)
            _loss = 0

            loss_value_mm, loss_value_v, loss_value_t = [], [], []
            cos_txt, cos_visual = [], []
            record_names_txt, record_names_visual  = [], []

        
            for name, param in model.named_parameters():
                if 'head' in name: 
                    continue
                if ('txt' in name):
                    record_names_txt.append((name, param))
                    # print("txt", name)
                    continue
                if ('img' in name):
                    record_names_visual.append((name, param))
                    # print("visual", name)
                    
                    continue

            loss_value_mm.append(fusion_loss.item())
            loss_value_v.append(img_clf_loss.item())
            loss_value_t.append(txt_clf_loss.item())

            losses=[fusion_loss, txt_clf_loss, img_clf_loss]
            all_loss = ['both', 'txt', 'visual']

            grads_txt = {}
            grads_visual={}

            for idx, loss_type in enumerate(all_loss):
                loss = losses[idx]
                loss.backward(retain_graph=True)  # 保留计算图，下次还要使用

                if(loss_type=='visual'):
                    for tensor_name, param in record_names_visual:
                        if loss_type not in grads_visual.keys():
                            grads_visual[loss_type] = {}
                        grads_visual[loss_type][tensor_name] = param.grad.data.clone()
                    grads_visual[loss_type]["concat"] = torch.cat([grads_visual[loss_type][tensor_name].flatten()  for tensor_name, _ in record_names_visual])
                    # print("visual")
                elif(loss_type=='txt'):
                    for tensor_name, param in record_names_txt:
                        if loss_type not in grads_txt.keys():
                            grads_txt[loss_type] = {}
                        grads_txt[loss_type][tensor_name] = param.grad.data.clone()
                    grads_txt[loss_type]["concat"] = torch.cat([grads_txt[loss_type][tensor_name].flatten()  for tensor_name, _ in record_names_txt])
                    # print("txt")
                else:
                    for tensor_name, param in record_names_txt:
                        if loss_type not in grads_txt.keys():
                            grads_txt[loss_type] = {}
                        grads_txt[loss_type][tensor_name] = param.grad.data.clone() 
                    grads_txt[loss_type]["concat"] = torch.cat([grads_txt[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_txt])
                    for tensor_name, param in record_names_visual:
                        if loss_type not in grads_visual.keys():
                            grads_visual[loss_type] = {}
                        grads_visual[loss_type][tensor_name] = param.grad.data.clone() 
                    grads_visual[loss_type]["concat"] = torch.cat([grads_visual[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_visual])

                optimizer.zero_grad()
        
            this_cos_txt = F.cosine_similarity(grads_txt['both']["concat"],grads_txt['txt']["concat"],dim=0)
            this_cos_visual = F.cosine_similarity(grads_visual['both']["concat"],grads_visual['visual']["concat"],dim=0)

            txt_task=['both','txt']
            visual_task=['both','visual']

            txt_k=[0,0]
            visual_k=[0,0]

            if(this_cos_txt>0):
                txt_k[0]=0.5
                txt_k[1]=0.5
            else:
                txt_k, min_norm = MinNormSolver.find_min_norm_element([list(grads_txt[t].values()) for t in txt_task])
            if(this_cos_visual>0):
                visual_k[0]=0.5
                visual_k[1]=0.5
            else:
                visual_k, min_norm = MinNormSolver.find_min_norm_element([list(grads_visual[t].values()) for t in visual_task])



            gamma=1.5

            loss = fusion_loss + txt_clf_loss + img_clf_loss
            loss.backward()


            for name, param in model.named_parameters():
                if param.grad is not None:
                    layer = re.split('[_.]',str(name))
                    if('head' in layer):
                        continue
                    if('txt' in layer):
                        three_norm=torch.norm(param.grad.data.clone())
                        new_grad=2*txt_k[0]*grads_txt['both'][name]+2*txt_k[1]*grads_txt['txt'][name]
                        new_norm=torch.norm(new_grad)
                        diff=three_norm/new_norm
                        if(diff>1):
                            param.grad=diff*new_grad*gamma
                        else:
                            param.grad=new_grad*gamma

                    if('visual' in layer):
                        three_norm=torch.norm(param.grad.data.clone())
                        new_grad=2*visual_k[0]*grads_visual['both'][name]+2*visual_k[1]*grads_visual['visual'][name]
                        new_norm=torch.norm(new_grad)
                        diff=three_norm/new_norm
                        if(diff>1):
                            param.grad=diff*new_grad*gamma
                        else:
                            param.grad=new_grad*gamma

            optimizer.step()

            train_losses.append(loss.item())

            global_step += 1
        
            optimizer.zero_grad()

        metrics = model_eval(i_epoch, val_loader, model, args, criterion)
        writer.add_scalar('Loss/train', np.mean(train_losses), i_epoch)
        writer.add_scalar('Accuracy/val', metrics["acc"], i_epoch)
        writer.add_scalar('Accuracy/val_audio', metrics["acc_txt"], i_epoch)
        writer.add_scalar('Accuracy/val_visual', metrics["acc_img"], i_epoch)
        logger.info("Train Loss: {:.4f}, epoch:{}".format(np.mean(train_losses), i_epoch+1))

        log_metrics("Val", metrics, args, logger)

        tuning_metric = (
            metrics["micro_f1"] if args.task_type == "multilabel" else metrics["acc"]
        )
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
    for test_name, test_loader in test_loaders.items():
        test_metrics = model_eval(
            np.inf, test_loader, model, args, criterion, store_preds=True
        )
        log_metrics(f"Test - {test_name}", test_metrics, args, logger)


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


