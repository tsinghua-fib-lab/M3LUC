import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pdb
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
import pickle
import argparse
from datasets_VLM import my_Dataset


# python train.py  --gpu cpu --n_class 4 --model dpot --output_fp ./test/ --cities yinchuan --use_cache_emb --cache_fp ./cache/test/ --n_epoch 30 --L2 1e-2 --dropout 0.2 --use_popu
args = argparse.ArgumentParser()
args.add_argument("--gpu", type=str, default="cuda:5")
args.add_argument("--seed", type=int, default=1234)
args.add_argument("--bs", type=int, default=6)
args.add_argument("--n_epochs", type=int, default=50)
args.add_argument("--lr_init", type=float, default=1e-4)
args.add_argument("--dropout", type=float, default=0)
args.add_argument("--L2", type=float, default=0)
args.add_argument("--n_class", type=int, default=4)
args.add_argument("--model", type=str, default="dpot")

args.add_argument("--n_layer", type=int, default=2)

## dataset
args.add_argument("--use_cache", action="store_true")
args.add_argument("--use_cache_emb",action="store_true")

args.add_argument("--cache_fp", type=str, default="./cache/BEJ/")
args.add_argument("--data_arg", action="store_true")
args.add_argument("--data_arg_num", type=int, default=500)
args.add_argument("--is_bce", type=bool, default=False)
args.add_argument("--cities", type=str, default='beijing')
args.add_argument("--use_popu", action="store_true")

args.add_argument("--output_fp", type=str, default="./res_BEJ/")
args.add_argument("--silience", action="store_true")

args = args.parse_args()

args.cities = args.cities.split("_") if "_" in args.cities else [args.cities]
torch.manual_seed(args.seed)

if not os.path.exists(args.output_fp):
    os.makedirs(args.output_fp)

if not os.path.exists(args.cache_fp):
    os.makedirs(args.cache_fp + "train")
    os.makedirs(args.cache_fp + "test")


n_epochs = args.n_epochs
device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")

if not args.silience:
    print("Use cache", args.use_cache)
    print(f"Using {device}")
    print("Cities: ", args.cities)


if args.model == "dpot":
    from model_att_dpot import landuse_net_att
    model = landuse_net_att(num_classes=args.n_class, dropout=args.dropout)
    model = model.to(device)
else:
    raise NotImplementedError

print("Model loaded: ", args.model)
## L2

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.L2)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=1e-6)

if not args.use_cache_emb or not os.path.exists(args.cache_fp + "train/train_emb.pkl"):
    pre_process = model.preprocess
    train_dataset = my_Dataset("train", cities=args.cities, 
                            pre_process=pre_process, n_class=args.n_class)

    train_dataLoader = DataLoader(train_dataset, batch_size=6, shuffle=True)
    data_distribution = train_dataset.get_distribution()
    data_distribution[-1] = 10000
    print(data_distribution)    
    weight = 1 / data_distribution

    test_dataset = my_Dataset("test", cities=args.cities, 
                            pre_process=pre_process, n_class=args.n_class)

    test_dataLoader = DataLoader(test_dataset, batch_size=6, shuffle=False)

    train_regions = train_dataset.r_id
    test_regions = test_dataset.r_id
    for r in test_regions:
        assert r not in train_regions

    criteria = nn.CrossEntropyLoss(weight=weight.float().to(device))  


def test(epoch):
    test_results = []
    test_loss = 0
    model.eval()
    with torch.no_grad():
        corr_cnt = 0
        total_cnt = 0
        corr = np.zeros((16,16)) ## 4 catg, TP,FP,TN,FN
        if not args.use_cache_emb:
            for i, (r_id, poi_vec, mask, img, label, img_height, img_building) in tqdm(enumerate(test_dataLoader),total=len(test_dataLoader)):
                for r in r_id:
                    assert r not in train_regions

                poi_vec, mask, img, label, img_height, img_building = \
                    poi_vec.to(device), mask.to(device),img.to(device), label.to(device), img_height.to(device),\
                    img_building.to(device)
                
                popu_batch = [popu_map[r.split(".")[0]] for r in r_id]
                popu_batch = torch.tensor(popu_batch).long().to(device)

                output = model(poi_vec, mask, img, img_height, img_building, popu_batch)

                loss = criteria(output, label)
                test_loss += loss.item()
                preds = torch.argmax(output, dim=1)
                

                corr_cnt += (preds == label).sum()
                total_cnt += len(r_id)

                ## acc for each catg
                preds = preds.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                for reg,l,p in zip(r_id,label,preds):
                    test_results.append((reg,l,p))
                    corr[l,p] += 1
 
        else:
            global test_emb, test_regs, test_labs
            for i, (r_id, emb, lab) in tqdm(enumerate(zip(test_regs, test_emb, test_labs)),total=len(test_regs),
                disable=args.silience):

                emb = torch.tensor(emb).float().to(device)
                lab = torch.tensor(lab).long().to(device)
                popu_batch = [popu_map[r.split(".")[0]] for r in r_id]
                popu_batch = torch.tensor(popu_batch).long().to(device)

                output = model.forward_emb(emb, popu_batch)

                loss = criteria(output, lab)
                test_loss += loss.item()

                preds = torch.argmax(output, dim=1)
                

                corr_cnt += (preds == lab).sum()
                total_cnt += len(r_id)

                ## acc for each catg
                preds = preds.detach().cpu().numpy()
                lab = lab.detach().cpu().numpy()
                for reg,l,p in zip(r_id,lab,preds):
                    test_results.append((reg,l,p))
                    corr[l,p] += 1

        prec = np.zeros((16,))
        rec = np.zeros((16,))

        for i in range(16):
            prec[i] = corr[i,i] / np.sum(corr[:,i])
            rec[i] = corr[i,i] / np.sum(corr[i,:])

        if not args.silience or epoch % 99 == 0:
            print(f"Test Accuracy: {corr_cnt / total_cnt}, {corr_cnt}/{total_cnt}")
            print("Test precision for each class: ", prec)
            print("Test recall for each class: ", rec)
            print("Test loss: ", test_loss / total_cnt)
            print("----------------------")

        pickle.dump(test_results, open(args.output_fp + "test_results.pkl", "wb"))

        return float(corr_cnt / total_cnt), corr_cnt, total_cnt, float(test_loss / total_cnt)
    

def train(n_epochs, use_cache_emb=False, cache_emb=None, cache_regs=None, cache_labs=None):
    train_result = []
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    if not use_cache_emb:
        print("Train","train_dataLoader",len(train_dataLoader))
    for epoch in range(n_epochs):
        train_result = []
        if not args.silience or epoch % 99 == 0:
            print(f"Epoch [{epoch}/{n_epochs}]")

        model.train()
        corr = np.zeros((16,16))
        corr_cnt = 0
        total_cnt = 0
        avg_loss = 0
        if not use_cache_emb:
            for i, (r_id, poi_vec, mask, img, label, img_height, img_building) in tqdm(enumerate(train_dataLoader),total=len(train_dataLoader)):

                poi_vec, mask, img, label, img_height, img_building = \
                    poi_vec.to(device), mask.to(device),img.to(device), label.to(device), img_height.to(device),\
                    img_building.to(device)
                
                popu_batch = [popu_map[r.split(".")[0]] for r in r_id]
                popu_batch = torch.tensor(popu_batch).long().to(device)

                optimizer.zero_grad()
                output = model(poi_vec,mask,img, img_height, img_building, popu_batch)

                loss = criteria(output, label)
                avg_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                preds = torch.argmax(output, dim=1)

                corr_cnt += (preds == label).sum()
                total_cnt += len(r_id)


                preds = preds.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                for reg,l,p in zip(r_id, label,preds):
                    corr[l,p] += 1
                    train_result.append((reg,l,p))

        else:
            for i, (r_id, emb, lab) in tqdm(enumerate(zip(cache_regs, cache_emb, cache_labs)),total=len(cache_regs),
                                           disable=args.silience ):

                emb = torch.tensor(emb).float().to(device)
                lab = torch.tensor(lab).long().to(device)
                popu_batch = [popu_map[r.split(".")[0]] for r in r_id]
                popu_batch = torch.tensor(popu_batch).long().to(device)

                optimizer.zero_grad()
                

                output = model.forward_emb(emb, popu_batch)
                
                loss = criteria(output, lab)
                avg_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                preds = torch.argmax(output, dim=1)

                corr_cnt += (preds == lab).sum()
                total_cnt += len(r_id)

                preds = preds.detach().cpu().numpy()
                lab = lab.detach().cpu().numpy()
                for reg,l,p in zip(r_id, lab,preds):
                    corr[l,p] += 1
                    train_result.append((reg,l,p)) 

        lr_scheduler.step()
        if not args.silience:
            print(f"epoch {epoch} train loss: {avg_loss / total_cnt} lr: {optimizer.param_groups[0]['lr']}")
                    
 
        prec = np.zeros((16,))
        rec = np.zeros((16,))

        for i in range(16):
            prec[i] = corr[i,i] / np.sum(corr[:,i])
            rec[i] = corr[i,i] / np.sum(corr[i,:])

        if not args.silience or epoch % 99 == 0:
            print(f"Train Accuracy: {corr_cnt / total_cnt}, {corr_cnt}/{total_cnt}")
            print("Train precision for each class: ", prec)
            print("Train recall for each class: ", rec)
        train_acc.append(float(corr_cnt / total_cnt))
        train_loss.append(float(avg_loss / total_cnt))

        te, _, _, tl = test(epoch)
        test_acc.append(te)
        test_loss.append(tl)

        pickle.dump(train_result, open(args.output_fp + "train_result.pkl", "wb"))

    return train_acc, test_acc, train_loss, test_loss
        
def get_pretrained_emb(dataloader):
    model.eval()
    embs = []
    regs = []
    labs = []
    for i, (r_id, poi_vec, mask, img, label, img_height, img_building) in tqdm(enumerate(dataloader),total=len(dataloader)):
        poi_vec, mask, img, label, img_height, img_building = \
            poi_vec.to(device), mask.to(device),img.to(device), label.to(device) , img_height.to(device),\
            img_building.to(device)
        
        emb = model.get_embedding(poi_vec, mask, img, img_height, img_building)
        emb = emb.detach().cpu().numpy()

        embs.append(emb)
        regs.append(r_id)
        labs.append(label.detach().cpu().numpy())

    return embs, regs, labs

if __name__ == "__main__":
    if args.use_popu:
        n_quant = 64
        popu_map = {}
        max_p = 0
        for c in args.cities:
            p = pickle.load(open(f"./datasets/source/{c}_popu.pkl", "rb"))
            for k,v in p.items():
                popu_map[c + "_" + str(k)] = v
                max_p = max(max_p, v)
        for k,v in popu_map.items():
            popu_map[k] = int(v / (max_p + 1) * n_quant)

    if args.use_cache_emb:
        if os.path.exists(args.cache_fp + "train/train_emb.pkl"):
            train_emb = pickle.load(open(args.cache_fp + "train/train_emb.pkl", "rb"))
            train_regs = pickle.load(open(args.cache_fp + "train/train_regs.pkl", "rb"))
            train_labs = pickle.load(open(args.cache_fp + "train/train_labs.pkl", "rb"))
        else:
            train_emb, train_regs, train_labs = get_pretrained_emb(train_dataLoader)
            pickle.dump(train_regs, open(args.cache_fp + "train/train_regs.pkl", "wb"))
            pickle.dump(train_labs, open(args.cache_fp + "train/train_labs.pkl", "wb"))
            pickle.dump(train_emb, open(args.cache_fp + "train/train_emb.pkl", "wb"))

        if os.path.exists(args.cache_fp + "test/test_emb.pkl"):
            test_emb = pickle.load(open(args.cache_fp + "test/test_emb.pkl", "rb"))
            test_regs = pickle.load(open(args.cache_fp + "test/test_regs.pkl", "rb"))
            test_labs = pickle.load(open(args.cache_fp + "test/test_labs.pkl", "rb"))
        else:
            test_emb, test_regs, test_labs = get_pretrained_emb(test_dataLoader)
            pickle.dump(test_regs, open(args.cache_fp + "test/test_regs.pkl", "wb"))
            pickle.dump(test_labs, open(args.cache_fp + "test/test_labs.pkl", "wb"))
            pickle.dump(test_emb, open(args.cache_fp + "test/test_emb.pkl", "wb"))

        ## batch-size transform
        
        for i, (r_id, emb, lab) in enumerate(zip(train_regs, train_emb, train_labs)):
            vlm_batch = []
            for r in r_id:
                vlm_emb_path = "./datasets/res_Cog2/" + r.split(".")[0] + ".pkl"
                vlm_emb = pickle.load(open(vlm_emb_path, "rb"))
                vlm_emb = vlm_emb[2].reshape(-1)
                vlm_batch.append(vlm_emb)

            vlm_batch = np.array(vlm_batch)
            # print(emb.shape, vlm_emb.shape)
            emb = np.concatenate([emb, vlm_batch], axis=1)
            train_emb[i] = emb
        
        for i, (r_id, emb, lab) in enumerate(zip(test_regs, test_emb, test_labs)):
            vlm_batch = []
            for r in r_id:
                vlm_emb_path = "./datasets/res_Cog2/" + r.split(".")[0] + ".pkl"
                vlm_emb = pickle.load(open(vlm_emb_path, "rb"))
                vlm_emb = vlm_emb[2].reshape(-1)
                vlm_batch.append(vlm_emb)

            vlm_batch = np.array(vlm_batch)
            # print(emb.shape, vlm_emb.shape)
            emb = np.concatenate([emb, vlm_batch], axis=1)
            test_emb[i] = emb

        data_distribution = np.zeros((args.n_class,))
        for l in train_labs:
            data_distribution += np.bincount(l, minlength=args.n_class)
        data_distribution[data_distribution==0] = 10000
        if not args.silience:
            print(data_distribution)
        weight = (1 / data_distribution)
        weight = weight / np.sum(weight)
        weight = torch.tensor(weight).float().to(device)
        criteria = nn.CrossEntropyLoss(weight=weight)   
        ta, ts, train_loss, test_loss = train(n_epochs, use_cache_emb=True, \
                                              cache_emb=train_emb, cache_regs=train_regs, cache_labs=train_labs)

    else:
        ta, ts, train_loss, test_loss = train(n_epochs)

    print(f"Train Accuracy: {ta[-1]}")
    print(f"Test Accuracy: {ts[-1]}")

    import matplotlib.pyplot as plt
    plt.plot(ta)
    plt.plot(ts)
    plt.legend(["train","test"])
    plt.savefig(args.output_fp + "acc.png")

    plt.figure()
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(["train","test"])
    plt.savefig(args.output_fp + "loss.png")
    ## save
    model_save_path = args.output_fp + "model.pth"
    model = model.to("cpu")
    save_dict = model.state_dict()
    for k in list(save_dict.keys()):
        if "vit_rs" in k or "vit_other" in k or "poi_encoder" in k:
            del save_dict[k]
    torch.save(save_dict, model_save_path)
    
    
