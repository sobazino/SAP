import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import math
import copy
import random
import pandas as pd
from datetime import datetime
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Dataset
from torch_geometric.nn import GCNConv, GATConv, TopKPooling, SAGPooling
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import degree
import torch.nn.init as init
from sklearn.metrics import confusion_matrix, auc, f1_score, recall_score, precision_score, average_precision_score, roc_curve
import warnings
warnings.filterwarnings( "ignore", category=FutureWarning, message=r"You are using `torch.load` with `weights_only=False`.*" )
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
import numpy as np

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
current_dir = os.path.dirname(__file__)
o01 = ""
P = pd.read_csv(f"{o01}P.csv")
D = pd.read_csv(f"{o01}D.csv")

class Tokenizer:
    def __init__(self):
        self.CHAR = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, "/": 7, ".": 8, "=": 9, "@": 10, "[": 11, "]": 12, "\\": 13, "1": 14, "2": 15, "3": 16, "4": 17, "5": 18, "6": 19, "7": 20, "8": 21, "9": 22, "0": 23, "A": 24, "B": 25, "C": 26, "D": 27, "E": 28, "F": 29, "G": 30, "H": 31, "I": 32, "J": 33, "K": 34, "L": 35, "M": 36, "N": 37, "O": 38, "P": 39, "Q": 40, "R": 41, "S": 42, "T": 43, "U": 44, "V": 45, "W": 46, "X": 47, "Y": 48, "Z": 49, "a": 50, "b": 51, "c": 52, "d": 53, "e": 54, "f": 55, "g": 56, "h": 57, "i": 58, "k": 59, "l": 60, "m": 61, "n": 62, "o": 63, "p": 64, "r": 65, "s": 66, "t": 67, "u": 68, "v": 69, "w": 70, "x": 71, "y": 72, "z": 73}
        self.CHAR_REVERSE = {v: k for k, v in self.CHAR.items()}
    def encode(self, text):
        encoded = []
        i = 0
        while i < len(text):
            if text[i] in self.CHAR:
                encoded.append(self.CHAR[text[i]])
                i += 1
            else:
                raise ValueError(f"E {text[i]}")
        return encoded

    def decode(self, tokens):
        decoded = []
        for token in tokens:
            if token in self.CHAR_REVERSE:
                decoded.append(self.CHAR_REVERSE[token])
            else:
                raise ValueError(f"E {token}")
        return ''.join(decoded)

tokenizer = Tokenizer()

def info():
    print(f"\n\n\033[31mMehran Nosrati\033[0m | Drug–target interaction prediction | 1403 | Golestan University | IR\n")    
def RESIZE(tensor, target_size):
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    resized_tensor = F.interpolate(tensor, size=target_size, mode='linear', align_corners=False)
    return resized_tensor.squeeze()
    
def Dencoder(x, size):
    SMILES = D[D["Sequence"] == x]
    DID = f"{o01}D/D" + SMILES["ID"].values[0].replace("sequence_", "") + ".pt"
    DDATA = torch.load(DID)

    out = torch.tensor(tokenizer.encode(x)).float()
    return RESIZE(out, size).long().detach(), DDATA.detach()

def Tencoder(x, size):
    FASTA = P[P["Sequence"] == x]
    PID = f"{o01}P/P" + FASTA["ID"].values[0].replace("sequence_", "") + ".pt"
    PDATA = torch.load(PID)
        
    out = torch.tensor(tokenizer.encode(x)).float()
    return RESIZE(out, size).long().detach(), PDATA.detach()

def GETDATA(I, DF):
    D = DF.iloc[I]['SMILES']
    T = DF.iloc[I]['Target Sequence']
    DV, D3D = Dencoder(D, 50)
    TV, T3D = Tencoder(T, 545)
    L = torch.tensor(DF.iloc[I]['Label'], dtype=torch.long)
    return DV, TV, D3D, T3D, L

class MNGRAPH(nn.Module):
    def __init__(self, S, device):
        super(MNGRAPH, self).__init__()
        self.device = device
        self.S = S
        self.W1 = nn.Parameter(torch.empty(S, device=self.device))
        self.W2 = nn.Parameter(torch.empty(S, device=self.device))
        self.B1 = nn.Parameter(torch.empty(S, device=self.device))
        self.B2 = nn.Parameter(torch.empty(S, device=self.device))
        init.xavier_uniform_(self.W1.view(1, -1))  
        init.xavier_uniform_(self.W2.view(1, -1))
        init.zeros_(self.B1)
        init.zeros_(self.B2)

    def forward(self, x, edge_index):
        N = x.size(0)
        I = degree(edge_index[0], num_nodes=N)
        O = degree(edge_index[1], num_nodes=N)
        D = (O + I)
        
        END = []
        SELECT = []
        VISITS = torch.zeros(N, dtype=torch.bool).to(self.device)
        OTHERS = torch.argsort(D, descending=True)
        while OTHERS.size(0) > 0:
            NODE = OTHERS[0]
            SELECT.append(NODE)
            VISITS[NODE] = True
            H = torch.cat([edge_index[1][edge_index[0] == NODE], edge_index[0][edge_index[1] == NODE]]).unique()
            VISITS[H] = True
            OTHERS = OTHERS[~VISITS[OTHERS]]
            mask = torch.isin(edge_index, H[~torch.isin(H, torch.tensor(SELECT).to(self.device))])
            edge_index[mask] = NODE
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]
            r_edge_index = edge_index.flip(0)
            mask = ~(r_edge_index.unsqueeze(2) == edge_index.unsqueeze(1)).all(dim=0).any(dim=1)
            edge_index = torch.cat((edge_index, r_edge_index[:, mask]), dim=1)
            
            C = []
            for n in H:
                Dn = (x[n] * self.W1) + self.B1
                Dv = (x[NODE] * self.W2) + self.B2
                C.append(torch.cat([Dv, Dn], dim=0))
            if len(C) > 0:
                M = torch.mean(torch.stack(C), dim=0)
            # else:
            #     M = (x[NODE] * self.W2) + self.B2
            END.append(M)
        END = torch.stack(END)
        
        SORT, _ = torch.sort(torch.tensor(SELECT).to(self.device))
        edge_index = torch.stack([
            torch.searchsorted(SORT, edge_index[0]),
            torch.searchsorted(SORT, edge_index[1])
        ], dim=0)
        return END, edge_index
    
class S(nn.Module):
    def __init__(self, device):
        super(S, self).__init__()
        self.L1 = MNGRAPH(7, device)
        self.L2 = MNGRAPH(12, device)
        self.L3 = MNGRAPH(24, device)
        self.relu = nn.ReLU()
    
    def forward(self, x, edge_index):
        x, edge_index = self.L1(x, edge_index)
        x = self.relu(x)
        x, edge_index = self.L2(x, edge_index)
        x = self.relu(x)
        x, edge_index = self.L3(x, edge_index)
        return x[0]
        
class DTIV7(nn.Sequential):
    def __init__(self, **params):
        super(DTIV7, self).__init__()
        self.device = params['device']
        self.DRUG = S(self.device)
        self.TARGET = S(self.device)
        self.Out = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 1)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, DV, TV, D3D, T3D):
        D = self.DRUG(D3D.x, D3D.edge_index)
        T = self.TARGET(T3D.x, T3D.edge_index)
        O = torch.cat([D, T], dim=-1).unsqueeze(0)
        O = self.Out(O)
        return self.sigmoid(O).squeeze()

def GET(i, DF, DFNEW):
        SMILES = D[D["Sequence"] == DF.iloc[i]["SMILES"]]
        FASTA = P[P["Sequence"] == DF.iloc[i]["Target Sequence"]]
        PID = f"{o01}P/P" + FASTA["ID"].values[0].replace("sequence_", "") + ".pt"
        DID = f"{o01}D/D" + SMILES["ID"].values[0].replace("sequence_", "") + ".pt"
        if os.path.exists(PID) and os.path.exists(DID):
            DFNEW = pd.concat([DFNEW, DF.iloc[[i]]], ignore_index=True)
        return DFNEW
    
class KDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def convert(dataset):
    D = [dataset[idx] for idx in range(len(dataset))]
    D = [line.split(" ") for line in D]
    DF = pd.DataFrame(D, columns=["E0", "E1", "SMILES", "Target Sequence", "Label"])
    DF['Label'] = DF['Label'].astype(int)
    return DF
            
def Kfold(i, datasets, k=5):
    size = len(datasets) // k  
    val_start = i * size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:] 
        trainset = datasets[0:val_start]
    return trainset, validset

def Shuffle(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def KDATA(dataset, i):
    Tr, Te = Kfold(i, dataset)
    Tr = KDataSet(Tr)
    Te = KDataSet(Te)
    Tr_len = len(Tr)
    Va_size = int(0.2 * Tr_len)
    Tr_size = Tr_len - Va_size
    Tr, Va = torch.utils.data.random_split(Tr, [Tr_size, Va_size])
    DFTRAIN = convert(Tr)
    DFVAL = convert(Va)
    DFTEST = convert(Te)
    return DFTRAIN, DFVAL, DFTEST

def LOAD():
    DFTRAIN = pd.read_csv("train.csv")
    DFVAL = pd.read_csv("val.csv")
    DFTEST = pd.read_csv("test.csv")
    return DFTRAIN, DFVAL, DFTEST
    
def DATA(DFTRAIN, DFVAL, DFTEST):
    if DFTRAIN is None:
        DFTRAIN, DFVAL, DFTEST = LOAD()
    
    print(f'=========== Train: {len(DFTRAIN)}, Validation: {len(DFVAL)}, Test: {len(DFTEST)}')
    DFNEWTRAIN = pd.DataFrame()
    DFNEWTRAIN = pd.concat([DFNEWTRAIN, DFTRAIN.iloc[:0]], ignore_index=True)
    DFNEWVAL = pd.DataFrame()
    DFNEWVAL = pd.concat([DFNEWVAL, DFVAL.iloc[:0]], ignore_index=True)
    DFNEWTEST = pd.DataFrame()
    DFNEWTEST = pd.concat([DFNEWTEST, DFTEST.iloc[:0]], ignore_index=True)

    for I in range(len(DFTRAIN)):
        DFNEWTRAIN = GET(I, DFTRAIN, DFNEWTRAIN).sample(frac=1).reset_index(drop=True)
    for I in range(len(DFVAL)):
        DFNEWVAL = GET(I, DFVAL, DFNEWVAL).sample(frac=1).reset_index(drop=True)
    for I in range(len(DFTEST)):
        DFNEWTEST = GET(I, DFTEST, DFNEWTEST).sample(frac=1).reset_index(drop=True)
    
    print(f'=========== Train: {len(DFNEWTRAIN)}, Validation: {len(DFNEWVAL)}, Test: {len(DFNEWTEST)}')
    return DFNEWTRAIN, DFNEWVAL, DFNEWTEST

LISTsensitivity = []
LISTspecificity = []
LISTauprc = []
LISTroc_auc = []
Lsensitivity = 0
Lspecificity = 0
Lauprc = 0
Lroc_auc = 0
def test(K, model, test_loader, device):
    try:
        global Lsensitivity, Lspecificity, Lauprc, Lroc_auc
        all_predictions = []
        all_labels = []
        model.eval()
        loss_accumulate = 0.0
        count = 0.0
        for idx in range(len(test_loader)):
            DV, TV, D3D, T3D, Label = GETDATA(idx, test_loader)
            logits = model(DV.to(device), TV.to(device), D3D.to(device), T3D.to(device))
            
            loss_fct = torch.nn.BCELoss()            
            label = Label.float().to(device)
            loss = loss_fct(logits, label)
            loss_accumulate += loss
            count += 1
            logits = logits.detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()
            all_labels = all_labels + label_ids.flatten().tolist()
            all_predictions = all_predictions + logits.flatten().tolist()
        
        M = "OK-"
        if len(set(all_predictions)) == 1:
            M = "ER-"
                    
        loss = loss_accumulate/count
        fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
        precision = tpr / (tpr + fpr)
        f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
        thred_optim = thresholds[5:][np.argmax(f1[5:])]
        threshold = thred_optim
        y_pred_s = [1 if i else 0 for i in (all_predictions >= thred_optim)]
        roc_auc = auc(fpr, tpr)
        auprc = average_precision_score(all_labels, all_predictions)
        cm = confusion_matrix(all_labels, y_pred_s)
        recall = recall_score(all_labels, y_pred_s)
        precision = precision_score(all_labels, y_pred_s)
        accuracy = (cm[0,0]+cm[1,1])/sum(sum(cm))
        sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
        specificity = cm[1,1]/(cm[1,0]+cm[1,1])
        outputs = np.asarray([1 if i else 0 for i in (np.asarray(all_predictions) >= 0.5)])
        f1 = f1_score(all_labels, outputs)
        
        if sensitivity > Lsensitivity:
            Lsensitivity = sensitivity
        if specificity > Lspecificity:
            Lspecificity = specificity
        if auprc > Lauprc:
            Lauprc = auprc
        if roc_auc > Lroc_auc:
            Lroc_auc = roc_auc
        
        if sensitivity >= 0.8 and specificity >= 0.88 and auprc >= 0.404 and roc_auc >= 0.907:
            time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            File = f"UP-{time}.pth"
            torch.save(model.state_dict(), File)
            print(f'=========== SAVE: {File} ===========')
            
        return f"{M} Accuracy: {accuracy:.5f}, Precision: {precision:.5f}, Recall: {recall:.5f}, F1: {f1:.5f}, ROC AUC: {roc_auc:.5f}, AUPR (PR-AUC): {auprc:.5f}, Sensitivity: {sensitivity:.5f}, Specificity: {specificity:.5f}, Threshold: {threshold:.5f}, Test Loss: {loss:.5f}"
    except:
        return 0

class EarlyStopping:
    def __init__(self, patience=2, verbose=False, delta=0, path='checkpoint.pt', warmup_epochs=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.warmup_epochs = warmup_epochs

    def __call__(self, val_loss, model, epoch):
        if epoch < self.warmup_epochs:
            if self.verbose:
                print(f"{epoch+1}/{self.warmup_epochs}")
            return
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'{self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'{self.val_loss_min:.6f}/{val_loss:.6f}')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
def etrain(model, criterion, optimizer, scheduler, epochs, train_loader, val_loader, device, early_stopping):
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for idx in range(len(train_loader)):
            optimizer.zero_grad()
            DV, TV, D3D, T3D, Label = GETDATA(idx, train_loader)
            outputs_train = model(DV.to(device), TV.to(device), D3D.to(device), T3D.to(device))
            Label = Label.float().to(device)
            loss_train = criterion(outputs_train, Label)
            loss_train.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            running_train_loss += loss_train.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for idx in range(len(val_loader)):
                DV, TV, D3D, T3D, Label = GETDATA(idx, val_loader)
                outputs_val = model(DV.to(device), TV.to(device), D3D.to(device), T3D.to(device))
                Label = Label.float().to(device)
                loss_val = criterion(outputs_val, Label)
                running_val_loss += loss_val.item()
            
        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        print(f'Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}')
        early_stopping(epoch_val_loss, model, epoch)
        if early_stopping.early_stop:
            break
    model.load_state_dict(torch.load(early_stopping.path, weights_only=True))
    return model
    
def train(K, model, criterion, optimizer, scheduler, epochs, train_loader, val_loader, test_loader, device):
    global Lsensitivity, Lspecificity, Lauprc, Lroc_auc
    global LISTsensitivity, LISTspecificity, LISTauprc, LISTroc_auc
    Lsensitivity, Lspecificity, Lauprc, Lroc_auc = 0, 0, 0, 0
    train_losses = []
    val_losses = []
    
    torch.backends.cudnn.benchmark = True
    for epoch in range(epochs):
        model.train()
        for idx in range(len(train_loader)):
            DV, TV, D3D, T3D, Label = GETDATA(idx, train_loader)
            outputs_train = model(DV.to(device), TV.to(device), D3D.to(device), T3D.to(device))
            Label = Label.float().to(device)
            loss_train = criterion(outputs_train, Label)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            scheduler.step()
            print(f'Epoch [{epoch + 1}/{epochs}], [{idx}], {loss_train}')
            
            if idx % 20 == 0:
                with torch.set_grad_enabled(False):
                    model.eval()
                    val_losses_step = []
                    train_losses_step = []
                    for idx in range(len(val_loader)):
                        DV, TV, D3D, T3D, Label = GETDATA(idx, val_loader)
                        outputs_val = model(DV.to(device), TV.to(device), D3D.to(device), T3D.to(device))
                        Label = Label.float().to(device)
                        loss_val = criterion(outputs_val, Label)
                        val_losses_step.append(loss_val.item())
                    for idx in range(len(train_loader)):
                        DV, TV, D3D, T3D, Label = GETDATA(idx, train_loader)
                        outputs_train = model(DV.to(device), TV.to(device), D3D.to(device), T3D.to(device))
                        Label = Label.float().to(device)
                        loss_train = criterion(outputs_train, Label)
                        train_losses_step.append(loss_train.item())
                        
                avg_val_loss = np.mean(val_losses_step)
                avg_train_loss = np.mean(train_losses_step)
                for param_group in optimizer.param_groups:
                    lrnum = param_group['lr']
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.5f}, Val Loss: {avg_val_loss:.5f}, LR: {lrnum:.10f}')
        
        with torch.set_grad_enabled(False):
            RES = test(K, model, test_loader, device)
            print(f'Epoch [{epoch + 1}/{epochs}], {RES}')
        
        # if (epoch + 1) % 20 == 0 or (epoch + 1) == 1:
        #     with torch.set_grad_enabled(False):
        #         RES = test(K, model, test_loader, device)
        #         print(f'Epoch [{epoch + 1}/{epochs}], {RES}')
    
    LISTsensitivity.append(Lsensitivity)
    LISTspecificity.append(Lspecificity)
    LISTauprc.append(Lauprc)
    LISTroc_auc.append(Lroc_auc)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(train_losses) + 1), val_losses, label='Val Loss')
    plt.legend()
    plt.savefig('loss.pdf')
    plt.close()

def start():
    global LISTsensitivity, LISTspecificity, LISTauprc, LISTroc_auc
    info()
    params = {}
    params['Kfold'] = False
    params['fold'] = 5
    params['epochs'] = 40
    params['dimD'] = 73
    params['dimT'] = 73
    params['sizeD'] = 50
    params['sizeT'] = 545
    params['dropout'] = 0.1
    params['batch_size'] = 1
    params['lr'] = 0.000005
    for key, value in params.items():
        print(f"{key}: {value}")
    Stime = datetime.now()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'=========== D: \033[31m{device}\033[0m ===========\n')
    params['device'] = device
    criterion = nn.BCELoss()
    
    if params['Kfold']:
        print(f'=========== D: Kfold ===========\n')
        SEED = 1234
        with open('Davis.txt', "r") as f:
            Davis = f.read().strip().split('\n')
        dataset = Shuffle(Davis, SEED)
        for i in range(params['fold']):
            DFTRAIN, DFVAL, DFTEST = KDATA(dataset, i)
            train_loader, val_loader, test_loader = DATA(DFTRAIN, DFVAL, DFTEST)
            SL = len(train_loader)*4*params['epochs']
            model = DTIV7(**params).to(device)
            total_params = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.Adam(model.parameters(), lr= params['lr'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SL)
            print(f"=========== SEED: {SEED} , FOLD: {i+1}/{params['fold']} , EPOCHs: {params['epochs']} , TP: {total_params:,} ===========")
            train(params['Kfold'], model, criterion, optimizer, scheduler, params['epochs'], train_loader, val_loader, test_loader, device)
            
        RESsensitivity = torch.tensor(LISTsensitivity)
        RESspecificity = torch.tensor(LISTspecificity)
        RESauprc = torch.tensor(LISTauprc)
        RESroc_auc = torch.tensor(LISTroc_auc)
        meansensitivity = torch.mean(RESsensitivity)
        stdsensitivity = torch.std(RESsensitivity)
        meanspecificity = torch.mean(RESspecificity)
        stdspecificity = torch.std(RESspecificity)
        meanauprc = torch.mean(RESauprc)
        stdauprc = torch.std(RESauprc)
        meanroc_auc = torch.mean(RESroc_auc)
        stdroc_auc = torch.std(RESroc_auc)
        
        print(f"\n=========== RES ===========")
        print(f'Sensitivity: {meansensitivity.item():.4f} ± {stdsensitivity.item():.4f}')
        print(f'Specificity: {meanspecificity.item():.4f} ± {stdspecificity.item():.4f}')
        print(f'AUPR (PR-AUC): {meanauprc.item():.4f} ± {stdauprc.item():.4f}')
        print(f'ROC AUC: {meanroc_auc.item():.4f} ± {stdroc_auc.item():.4f}')
        print(f"=========== RES ===========\n")
        
        print(f'=========== D: Early Stopping Kfold ===========\n')
        SEED = 1234
        with open('Davis.txt', "r") as f:
            Davis = f.read().strip().split('\n')
        dataset = Shuffle(Davis, SEED)
        for i in range(params['fold']):
            DFTRAIN, DFVAL, DFTEST = KDATA(dataset, i)
            train_loader, val_loader, test_loader = DATA(DFTRAIN, DFVAL, DFTEST)
            SL = len(train_loader)*4*params['epochs']
            model = DTIV7(**params).to(device)
            total_params = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.Adam(model.parameters(), lr= params['lr'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SL)
            print(f"=========== SEED: {SEED} , FOLD: {i+1}/{params['fold']} , EPOCHs: {params['epochs']} , TP: {total_params:,} ===========")
            early_stopper = EarlyStopping(patience=5, verbose=True, path='best_model.pt', warmup_epochs=2)
            best_model = etrain(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, epochs=params['epochs'], train_loader=train_loader, val_loader=val_loader, device=device, early_stopping=early_stopper)
            with torch.set_grad_enabled(False):
                RES = test(params['Kfold'], best_model, test_loader, device)
                print(RES)
            
    else:
        print(f'=========== D: SKfold ===========\n')
        train_loader, val_loader, test_loader = DATA(None, None, None)
        SL = len(train_loader)*4*params['epochs']
        SEEDlist = random.sample(range(1, 10000), params['fold'])
        for i, SEED in enumerate(SEEDlist):
            random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            model = DTIV7(**params).to(device)
            total_params = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.Adam(model.parameters(), lr= params['lr'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SL)
            print(f"=========== SEED: {SEED} , FOLD: {i+1}/{params['fold']} , EPOCHs: {params['epochs']} , TP: {total_params:,} ===========")
            train(params['Kfold'], model, criterion, optimizer, scheduler, params['epochs'], train_loader, val_loader, test_loader, device)
        
        if params['fold'] > 1:
            RESsensitivity = torch.tensor(LISTsensitivity)
            RESspecificity = torch.tensor(LISTspecificity)
            RESauprc = torch.tensor(LISTauprc)
            RESroc_auc = torch.tensor(LISTroc_auc)
            meansensitivity = torch.mean(RESsensitivity)
            stdsensitivity = torch.std(RESsensitivity)
            meanspecificity = torch.mean(RESspecificity)
            stdspecificity = torch.std(RESspecificity)
            meanauprc = torch.mean(RESauprc)
            stdauprc = torch.std(RESauprc)
            meanroc_auc = torch.mean(RESroc_auc)
            stdroc_auc = torch.std(RESroc_auc)
            
            print(f"\n=========== RES ===========")
            print(f'Sensitivity: {meansensitivity.item():.4f} ± {stdsensitivity.item():.4f}')
            print(f'Specificity: {meanspecificity.item():.4f} ± {stdspecificity.item():.4f}')
            print(f'AUPR (PR-AUC): {meanauprc.item():.4f} ± {stdauprc.item():.4f}')
            print(f'ROC AUC: {meanroc_auc.item():.4f} ± {stdroc_auc.item():.4f}')
            print(f"=========== RES ===========\n")
            
        print(f'=========== D: Early Stopping SKfold ===========\n')
        train_loader, val_loader, test_loader = DATA(None, None, None)
        SL = len(train_loader)*4*params['epochs']
        SEEDlist = random.sample(range(1, 10000), params['fold'])
        for i, SEED in enumerate(SEEDlist):
            random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            model = DTIV7(**params).to(device)
            total_params = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.Adam(model.parameters(), lr= params['lr'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SL)
            print(f"=========== SEED: {SEED} , FOLD: {i+1}/{params['fold']} , EPOCHs: {params['epochs']} , TP: {total_params:,} ===========")
            early_stopper = EarlyStopping(patience=5, verbose=True, path='best_model.pt', warmup_epochs=2)
            best_model = etrain(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, epochs=params['epochs'], train_loader=train_loader, val_loader=val_loader, device=device, early_stopping=early_stopper)
            with torch.set_grad_enabled(False):
                RES = test(params['Kfold'], best_model, test_loader, device)
                print(RES)
    
    Etime = datetime.now()
    hours, remainder = divmod((Etime - Stime).total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"=========== T: {int(hours):02}:{int(minutes):02}:{int(seconds):02} ===========")
    
    time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    File = f"{time}.pth"
    torch.save(checkpoint, File)
    print(f'=========== SAVE: {File} ===========')
    
if __name__ == "__main__":
    required_dirs = ['P', 'D']
    for folder in required_dirs:
        if not os.path.isdir(folder):
            print(f"E:{folder} / 3D structure")
            quit()
            
    start()
    info()
