import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import KFold
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
import argparse

from augmentations import embed_data_mask
from pretraining import SAINT_pretrain
from pretrainmodel import SAINT
import wandb


def calculate_metrics(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    PD = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    PF = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    balance = 1 - np.sqrt((1 - PD)**2 + PF**2) / np.sqrt(2)
    FI = (cm[1, 1] + cm[0, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
    FIR = (PD - FI) / PD
    return PD, PF, balance, FIR

# Function to evaluate classifier
def classifier_eval(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    PD, PF, balance, FIR = calculate_metrics(y_test, y_pred)
    pd_list.append(PD)
    pf_list.append(PF)
    bal_list.append(balance)
    fir_list.append(FIR)
    print('Confusion Matrix:', cm)
    print('Length of y_test:', len(y_test))
    print('Length of y_pred:', len(y_pred))
    print(f'PD: {PD}, PF: {PF}, Balance: {balance}, FIR: {FIR}')



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision_dset', action='store_true')
    parser.add_argument('--cont_embeddings', default='MLP', type=str, choices=['MLP', 'Noemb', 'pos_singleMLP'])
    parser.add_argument('--embedding_size', default=32, type=int)
    parser.add_argument('--transformer_depth', default=6, type=int)
    parser.add_argument('--attention_heads', default=8, type=int)
    parser.add_argument('--attention_dropout', default=0.1, type=float)
    parser.add_argument('--ff_dropout', default=0.1, type=float)
    parser.add_argument('--attentiontype', default='colrow', type=str,
                        choices=['col', 'colrow', 'row', 'justmlp', 'attn', 'attnmlp'])

    parser.add_argument('--optimizer', default='AdamW', type=str, choices=['AdamW', 'Adam', 'SGD'])
    parser.add_argument('--scheduler', default='cosine', type=str, choices=['cosine', 'linear'])

    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batchsize', default=32, type=int)
    parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
    parser.add_argument('--run_name', default='testrun', type=str)
    parser.add_argument('--set_seed', default=1, type=int)
    parser.add_argument('--dset_seed', default=5, type=int)
    parser.add_argument('--active_log', action='store_true')

    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pretrain_epochs', default=1, type=int)
    parser.add_argument('--pt_tasks', default=['contrastive', 'denoising'], type=str, nargs='*',
                        choices=['contrastive', 'contrastive_sim', 'denoising'])
    parser.add_argument('--pt_aug', default=[], type=str, nargs='*', choices=['mixup', 'cutmix'])
    parser.add_argument('--pt_aug_lam', default=0.1, type=float)
    parser.add_argument('--mixup_lam', default=0.3, type=float)

    parser.add_argument('--train_mask_prob', default=0, type=float)
    parser.add_argument('--mask_prob', default=0, type=float)

    parser.add_argument('--ssl_avail_y', default=0, type=int)
    parser.add_argument('--pt_projhead_style', default='diff', type=str, choices=['diff', 'same', 'nohead'])
    parser.add_argument('--nce_temp', default=0.7, type=float)

    parser.add_argument('--lam0', default=0.5, type=float)
    parser.add_argument('--lam1', default=10, type=float)
    parser.add_argument('--lam2', default=1, type=float)
    parser.add_argument('--lam3', default=10, type=float)
    parser.add_argument('--final_mlp_style', default='sep', type=str, choices=['common', 'sep'])

    opt = parser.parse_args()
    return opt




# CSV file path
csv_file_path = "EQ.csv"

# Read CSV file into a dataframe
df = pd.read_csv(csv_file_path)

# Extract features (X) and target variables (y) from the dataframe
X = df.drop(columns=['class'])
y = df['class']
print(X)

# Split the data into train, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# K-Fold Cross Validation
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Initialize lists to store metrics
pd_list = []
pf_list = []
bal_list = []
fir_list = []


# Min-Max Scaling
scaler = MinMaxScaler()
X_test_normalized = scaler.fit_transform(X_test)

# Get arguments
opt = get_args()

# K-Fold Cross Validation
for train_index, val_index in kf.split(X_train):
    X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

    # Preprocessing
    X_fold_train_normalized = scaler.fit_transform(X_fold_train)
    X_fold_val_normalized = scaler.transform(X_fold_val)

    # SMOTE for oversampling
    smote = SMOTE(random_state=42)
    X_fold_train_resampled, y_fold_train_resampled = smote.fit_resample(X_fold_train_normalized, y_fold_train)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_fold_train_resampled)
    y_train_tensor = torch.tensor(y_fold_train_resampled)

    # Create DataLoader
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True)
    vision_dset = opt.vision_dset

    # Convert test data to PyTorch tensors
    X_test_tensor = torch.tensor(X_test_normalized)
    y_test_array = y_test.to_numpy()
    y_test_tensor = torch.tensor(y_test_array)

    test_ds = TensorDataset(X_test_tensor, y_test_tensor)
    testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=True, num_workers=0, pin_memory=False)

    # Define model
    cat_dims = []
    con_idxs = 61
    cat_idxs = 10

    model = SAINT(
        categories=tuple(cat_dims),
        num_continuous=con_idxs,
        dim=61,
        dim_out=1,
        depth=opt.transformer_depth,
        heads=opt.attention_heads,
        attn_dropout=opt.attention_dropout,
        ff_dropout=opt.ff_dropout,
        mlp_hidden_mults=(4, 2),
        cont_embeddings=opt.cont_embeddings,
        attentiontype=opt.attentiontype,
        final_mlp_style=opt.final_mlp_style,
        y_dim=2
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}.")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr)

    print('학습을 시작합니다.')
    for epoch in range(opt.epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            x_cont = data[0].to(device).float()

            y_gts = data[1].to(device)

            # 연속형 데이터를 임베딩으로 변환합니다.
            x_cont_enc = embed_data_mask(x_cont, model, vision_dset)
            reps = model.transformer(x_cont_enc)
            y_reps = reps[:, 0, :]

            # 모델에 연속형 데이터를 전달하여 출력을 얻습니다.
            y_outs = model.mlpfory(y_reps)

            import torch.nn.functional as F
            # 소프트맥스 함수를 적용하여 확률로 변환합니다.
            y_probs = F.softmax(y_outs, dim=1)


            criterion = nn.CrossEntropyLoss().to(device)

            # 크로스 엔트로피 손실을 계산합니다.
            loss = criterion(y_probs, y_gts.squeeze())

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch % 5 == 0:
            print(f"Epoch [{epoch}/{opt.epochs}], Loss: {running_loss}")

    # 모델을 평가 모드로 변경
    model.eval()



    # 입력 데이터를 모델에 전달하여 예측값을 얻음
    with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                optimizer.zero_grad()
                x_cont = data[0].to(device).float()

                # 연속형 데이터를 임베딩으로 변환합니다.
                x_cont_enc = embed_data_mask(x_cont, model, vision_dset)
                reps = model.transformer(x_cont_enc)
                y_reps = reps[:, 0, :]

                y_outs = model.mlpfory(y_reps)

                # 소프트맥스 함수를 적용하여 확률로 변환합니다.
                y_probs = F.softmax(y_outs, dim=1)
                print(y_probs)




# # Calculating average metrics
# avg_PD = sum(pd_list) / len(pd_list) if len(pd_list) > 0 else 'No values in pd_list'
# avg_PF = sum(pf_list) / len(pf_list) if len(pf_list) > 0 else 'No values in pf_list'
# avg_balance = sum(bal_list) / len(bal_list) if len(bal_list) > 0 else 'No values in bal_list'
# avg_FIR = sum(fir_list) / len(fir_list) if len(fir_list) > 0 else 'No values in fir_list'
#
# # Print or use the average metrics as needed
# print('Average PD:', avg_PD)
# print('Average PF:', avg_PF)
# print('Average balance:', avg_balance)
# print('Average FIR:', avg_FIR)