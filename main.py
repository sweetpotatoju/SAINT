import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import KFold
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

from augmentations import embed_data_mask
from pretraining import SAINT_pretrain
from pretrainmodel import SAINT
import argparse
from torch.utils.data import DataLoader,TensorDataset
import wandb


from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error

pd_list = []
pf_list = []
bal_list = []
fir_list = []


def classifier_eval(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print('혼동행렬 : ', cm)
    PD = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    print('PD : ', PD)
    PF = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    print('PF : ', PF)
    balance = 1 - (((0 - PF) * (0 - PF) + (1 - PD) * (1 - PD)) / 2)
    print('balance : ', balance)
    FI = (cm[1, 1] + cm[0, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
    FIR = (PD - FI) / PD
    print('FIR : ', FIR)

    return PD, PF, balance, FIR




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
    parser.add_argument('--batchsize', default=64, type=int)
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





# CSV 파일 경로를 지정
csv_file_path = "EQ.csv"

# CSV 파일을 데이터프레임으로 읽어오기
df = pd.read_csv(csv_file_path)
# print(df)

# 데이터프레임에서 특징(X)과 목표 변수(y) 추출
X = df.drop(columns=['class'])
y = df['class']  # 'target' 열을 목표 변수로 사용'

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# K-겹 교차 검증을 설정합니다
k = 5 # K 값 (원하는 폴드 수) 설정
kf = KFold(n_splits=k, shuffle=True, random_state=42)

scaler = MinMaxScaler()
X_test_Nomalized = scaler.fit_transform(X_test)

opt = get_args()
# K-겹 교차 검증 수행
for train_index, val_index in kf.split(X_train):
    X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

    # 전처리
    # Min-Max 정규화 수행(o)
    X_fold_train_normalized = scaler.fit_transform(X_fold_train)
    X_fold_val_normalized = scaler.transform(X_fold_val)

    # SMOTE를 사용하여 학습 데이터 오버샘플링
    smote = SMOTE(random_state=42)
    X_fold_train_resampled, y_fold_train_resampled = smote.fit_resample(X_fold_train_normalized, y_fold_train)

    X_train_tensor = torch.tensor(X_fold_train_resampled)
    y_train_tensor = torch.tensor(y_fold_train_resampled)

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True, num_workers=0, pin_memory=False)
    vision_dset = opt.vision_dset

    X_test_tensor = torch.tensor(X_test_Nomalized)
    y_test_array = y_test.to_numpy()  # 시리즈를 넘파이 배열로 변환
    y_test_tensor = torch.tensor(y_test_array)

    test_ds = TensorDataset(X_test_tensor, y_test_tensor)
    testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=True, num_workers=0, pin_memory=False)


    cat_dims = []
    con_idxs = 61
    cat_idxs = 10

    model = SAINT(
        categories = tuple(cat_dims),
        num_continuous=con_idxs,
        dim=opt.embedding_size,
        dim_out=1,
        depth=opt.transformer_depth,
        heads=opt.attention_heads,
        attn_dropout=opt.attention_dropout,
        ff_dropout=opt.ff_dropout,
        mlp_hidden_mults=(4, 2),
        cont_embeddings=opt.cont_embeddings,
        attentiontype=opt.attentiontype,
        final_mlp_style=opt.final_mlp_style,
        y_dim= 2
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}.")

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr)

    print('Training begins now.')
    for epoch in range(opt.epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            x_input, y_label = data  # 수정: 데이터 튜플에서 입력과 레이블 추출

            x_cont = x_input.to(device)  # 모든 입력 데이터를 연속형 변수로 간주하여 하나의 텐서로 처리합니다.
            x_cont = x_cont.float()  # 데이터 유형을 float32로 변환

            cat_mask, con_mask = None, None  # cat_mask와 con_mask가 없는 경우

            x_cont_enc = embed_data_mask(x_cont, model, vision_dset)
            reps = model.transformer(x_cont_enc)
            # select only the representations corresponding to CLS token and apply mlp on it in the next step to get the predictions.
            y_reps = reps[:, 0, :]

            y_outs = model.mlpfory(y_reps)

            criterion = nn.CrossEntropyLoss().to(device)
            loss = criterion(y_outs, y_label.squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            print(({'epoch': epoch ,'train_epoch_loss': running_loss,
            'loss': loss.item()}))

    m = nn.Softmax(dim=1)
    # 모델을 평가 모드로 설정합니다.
    model.eval()

    y_pred = torch.empty(0).to(device)  # y_pred 초기화
    prob = torch.empty(0).to(device)

    # torch.no_grad() 블록 내에서 평가를 수행합니다.
    with torch.no_grad():
        for data in testloader:  # 테스트 데이터를 사용하여 반복합니다.
            x_input, y_label = data

            # 입력 데이터를 디바이스로 이동하고 float32로 변환합니다.
            x_cont = x_input.to(device)
            x_cont = x_cont.float()

            # 데이터를 모델에 전달하여 예측을 생성합니다.
            x_cont_enc = embed_data_mask(x_cont, model, vision_dset)
            reps = model.transformer(x_cont_enc)
            y_reps = reps[:, 0, :]
            y_outs = model.mlpfory(y_reps)

            prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
            print(prob)





print('avg_PD: {}'.format((sum(pd_list) / len(pd_list))))
print('avg_PF: {}'.format((sum(pf_list) / len(pf_list))))
print('avg_balance: {}'.format((sum(bal_list) / len(bal_list))))
print('avg_FIR: {}'.format((sum(fir_list) / len(fir_list))))


