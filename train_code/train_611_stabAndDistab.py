import utils
import time
import argparse
import numpy as np
from dataset import Dataset, collate_fn
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from models import gnn
import pandas as pd
from sklearn import metrics

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset
def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


if __name__ == "__main__":

    now = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    print(s)

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help="Random seed", type=int, default=42)
    parser.add_argument('--weight_decay', help="Weight decay (L2 loss on parameters)", type=float, default=5e-5)
    parser.add_argument("--train_data_fpath", help="file path of data", type=str, default='../data/ome_last_mutation_pdb')
    parser.add_argument("--train_wild_pdb", help="file path of wild_pdb", type=str, default='../data/ome_last_wild_pdb')
    parser.add_argument("--test_data_fpath", help="file path of data", type=str,  default='../data/testdata/ome_last_mutation_pdb')
    parser.add_argument("--test_wild_pdb", help="file path of wild_pdb", type=str, default='../data/testdata/ome_last_wild_pdb')
    parser.add_argument("--batch_size", help="batch_size", type=int, default=32)
    parser.add_argument("--num_workers", help="number of workers", type=int, default=0)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.0001)
    parser.add_argument("--d_layer", help="dimension of GNN layer", type=int, default=256)
    parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default=3)

    args = parser.parse_args()
    torch.cuda.empty_cache()
    lr = args.lr
    model = gnn(args)

    train = 'OK_S2647_4/OK_S2647'  #S611

    traindata = pd.read_excel('../data/' + train + '.xlsx')
    pdb_id = traindata['pdb_id']
    CHAIN = traindata['CHAIN']
    wild = traindata['wild_type']
    mutation = traindata['mutation']
    position = traindata['pdb_seq_position']
    ddG = traindata['ddG']
    keys = []
    pssm_features_list = []
    pssm_features = pd.read_csv('../data/' + train + 'features.csv')  # mutation_features_direct
    pssm_features = np.array(pssm_features)

    trans_wild_features_list = []
    trans_wild_features = pd.read_csv('../data/' + train + 'Wild_features.csv')  # pretained_wild_features
    trans_wild_features = np.array(trans_wild_features)

    trans_mut_features_list = []
    trans_mut_features = pd.read_csv('../data/' + train + 'Mut_features.csv')  # pretained_mutant_features特征
    trans_mut_features = np.array(trans_mut_features)

    #正向-direct
    train_zone = []
    for i in range(len(traindata)):
        temp = pdb_id[i] + str(CHAIN[i]) + "_" + wild[i] + str(position[i]) + mutation[i]
        keys.append(temp)
        temp = torch.tensor(pssm_features[i], dtype=torch.float32)
        pssm_features_list.append(temp)
        temp = torch.tensor(trans_wild_features[i], dtype=torch.float32)
        trans_wild_features_list.append(temp)
        temp = torch.tensor(trans_mut_features[i], dtype=torch.float32)
        trans_mut_features_list.append(temp)
        train_zone.append(1)  #1为direct，2为inverse

    #反向-inverse
    temp = ddG.values
    ddG = temp
    ddG = np.append(ddG, values=temp)
    pssm_features = pd.read_csv('../data/' + train + 'invfeatures.csv')  # mutation_features_inverse
    pssm_features = np.array(pssm_features)
    for i in range(len(traindata)):
        temp = pdb_id[i] + str(CHAIN[i]) + "_" + wild[i] + str(position[i]) + mutation[i]
        keys.append(temp)
        temp = torch.tensor(pssm_features[i], dtype=torch.float32)
        pssm_features_list.append(temp)
        temp = torch.tensor(trans_wild_features[i], dtype=torch.float32)
        trans_wild_features_list.append(temp)
        temp = torch.tensor(trans_mut_features[i], dtype=torch.float32)
        trans_mut_features_list.append(temp)
        train_zone.append(2)  #1为direct，2为inverse


    train_keys = list(zip(keys, pssm_features_list, trans_wild_features_list, trans_mut_features_list, ddG, train_zone))
    train_keys = shuffle_dataset(train_keys, 1234)
    train_dataset = Dataset(train_keys, args.train_data_fpath, args.train_wild_pdb)


    test = 'OK_S611/OK_S611_direct_1/OK_S611_direct'
    # test = 'OK_S611/OK_S611_reverse_1/OK_S611_reverse'   #注意，要改变 wild mutation位置
    testdata = pd.read_excel('../data/testdata/' + test + '.xlsx')
    pdb_id = testdata['pdb_id']
    CHAIN = testdata['CHAIN']
    wild = testdata['wild_type']
    mutation = testdata['mutation']
    position = testdata['pdb_seq_position']
    ddG = testdata['ddG']
    keys = []

    pssm_features_list = []
    pssm_features = pd.read_csv('../data/testdata/' + test + 'features.csv')
    pssm_features = np.array(pssm_features)


    trans_wild_features_list = []
    trans_wild_features = pd.read_csv('../data/testdata/' + test + 'Wild_features.csv')
    trans_wild_features = np.array(trans_wild_features)


    trans_mut_features_list = []
    trans_mut_features = pd.read_csv('../data/testdata/' + test + 'Mut_features.csv')
    trans_mut_features = np.array(trans_mut_features)


    test_ddg = []
    # 正向
    test_zone = []
    for i in range(len(testdata)):
        if ddG[i] >= 0:  #不稳定性
        # if ddG[i] < 0:  #稳定性
            temp = pdb_id[i] + str(CHAIN[i]) + "_" + wild[i] + str(position[i]) + mutation[i]
            keys.append(temp)
            temp = torch.tensor(pssm_features[i], dtype=torch.float32)
            pssm_features_list.append(temp)
            temp = torch.tensor(trans_wild_features[i], dtype=torch.float32)
            trans_wild_features_list.append(temp)
            temp = torch.tensor(trans_mut_features[i], dtype=torch.float32)
            trans_mut_features_list.append(temp)
            test_zone.append(1)
            test_ddg.append(ddG[i])

    #反向
    test = 'OK_S611/OK_S611_reverse_1/OK_S611_reverse'   #注意，要改变 wild mutation位置
    testdata = pd.read_excel('../data/testdata/' + test + '.xlsx')
    pdb_id = testdata['pdb_id']
    CHAIN = testdata['CHAIN']
    wild = testdata['mutation']
    mutation = testdata['wild_type']
    # wild = testdata['wild_type']
    # mutation = testdata['mutation']
    position = testdata['pdb_seq_position']
    ddG2 = -testdata['ddG']

    pssm_features = pd.read_csv('../data/testdata/' + test + 'invfeatures.csv')
    pssm_features = np.array(pssm_features)
    for i in range(len(testdata)):
        if ddG2[i] <= 0:  # 不稳定性
        # if ddG2[i] > 0: #稳定性
            temp = pdb_id[i] + str(CHAIN[i]) + "_" + wild[i] + str(position[i]) + mutation[i]
            keys.append(temp)
            temp = torch.tensor(pssm_features[i], dtype=torch.float32)
            pssm_features_list.append(temp)
            temp = torch.tensor(trans_wild_features[i], dtype=torch.float32)
            trans_wild_features_list.append(temp)
            temp = torch.tensor(trans_mut_features[i], dtype=torch.float32)
            trans_mut_features_list.append(temp)
            test_zone.append(2)  #1为direct，2为inverse
            test_ddg.append(ddG2[i])

    test_keys = list(zip(keys, pssm_features_list, trans_wild_features_list, trans_mut_features_list, test_ddg, test_zone))
    test_keys = shuffle_dataset(test_keys, 1234)
    test_dataset = Dataset(test_keys, args.test_data_fpath, args.test_wild_pdb)  # 去除ddG输入

    train_dataloader = DataLoader(train_dataset, args.batch_size, \
                                  shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, args.batch_size, \
                                 shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = utils.initialize_model(model, device)


    def heteroscedastic_loss(true, mean, log_var):
        precision = torch.exp(-log_var)
        return torch.mean(torch.sum(precision * (true - mean) ** 2 + log_var, 1), 0)
    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    PEARSON = []
    RMSE = []
    MAE = []
    LABELS = []
    PRED = []

    for epochs in range(300):
        list1_test = []
        list2_test = []
        list1_train = []
        list2_train = []
        train_losses = []
        test_losses = []
        list_log_var = []
        st = time.time()
        model.train()

        for i_batch, sample in enumerate(train_dataloader):
            model.zero_grad()
            H1, H2, labels, features,wild_features, mut_features = sample
            labels = torch.Tensor(labels)
            H1, H2, labels = H1.to(device), H2.to(device), labels.to(device)
            wild_features = wild_features.to(device)
            mut_features = mut_features.to(device)
            features = features.to(device)

            pred = model.train_model((H1, H2, features, wild_features, mut_features))
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.data.cpu().numpy())
            pred = pred.data.cpu().numpy()
            labels = labels.data.cpu().numpy()
            list1_train = np.append(list1_train, labels)
            list2_train = np.append(list2_train, pred)
        model.eval()


        for i_batch, sample in enumerate(test_dataloader):
            model.zero_grad()
            H1, H2, labels, features, wild_features, mut_features = sample
            labels = torch.Tensor(labels)
            H1, H2, labels = H1.to(device), H2.to(device), labels.to(device)

            wild_features = wild_features.to(device)
            mut_features = mut_features.to(device)
            features = features.to(device)
            pred = model.test_model((H1, H2, features, wild_features, mut_features))
            loss = loss_fn(pred, labels)
            test_losses.append(loss.data.cpu().numpy())
            labels = labels.data.cpu().numpy()
            pred = pred.data.cpu().numpy()
            list1_test = np.append(list1_test, labels)
            list2_test = np.append(list2_test, pred)

        et = time.time()

        rp_train = np.corrcoef(list2_train, list1_train)[0, 1]
        rp_test = np.corrcoef(list2_test, list1_test)[0, 1]
        test_losses = np.mean(np.array(test_losses))
        train_losses = np.mean(np.array(train_losses))
        x = np.array(list1_test).reshape(-1, 1)
        y = np.array(list2_test).reshape(-1, 1)
        end = time.time()
        rmse = np.sqrt(((y - x) ** 2).mean())
        mae = metrics.mean_absolute_error(y, x)

        PEARSON.append(rp_test)
        RMSE.append(rmse)
        MAE.append(mae)
        LABELS.append(list1_test)
        PRED.append(list2_test)

        print('epochs  train_losses   test_losses     pcc_train        pcc_test        rmse        mae       time ')
        print("%s   \t%.5f     \t%.5f    \t%.5f     \t%.5f  \t%.5f   \t%.5f   \t%.5f"
              % (epochs, train_losses, test_losses, rp_train, rp_test, rmse, mae, et - st))


        if epochs > 30:
            max_pr = np.array(PEARSON)
            max_r = np.array(RMSE)
            max_mae = np.array(MAE)
            index = max_pr.argsort()[-1:][::-1]

            max_pr = max_pr[index].mean()
            max_r = max_r[index].mean()
            max_mae = max_mae[index].mean()
            print('epochs   PEARSON      RMSE        MAE')
            print("%s   \t%.5f  \t%.5f  \t%.5f"
                  % (epochs, max_pr, max_r, max_mae))

        if epochs == 299:
            # name = 'S2648-S611-two-two-stable' #训练集-测试集-正向或反向-只有>0或只有<0或完整数据
            name = 'S2648-S611-two-two-destable' #训练集-测试集-正向或反向-只有>0或只有<0或完整数据
            torch.save(model, "../results2/" + name + ".pkl")
            dataframe_test = pd.DataFrame({'label': LABELS, 'pred': PRED, 'PEARSON': PEARSON, 'RMSE': RMSE, 'MAE':MAE})
            dataframe_test.to_csv(r"../results2/" + name + ".csv",sep=',')