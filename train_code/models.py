import torch.nn.functional as F
from utils import *
N_atom_features = 30

class gnn(torch.nn.Module):
    def __init__(self, args):
        super(gnn, self).__init__()
        d_layer = args.d_layer
        n_FC_layer = args.n_FC_layer

        self.FC = nn.ModuleList([nn.Linear(2*(64 + 768) + 380, 512) if i == 0 else
                                 nn.Linear(256, 128) if i == n_FC_layer - 1 else
                                 nn.Linear(512, 256) for i in range(n_FC_layer)])
        self.linear_mu = nn.Linear(128, 1)
        self.embede = nn.Linear(N_atom_features, d_layer, bias=False)

        self.kernel = [4, 6, 8]
        self.conv = 64
        self.CNNs = nn.Sequential(
            nn.Conv1d(in_channels=d_layer, out_channels=self.conv, kernel_size=self.kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv , kernel_size=self.kernel[2]),
            nn.ReLU(),
        )
        self.sigmoid = nn.Sigmoid()

    def embede_graph(self, data):
        c_hs1, c_hs2, features, wild_features, mut_features = data
        c_hs1 = self.embede(c_hs1)
        c_hs1 = c_hs1.permute(0, 2, 1)
        c_hs2 = self.embede(c_hs2)
        c_hs2 = c_hs2.permute(0, 2, 1)
        c_hs1_conv = self.CNNs(c_hs1)
        c_hs2_conv = self.CNNs(c_hs2)
        c_hs1_atte = self.sigmoid(c_hs1_conv)
        c_hs2_atte = self.sigmoid(c_hs2_conv)
        c_hs1_conv = c_hs1_conv * 0.5 + c_hs1_conv * c_hs1_atte
        c_hs2_conv = c_hs2_conv * 0.5 + c_hs2_conv * c_hs2_atte
        c_hs1 = c_hs1_conv.permute(0, 2, 1).sum(1)
        c_hs2 = c_hs2_conv.permute(0, 2, 1).sum(1)
        return c_hs1, c_hs2


    def fully_connected(self, c_hs1):
        n1 = c_hs1.shape[0]
        fc_bn = nn.BatchNorm1d(n1)
        fc_bn.cuda()
        for k in range(len(self.FC)):
            if k < len(self.FC):
                c_hs1 = self.FC[k](c_hs1)
                c_hs1 = c_hs1.unsqueeze(0)
                c_hs1 = fc_bn(c_hs1)
                c_hs1 = c_hs1.squeeze(0)
                c_hs1 = F.leaky_relu(c_hs1)
        mean1 = self.linear_mu(c_hs1)
        return mean1


    def train_model(self, data):
        c_hs1, c_hs2,  features, wild_features, mut_features = data
        c_hs1, c_hs2 = self.embede_graph(data)
        features = features.squeeze(2)
        wild_features = wild_features.squeeze(2)
        mut_features = mut_features.squeeze(2)
        c_hs = torch.cat((c_hs1, wild_features), 1)
        c_hs = torch.cat((c_hs, c_hs2), 1)
        c_hs = torch.cat((c_hs, mut_features), 1)
        c_hs = torch.cat((c_hs, features), 1)
        predict = self.fully_connected(c_hs)
        predict = predict.view(-1)

        return predict

    def test_model(self, data):
        c_hs1, c_hs2,  features, wild_features, mut_features = data
        c_hs1, c_hs2 = self.embede_graph(data)
        features = features.squeeze(2)
        wild_features = wild_features.squeeze(2)
        mut_features = mut_features.squeeze(2)
        c_hs = torch.cat((c_hs1, wild_features), 1)
        c_hs = torch.cat((c_hs, c_hs2), 1)
        c_hs = torch.cat((c_hs, mut_features), 1)
        c_hs = torch.cat((c_hs, features), 1)
        predict = self.fully_connected(c_hs)
        predict = predict.view(-1)
        return predict



