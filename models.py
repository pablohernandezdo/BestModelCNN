import torch.nn as nn
import torch.nn.functional as F


class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv5 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv6 = nn.Conv1d(128, 256, 3, padding=1, stride=2)
        self.conv7 = nn.Conv1d(256, 512, 3, padding=1, stride=1)
        self.conv8 = nn.Conv1d(512, 1024, 3, padding=1, stride=2)
        self.l1 = nn.Linear(1024, 512)
        self.l2 = nn.Linear(512, 1)
        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)
        self.p3 = nn.MaxPool1d(5)
        self.p4 = nn.MaxPool1d(5)
        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn6 = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(1024)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = wave.view(-1, 1, 6000)
        wave = self.bn1(F.relu(self.conv1(wave)))
        wave = self.bn2(F.relu(self.conv2(wave)))
        wave = self.p1(wave)
        wave = self.bn3(F.relu(self.conv3(wave)))
        wave = self.bn4(F.relu(self.conv4(wave)))
        wave = self.p2(wave)
        wave = self.bn5(F.relu(self.conv5(wave)))
        wave = self.bn6(F.relu(self.conv6(wave)))
        wave = self.p3(wave)
        wave = self.bn7(F.relu(self.conv7(wave)))
        wave = self.bn8(F.relu(self.conv8(wave)))
        wave = self.p4(wave)
        wave = wave.squeeze()
        wave = F.relu(self.l1(wave))
        wave = self.l2(wave)
        return self.sigmoid(wave)


class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 16, 3, padding=1, stride=2)
        self.conv5 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv6 = nn.Conv1d(32, 32, 3, padding=1, stride=2)
        self.conv7 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv8 = nn.Conv1d(64, 64, 3, padding=1, stride=2)
        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)
        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)
        self.p3 = nn.MaxPool1d(5)
        self.p4 = nn.MaxPool1d(5)
        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(16)
        self.bn5 = nn.BatchNorm1d(32)
        self.bn6 = nn.BatchNorm1d(32)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(64)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = wave.view(-1, 1, 6000)
        wave = self.bn1(F.relu(self.conv1(wave)))
        wave = self.bn2(F.relu(self.conv2(wave)))
        wave = self.p1(wave)
        wave = self.bn3(F.relu(self.conv3(wave)))
        wave = self.bn4(F.relu(self.conv4(wave)))
        wave = self.p2(wave)
        wave = self.bn5(F.relu(self.conv5(wave)))
        wave = self.bn6(F.relu(self.conv6(wave)))
        wave = self.p3(wave)
        wave = self.bn7(F.relu(self.conv7(wave)))
        wave = self.bn8(F.relu(self.conv8(wave)))
        wave = self.p4(wave)
        wave = wave.squeeze()
        wave = F.relu(self.l1(wave))
        wave = self.l2(wave)
        return self.sigmoid(wave)


class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)
        self.conv5 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv6 = nn.Conv1d(64, 128, 3, padding=1, stride=2)
        self.conv7 = nn.Conv1d(128, 256, 3, padding=1, stride=1)
        self.conv8 = nn.Conv1d(256, 512, 3, padding=1, stride=2)
        self.l1 = nn.Linear(512, 256)
        self.l2 = nn.Linear(256, 1)
        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)
        self.p3 = nn.MaxPool1d(5)
        self.p4 = nn.MaxPool1d(5)
        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(128)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(512)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = wave.view(-1, 1, 6000)
        wave = self.bn1(F.relu(self.conv1(wave)))
        wave = self.bn2(F.relu(self.conv2(wave)))
        wave = self.p1(wave)
        wave = self.bn3(F.relu(self.conv3(wave)))
        wave = self.bn4(F.relu(self.conv4(wave)))
        wave = self.p2(wave)
        wave = self.bn5(F.relu(self.conv5(wave)))
        wave = self.bn6(F.relu(self.conv6(wave)))
        wave = self.p3(wave)
        wave = self.bn7(F.relu(self.conv7(wave)))
        wave = self.bn8(F.relu(self.conv8(wave)))
        wave = self.p4(wave)
        wave = wave.squeeze()
        wave = F.relu(self.l1(wave))
        wave = self.l2(wave)
        return self.sigmoid(wave)
