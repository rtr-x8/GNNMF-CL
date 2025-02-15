import torch
import torch.nn.functional as F
import torch.nn as nn


class BPRLoss(nn.Module):
    def __init__(self, reg_lambda=0.01):
        super(BPRLoss, self).__init__()
        self.reg_lambda = reg_lambda  # 正則化パラメータ

    def forward(self, pos_scores, neg_scores, model_params):
        differences = pos_scores - neg_scores

        # BPRのペアワイズ損失計算
        loss = -torch.mean(torch.log(torch.sigmoid(differences)))

        # L2正則化項
        reg_loss = 0
        for param in model_params:
            reg_loss += torch.norm(param, p=2)

        return loss + self.reg_lambda * reg_loss


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, reg_lambda=0.01):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.reg_lambda = reg_lambda  # 正則化パラメータ

    def forward(self, pos_scores, neg_scores, model_params):
        differences = pos_scores - neg_scores
        loss = F.binary_cross_entropy_with_logits(differences, torch.ones_like(differences))

        reg_loss = 0
        for param in model_params:
            reg_loss += torch.norm(param, p=2)

        return loss + self.reg_lambda * reg_loss


class SeparationLoss(nn.Module):
    def __init__(self, reg_lambda=0.01):
        super(SeparationLoss, self).__init__()
        self.reg_lambda = reg_lambda

    def forward(self, feature1, feature2):
        cosine_similarity = F.cosine_similarity(feature1, feature2, dim=1)

        # 類似度の平均を計算
        loss = torch.mean(cosine_similarity)

        # 分離損失（類似度を最小化）
        return self.reg_lambda * loss


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, z_A, z_B):
        N = z_A.size(0)
        # 類似度行列の計算
        similarity_matrix = torch.matmul(z_A, z_B.T) / self.temperature  # (N, N)
        # 正のペアは対角線上にある
        labels = torch.arange(N).to(z_A.device)
        loss = self.cross_entropy(similarity_matrix, labels)
        return loss
