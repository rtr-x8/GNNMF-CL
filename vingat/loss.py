import torch
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics.retrieval import RetrievalNormalizedDCG


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


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_A, z_B):
        device = z_A.device
        N = z_A.size(0)
        z = torch.cat([z_A, z_B], dim=0)  # (2N, d) and convert to float16
        # z = F.normalize(z, dim=-1)  # 正規化（余分な計算削減）

        # バッチ処理の導入
        batch_size = 1024  # バッチサイズを設定
        losses = []
        for start in range(0, 2 * N, batch_size):
            end = min(start + batch_size, 2 * N)
            z_batch = z[start:end]
            # z_batchのサイズ: (batch_size, dim)
            # zのサイズ: (2 * N, dim)
            similarity_matrix = torch.mm(z_batch, z.T) / self.temperature  # (batch_size, 2 * N)

            # マスクの作成
            mask = torch.zeros((z_batch.size(0), z.size(0)), dtype=torch.bool, device=device)
            mask[torch.arange(z_batch.size(0)), start + torch.arange(z_batch.size(0))] = True

            # 自分自身を無効化
            similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

            # ラベルの設定：正例のインデックスを設定
            labels = (start + torch.arange(z_batch.size(0), device=device) + N) % (2 * N)

            loss = F.cross_entropy(similarity_matrix, labels)
            losses.append(loss)

        return torch.mean(torch.stack(losses))  # 損失の平均を返す


class XENDCGLoss(nn.Module):
    def __init__(self, k):
        super(XENDCGLoss, self).__init__()
        self.k = k
        self.ndcg = RetrievalNormalizedDCG(top_k=k)

    def forward(self, predictions, targets, indexes):
        xe_loss = nn.functional.binary_cross_entropy(predictions, targets)
        ndcg_score = self.ndcg(predictions, targets, indexes=indexes)
        loss = xe_loss * (1 - ndcg_score)
        return loss
