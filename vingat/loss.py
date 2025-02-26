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


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_A, z_B):
        device = z_A.device
        N = z_A.size(0)
        z = torch.cat([z_A, z_B], dim=0).half()  # (2N, d) and convert to float16
        z = F.normalize(z, dim=-1)  # 正規化（余分な計算削減）

        # バッチ処理の導入
        batch_size = 1024  # バッチサイズを設定
        losses = []
        for start in range(0, 2 * N, batch_size):
            end = min(start + batch_size, 2 * N)
            z_batch = z[start:end]
            similarity_matrix = torch.mm(z_batch, z.transpose(0, 1)) / self.temperature
            labels = torch.cat([
                torch.arange(N, 2 * N, device=device),
                torch.arange(0, N, device=device)
            ]) % (2 * N - 1)
            mask = torch.eye(end - start, dtype=torch.bool, device=device)  # 自己相関を除く
            similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))  # 自分自身を無効化
            loss = F.cross_entropy(similarity_matrix, labels[start:end])
            losses.append(loss)

        return torch.mean(torch.stack(losses))
