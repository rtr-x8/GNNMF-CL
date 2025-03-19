import numpy as np
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.retrieval import (
    RetrievalRecall,
    RetrievalPrecision,
    RetrievalNormalizedDCG,
    # RetrievalMAP,
    # RetrievalMRR,
)
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryRecall,
    BinaryF1Score,
    BinaryConfusionMatrix,
    BinaryAUROC
)
from typing import List


def ndcg_at_k(r: np.ndarray, k: int):
    r = np.asfarray(r)[:k]
    if r.size:
        dcg = np.sum(r / np.log2(np.arange(2, r.size + 2)))
        idcg = np.sum(np.ones_like(r) / np.log2(np.arange(2, r.size + 2)))
        return dcg / idcg
    return 0.


class ScoreMetricHandler():
    def __init__(
        self,
        device: torch.device
    ):
        self.pos_scores: List[torch.Tensor] = []
        self.neg_scores: List[torch.Tensor] = []
        self.is_calculated = False
        self.result = None
        self.device = device

    def update(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor):
        self.pos_scores.append(pos_scores.clone().detach().to(self.device))
        self.neg_scores.append(neg_scores.clone().detach().to(self.device))
        self.is_calculated = False

    def compute(self):
        if not self.is_calculated:
            pos_scores = torch.cat(self.pos_scores)
            neg_scores = torch.cat(self.neg_scores)
            self.pos_mean = pos_scores.mean().item()
            self.pos_min = pos_scores.min().item()
            self.pos_max = pos_scores.max().item()
            self.pos_std = pos_scores.std().item()
            self.neg_mean = neg_scores.mean().item()
            self.neg_min = neg_scores.min().item()
            self.neg_max = neg_scores.max().item()
            self.neg_std = neg_scores.std().item()
            self.diff_mean = self.pos_mean - self.neg_mean

            self.is_calculated = True

        return {
            "pos_mean": self.pos_mean,
            "pos_min": self.pos_min,
            "pos_max": self.pos_max,
            "pos_std": self.pos_std,
            "neg_mean": self.neg_mean,
            "neg_min": self.neg_min,
            "neg_max": self.neg_max,
            "neg_std": self.neg_std,
            "diff_mean": self.diff_mean,
        }

    def log(self, prefix: str = "", separator: str = "/", num_round: int = 8):
        return {
            f"{prefix}{separator}{k}": round(v, num_round)
            for k, v in self.compute().items()
        }


class MetricsHandler():
    """
    入力を繰り返し受け取り、最終的な計算を行う。
    """
    def __init__(self, device, threshold: float = 0.5):
        self.threshold = threshold
        self.device = device
        self.reset()

    def reset(self):
        self.probas = []
        self.targets = []
        self.user_indices = []
        self.is_calculated = False

    def update(self,
               probas: torch.Tensor,
               targets: torch.Tensor,
               user_indices: torch.Tensor):
        probas = probas.clone().detach().to(self.device)
        targets = targets.clone().detach().to(self.device)
        user_indices = user_indices.to(self.device)
        self.probas.append(probas)
        self.targets.append(targets)
        self.user_indices.append(user_indices)

    def compute(self):
        if not self.is_calculated:
            all_probas = torch.cat(self.probas)
            all_targets = torch.cat(self.targets)
            all_user_indices = torch.cat(self.user_indices)

            collection = MetricCollection({
                "recall@10": RetrievalRecall(top_k=10),
                # "recall@20": RetrievalRecall(top_k=20),
                "precision@10": RetrievalPrecision(top_k=10, adaptive_k=True),
                # "precision@20": RetrievalPrecision(top_k=20, adaptive_k=True),
                "ndcg@10": RetrievalNormalizedDCG(top_k=10),
                # "ndcg@20": RetrievalNormalizedDCG(top_k=20),
                # "map@10": RetrievalMAP(top_k=10),
                # "map@20": RetrievalMAP(top_k=20),
                # "mrr@10": RetrievalMRR(top_k=10),
                # "mrr@20": RetrievalMRR(top_k=20),
                "accuracy": BinaryAccuracy(threshold=self.threshold),
                "recall": BinaryRecall(threshold=self.threshold),
                "f1": BinaryF1Score(threshold=self.threshold),
                "cm": BinaryConfusionMatrix(threshold=self.threshold),
                "AUROC": BinaryAUROC(),
            }).to(self.device)

            result = collection(all_probas, all_targets, indexes=all_user_indices)
            result["tn"] = result["cm"][0][0]
            result["fp"] = result["cm"][0][1]
            result["fn"] = result["cm"][1][0]
            result["tp"] = result["cm"][1][1]
            del result["cm"]

            self.result = result

            self.is_calculated = True
        return self.result

    def log(self, prefix: str = "", separator: str = "/", num_round: int = 8):
        return {
            f"{prefix}{separator}{k}": round(v.item(), num_round)
            for k, v in self.compute().items()
        }


class FastNDCG(nn.Module):
    def __init__(self, top_k):
        super().__init__()
        self.k = top_k

    def forward(self, predictions, targets, indexes):
        device = predictions.device
        k = self.k

        if predictions.numel() == 0:
            return torch.tensor(0.0, device=device)

        # ユーザーごとにデータをソート
        sorted_indexes, indices = indexes.sort()
        predictions = predictions[indices]
        targets = targets[indices]

        # ユーザーごとのユニークなインデックスとカウントを取得
        unique_users, counts = torch.unique_consecutive(sorted_indexes, return_counts=True)

        # ユーザーを30%サンプリング
        num_sampled_users = max(1, int(len(unique_users) * 0.3))
        sampled_users = unique_users[torch.randperm(len(unique_users))[:num_sampled_users]]
        sampled_users_set = set(sampled_users.tolist())

        # 各ユーザーごとにDCGとIDCGを計算
        ndcg_list = []
        start = 0
        for user, count in zip(unique_users, counts):
            end = start + count
            if user.item() not in sampled_users_set:
                start = end
                continue

            preds_user = predictions[start:end]
            targets_user = targets[start:end]
            start = end

            if targets_user.numel() == 0:
                continue

            k_user = min(k, preds_user.size(0))

            # 予測値に基づいてtargetsをソート
            sorted_indices = torch.argsort(preds_user, descending=True)
            sorted_targets_k = targets_user[sorted_indices][:k_user]

            discounts = torch.log2(torch.arange(2, k_user + 2, device=device))
            dcg = (sorted_targets_k / discounts).sum()

            # 理想的なtargetsは実際のtargetsを降順ソートしたもの
            ideal_targets_k = torch.sort(targets_user, descending=True)[0][:k_user]
            idcg = (ideal_targets_k / discounts).sum()

            if idcg == 0:
                ndcg = torch.tensor(0.0, device=device)
            else:
                ndcg = dcg / idcg

            ndcg_list.append(ndcg)

        if len(ndcg_list) == 0:
            return torch.tensor(0.0, device=device)

        ndcg_scores = torch.stack(ndcg_list)
        return ndcg_scores.mean()
