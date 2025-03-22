import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import os
import numpy as np
from torch_geometric.data import HeteroData
from typing import Callable, Dict, List, Optional
import pandas as pd
from vingat.metrics import ScoreMetricHandler
from vingat.metrics import MetricsHandler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import pytz
from vingat.loader import core_file_loader
from vingat.loss import XENDCGLoss
from vingat.loss import LossItem


def now():
    return datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S')


def evaluate_model(
    model: nn.Module,
    data: HeteroData,
    device: torch.device,
    freq_tensor: dict,
    desc: str = "",
):
    model.eval()

    os.environ['TORCH_USE_CUDA_DSA'] = '1'

    with torch.no_grad():
        mhandler = MetricsHandler(device=device, threshold=0.5)
        shandler = ScoreMetricHandler(device=device)

        data = data.to(device)
        out, _ = model(data)

        # 評価用のエッジラベルとエッジインデックスを取得
        edge_label_index = data['user', 'buys', 'item'].edge_label_index

        # ユーザーとレシピの埋め込みを取得
        user_embeddings = out['user'].x
        recipe_embeddings = out['item'].x

        # userのindexを取得
        unique_user_ids = edge_label_index[0].unique()

        # 各ユーザーごとにループ
        for user_id in tqdm(unique_user_ids.cpu().numpy(), desc=desc):
            # 現在のユーザーに対する正例ペアを取得
            user_pos_indices = edge_label_index[1][edge_label_index[0] == user_id]
            num_pos_samples = len(user_pos_indices)    # 正例ペアの数を取得
            if num_pos_samples == 0:
                continue

            num_neg_samples = 500
            user_neg_indices, negative_edge_index = negative_sampling_with_popularity(
                user_id=user_id,
                user_pos_indices=user_pos_indices,
                freq_tensor=freq_tensor,
                num_neg_samples=num_neg_samples,
                device=device
            )

            # 正例と負例のユーザーとレシピの埋め込みを作成
            pos_user_embed = user_embeddings[user_id].expand(num_pos_samples, -1)
            pos_recipe_embed = recipe_embeddings[user_pos_indices]
            neg_user_embed = user_embeddings[user_id].expand(num_neg_samples, -1)
            neg_recipe_embed = recipe_embeddings[user_neg_indices]

            # 正例と負例のスコアを計算
            pos_scores = model.predict(pos_user_embed, pos_recipe_embed).squeeze(dim=1)
            neg_scores = model.predict(neg_user_embed, neg_recipe_embed).squeeze(dim=1)

            # スコアの統計量を更新
            shandler.update(pos_scores, neg_scores)
            mhandler.update(
                probas=torch.cat([pos_scores, neg_scores]),
                targets=torch.cat([
                    torch.ones_like(pos_scores, device=device),
                    torch.zeros_like(neg_scores, device=device)
                ]),
                user_indices=torch.full((len(pos_scores) + len(neg_scores),),
                                        user_id, device=device)
            )
        shandler.compute()
        mhandler.compute()

    return shandler, mhandler


def save_model(model: nn.Module,  save_directory: str, filename: str):
    os.makedirs(save_directory, exist_ok=True)
    torch.save(model.state_dict(), f"{save_directory}/{filename}.pth")


def calculate_statistics(data):
    """
    与えられた形式のデータを、項目ごとに最小、最大、平均、標準偏差を出算したDataFrameに変換する関数

    Args:
        data (list): 辞書のリスト形式のデータ

    Returns:
        pandas.DataFrame: 統計量をまとめたDataFrame
    """
    def min_max(x, axis=None):
        min = x.min(axis=axis, keepdims=True)
        max = x.max(axis=axis, keepdims=True)
        result = (x-min)/(max-min+1e-8)
        return result

    # 項目名を取得
    items = list(data[0].keys())

    # 統計量を格納する辞書
    statistics = {}
    for item in items:
        values = [d[item] for d in data]
        values_minmax = min_max(np.array(values))
        hist, bins = np.histogram(values_minmax,
                                  range=(0, 1),
                                  bins=np.linspace(0, 1, 6))
        statistics[item] = {
            'min': np.min(values),
            'max': np.max(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'count': len(values),
            **{f"bin_{round(v, 1)}_rate": h/len(values) for h, v in zip(hist, bins)}
        }

    # DataFrameに変換
    df = pd.DataFrame(statistics).T

    return df


def create_batch_predictions_targets(
    pos_scores: torch.Tensor,
    pos_user_ids: torch.Tensor,
    neg_scores: torch.Tensor,
    neg_user_ids: torch.Tensor
):
    user_scores: Dict[int, torch.Tensor] = {}
    device = pos_scores.device

    # 正例を追加
    for uid, score in zip(pos_user_ids.tolist(), pos_scores.tolist()):
        user_scores.setdefault(uid, []).append((score, 1.0))

    # 負例を追加
    for uid, score in zip(neg_user_ids.tolist(), neg_scores.tolist()):
        user_scores.setdefault(uid, []).append((score, 0.0))

    # 最大のアイテム数を計算
    max_items = max(len(scores) for scores in user_scores.values())

    predictions, targets = [], []

    # 各ユーザーごとにpaddingしてpredictions, targetsを生成
    for uid in sorted(user_scores.keys()):
        scores_labels = user_scores[uid]
        scores = torch.tensor([sl[0] for sl in scores_labels])
        labels = torch.tensor([sl[1] for sl in scores_labels])

        padding_length = max_items - len(scores)

        predictions.append(torch.cat([scores, torch.full((padding_length,), -1.0)]))
        labels = torch.tensor([sl[1] for sl in scores_labels])
        padding_length = max_items - len(labels)

        targets.append(torch.cat([
            labels,
            torch.zeros(padding_length)
        ]))

    return torch.stack(predictions).to(device), torch.stack(targets).to(device)


def train_one_epoch(
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    criterion: Callable,
    max_grad_norm: float,
    recommendation_loss_weight: float,
):
    """
    DataLoaderによってネガティブサンプリングされている時用
    """
    model.to(device)

    loss_histories: Dict[str, List[torch.Tensor]] = {}
    node_mean = []
    mhandler = MetricsHandler(device=device, threshold=0.5)
    shandler = ScoreMetricHandler(device=device)
    top_k = 10
    xe_loss = XENDCGLoss(k=top_k)

    model.train()
    counter = 0
    for batch_data in tqdm(train_loader, desc="[Train]"):
        counter += 1
        optimizer.zero_grad()
        batch_data = batch_data.to(device)

        out, loss_entories = model(batch_data)

        main_loss_rate = 1.0

        # エッジのラベルとエッジインデックスを取得
        edge_label_index = batch_data['user', 'buys', 'item'].edge_label_index

        # ユーザーとレシピの埋め込みを取得
        user_embeddings = out['user'].x
        user_ids = out["user"].id
        recipe_embeddings = out['item'].x

        # 正例と負例のマスクを取得
        pos_mask = batch_data['user', 'buys', 'item'].edge_label == 1
        neg_mask = batch_data['user', 'buys', 'item'].edge_label == 0

        # エッジインデックスからノードの埋め込みを取得
        user_embed = user_embeddings[edge_label_index[0]]
        recipe_embed = recipe_embeddings[edge_label_index[1]]

        # 正例のスコアを計算
        pos_user_embed = user_embed[pos_mask]
        pos_recipe_embed = recipe_embed[pos_mask]
        pos_user_ids = user_ids[edge_label_index[0]][pos_mask]
        pos_scores = model.predict(pos_user_embed, pos_recipe_embed).squeeze()

        # 負例のスコアを計算
        neg_user_embed = user_embed[neg_mask]
        neg_recipe_embed = recipe_embed[neg_mask]
        neg_user_ids = user_ids[edge_label_index[0]][neg_mask]
        neg_scores = model.predict(neg_user_embed, neg_recipe_embed).squeeze()

        if len(pos_scores) != len(neg_scores):
            print(f"pos: {len(pos_scores)}, neg: {len(neg_scores)}")
            raise ValueError("Positive scores and Negative scores are not same length")

        # 損失の計算
        main_loss = criterion(pos_scores, neg_scores, model.parameters())
        main_loss_item = LossItem(name="main_loss", loss=main_loss,
                                  weight=recommendation_loss_weight)
        loss_entories.append(main_loss_item)

        pos_nan_count = torch.isnan(pos_scores).sum().item()
        neg_nan_count = torch.isnan(neg_scores).sum().item()
        pos_count = pos_scores.numel()
        neg_count = neg_scores.numel()
        if pos_nan_count > 0 or neg_nan_count > 0:
            print("pos nan: ", pos_nan_count, "/", pos_count)
            print("neg nan: ", neg_nan_count, "/", neg_count)

        if counter % 1 == 0:
            all_scores = torch.cat([pos_scores, neg_scores])
            all_targets = torch.cat([
                torch.ones_like(pos_scores),
                torch.zeros_like(neg_scores)
            ])
            all_user_ids = torch.cat([pos_user_ids, neg_user_ids])
            unique_users, inverse_indices, counts = torch.unique(all_user_ids,
                                                                 return_inverse=True,
                                                                 return_counts=True)
            valid_mask = counts[inverse_indices] >= 10
            filtered_scores = all_scores[valid_mask]
            filtered_targets = all_targets[valid_mask]
            filtered_users = all_user_ids[valid_mask]

            if filtered_scores.numel() > 0:
                xe_loss_result = xe_loss(filtered_scores, filtered_targets, filtered_users)
                loss_entories.append(LossItem(name="xe_loss", loss=xe_loss_result,
                                     weight=main_loss_rate))

        loss = sum(loss_item.loss * loss_item.weight for loss_item in loss_entories)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        for loss_item in loss_entories:
            loss_histories.setdefault(loss_item.name, []).append(
                loss_item.loss.item()
            )

        mhandler.update(
            probas=torch.cat([pos_scores, neg_scores]),
            targets=torch.cat([
                torch.ones_like(pos_scores, device=device),
                torch.zeros_like(neg_scores, device=device)
            ]),
            user_indices=torch.cat([
                edge_label_index[0][pos_mask],
                edge_label_index[0][neg_mask]
            ])
        )
        shandler.update(pos_scores, neg_scores)

        # check
        node_mean.append({
            key: val.mean().mean().item()
            for key, val in out.x_dict.items()
        })

    node_stats = calculate_statistics(node_mean)

    return (
        model,
        loss_histories,
        node_stats,
        mhandler,
        shandler,
    )


def show_model_parameters(model: nn.Module):
    model.eval()
    _res = []
    for name, param in model.named_parameters():
        _res.append({"name": name, "param": param.norm().item()})
    res = pd.DataFrame(_res)
    mean = res["param"].mean()
    std = res["param"].std()
    upper = mean + 2 * std
    lower = mean - 2 * std
    outer_val = res[(res['param'] > upper) | (res['param'] < lower)]
    na_val = res.loc[res["param"].isna()]

    print("Model Parameters Describe: ")
    print(res.describe())

    if len(outer_val) > 0:
        print("Outer Value of Model Params (2σ): ")
        print(outer_val)

    if len(na_val) > 0:
        print("Nan Value of Model Params: ")
        print(na_val)


def train_func(
    train_loader,
    val_data,
    model,
    optimizer,
    scheduler,
    criterion,
    epochs,
    device,
    wbLogger: Callable,
    wbTagger: Callable,
    wbScatter: Callable,
    directory_path: str,
    project_name: str,
    experiment_name: str,
    popularities: dict,
    recommendation_loss_weight: float,
    patience=4,
    validation_interval=5,
    max_grad_norm=1.0,
    pca_cols=["user", "item", "intention", "taste", "image"],
):
    os.environ['TORCH_USE_CUDA_DSA'] = '1'

    best_val_metric = 0    # 現時点での最良のバリデーションメトリクスを初期化
    patience_counter = 0    # Early Stoppingのカウンターを初期化
    best_model_epoch = 0

    save_dir = f"{directory_path}/models/{project_name}/{experiment_name}"

    for epoch in range(1, epochs+1):

        print("\n======================\n", f"Epoch {epoch}/{epochs}", now())
        model, loss_histories, node_stats, mhandler, shandler = train_one_epoch(
            model=model,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            max_grad_norm=max_grad_norm,
            recommendation_loss_weight=recommendation_loss_weight
            # freq_tensor=popularities  # For Negative Sampling
        )

        print("[Train] Node Statics: ")
        print(node_stats)

        tr_loss = {
            f"train-loss/{k}": np.mean(v)
            for k, v in loss_histories.items()
        }
        print("\n[Train] Loss: ", "\n", tr_loss)
        wbLogger(
            data=tr_loss,
            step=epoch
        )

        print("\n[Train] Metrics: ")
        print(mhandler.log(prefix="train-handler", num_round=4))
        wbLogger(data=mhandler.log("train-handler"), step=epoch)

        print("\n[Train] Score Statics: ")
        print(shandler.log(prefix="train-score-statics", num_round=4))
        wbLogger(data=shandler.log(prefix="train-score-statics"), step=epoch)

        print("\n[Train] Model Parameters: ")
        show_model_parameters(model)

        # Valid
        if epoch % validation_interval == 0:

            print("\nValidation -------------------")

            """
            _df = visualize_node_pca(batch_data,
                                     pca_cols,
                                     f"after_training. Epoch: {epoch}/{epochs}")
            wbScatter(_df, epoch, title=f"after training (epoch: {epoch})")
            """

            score_statics, v_mhandler = evaluate_model(
                model=model,
                data=val_data,
                device=device,
                freq_tensor=popularities,
                desc=f"[Valid] Epoch {epoch}/{epochs}"
            )

            val_metrics = {
                "val/last_lr": scheduler.get_last_lr()[0]
            }

            wbLogger(
                data=val_metrics,
                step=epoch
            )
            print(now(), "Score Statics: ")
            print(score_statics.log(prefix="val-score-statics", num_round=4))
            wbLogger(data=score_statics.log(prefix="val-score-statics"), step=epoch)

            print(now(), "handler Result: ")
            vmhlog = v_mhandler.log(prefix="val-handler", num_round=4)
            print(vmhlog)
            wbLogger(data=vmhlog, step=epoch)

            save_model(model, save_dir, f"model_{epoch}")

            v_base_metric = v_mhandler.compute().get("AUROC")

            # Early Stoppingの判定（バリデーションの精度または他のメトリクスで判定）
            if v_base_metric > best_val_metric:
                best_val_metric = v_base_metric    # 最良のバリデーションメトリクスを更新
                patience_counter = 0    # 改善が見られたためカウンターをリセット
                best_model_epoch = epoch
            else:
                patience_counter += 1    # 改善がなければカウンターを増やす

            # patienceを超えた場合にEarly Stoppingを実行
            if patience_counter >= patience:
                print(now(), f"エポック{epoch}でEarly Stoppingを実行します。")
                wbTagger("early_stopped")
                model.load_state_dict(torch.load(f"{save_dir}/model_{best_model_epoch}.pth"))
                break

            print(now(), f"patience_counter: {patience_counter} / {patience}")

        scheduler.step()

    if epochs <= epoch:
        wbTagger("epoch_completed")
    else:
        wbTagger(f"under_{(epoch // 10 + 1) * 10}_epochs")

    return model


def get_item_popularity(
    device: torch.device,
    item_lencoder: LabelEncoder,
    directory_path: str,
    rating_threshold: float = 0.0,
    r=0.7
) -> Dict:
    _, core_train_rating, _, _ = core_file_loader(directory_path, rating_threshold)
    core_train_rating["recipe_id"] = item_lencoder.transform(core_train_rating["recipe_id"])
    res = core_train_rating.groupby("recipe_id").count()["rating"].to_dict()
    tensor = np.zeros(len(item_lencoder.classes_), dtype=np.float32)
    for k, v in res.items():
        tensor[k] = v ** r
    return torch.tensor(tensor, device=device)


def negative_sampling_with_popularity(
    user_id: int,
    user_pos_indices: torch.Tensor,      # shape=(num_pos_samples,)
    freq_tensor: torch.Tensor,           # shape=(item_count,), freq^r (GPU上)
    num_neg_samples: int,
    device: torch.device = torch.device("cuda"),
    candidate_items: Optional[torch.Tensor] = None,
):
    """
    Args:
        user_id (int): ユーザID
        user_pos_indices (torch.Tensor): shape=(num_pos_samples,)
            このユーザが正例として持つアイテムのインデックス (GPU上)
        freq_tensor (torch.Tensor): shape=(item_count,)
            グローバルに freq^r を格納した Tensor (GPU上)
        num_neg_samples (int): サンプリングしたい負例数
        device (torch.device): 実行デバイス
        candidate_items (torch.Tensor, optional): shape=(subset_count,)
            サブセットとしてサンプリング対象にするアイテムID一覧 (GPU上).
            指定がなければ全アイテム (freq_tensor.shape[0]) が対象。

    Returns:
        chosen_indices (torch.Tensor): shape=(<= num_neg_samples,)
            サンプリングされた負例アイテムのID
        negative_edge_index (torch.Tensor): shape=(2, <= num_neg_samples)
            [[user_id, ..., user_id],
             [chosen_item_1, ..., chosen_item_N]]
    """
    if candidate_items is None:
        # freq_tensor をコピー（clone）し、正例に対応する index を 0 にして除外
        # （cloneしないと他のユーザの処理に影響が出る恐れがある）
        weights = freq_tensor.clone()
        weights[user_pos_indices] = 0.0

        sum_w = weights.sum()
        if sum_w <= 0:
            # まず正例以外を取得
            item_count = freq_tensor.shape[0]
            all_indices = torch.arange(item_count, device=device)
            mask = torch.ones(item_count, dtype=torch.bool, device=device)
            mask[user_pos_indices] = False
            candidate_indices = all_indices[mask]
            if candidate_indices.shape[0] < num_neg_samples:
                # replace=Trueで許容するか、サンプル数を減らすかは要件次第
                chosen_indices = candidate_indices
            else:
                chosen_indices = candidate_indices[
                    torch.randperm(candidate_indices.shape[0])[:num_neg_samples]
                ]
        else:
            # 確率分布として正規化してサンプリング
            probs = weights / sum_w
            chosen_indices = torch.multinomial(probs, num_neg_samples, replacement=False)
    else:
        # freq_tensor をコピーし、まず全て 0 にする (サブセット以外はサンプリングしない)
        # もしくは candidate_items だけを抜き出したウェイトを作るアプローチもOK
        weights = torch.zeros_like(freq_tensor, device=device)
        # candidate_items の要素だけ freq_tensor の値をコピー
        weights[candidate_items] = freq_tensor[candidate_items]

        # 正例アイテムは weight=0 で除外
        weights[user_pos_indices] = 0.0

        sum_w = weights.sum()
        if sum_w <= 0:
            # 一様ランダムフォールバック (candidate_items から pos を除いたもの)
            mask = torch.ones_like(candidate_items, dtype=torch.bool, device=device)
            # pos に含まれる要素を除外
            pos_set = set(user_pos_indices.tolist())
            for i, cid in enumerate(candidate_items):
                if cid.item() in pos_set:
                    mask[i] = False
            valid_candidates = candidate_items[mask]

            if valid_candidates.shape[0] == 0:
                # 極端ケース: サブセット内が全て正例
                chosen_indices = torch.empty(0, dtype=torch.long, device=device)
            elif valid_candidates.shape[0] < num_neg_samples:
                chosen_indices = valid_candidates
            else:
                chosen_indices = valid_candidates[
                    torch.randperm(valid_candidates.shape[0], device=device)[:num_neg_samples]
                ]
        else:
            probs = weights / sum_w
            chosen_indices = torch.multinomial(probs, num_neg_samples, replacement=False)

    # negative_edge_index もGPU上で作成
    negative_edge_index = torch.stack([
        torch.full((num_neg_samples,), user_id, dtype=torch.long, device=device),
        chosen_indices
    ], dim=0)

    return chosen_indices, negative_edge_index
