import pandas as pd
import os
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.io as io
from concurrent.futures import ThreadPoolExecutor
import torch.nn as nn
import numpy as np
import torch
from tqdm.notebook import tqdm
from typing import Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib


use_nutritions = ["niacin", "fiber", "sugars", "sodium", "carbohydrates",
                  "vitaminB6", "calories", "thiamin", "fat", "folate",
                  "caloriesFromFat", "calcium", "magnesium", "iron",
                  "cholesterol", "protein", "vitaminA", "potassium",
                  "saturatedFat", "vitaminC"]


def parse_nutrient_json(json_dict):
    res = {}
    for un in use_nutritions:
        res.update({un: eval(json_dict).get(un).get("amount")})
    return res


def text_to_embedding(
    directory_path: str,
    data: pd.DataFrame,
    name: str,
    cols: list,
    break_flag=False
):
    result = pd.DataFrame([], columns=cols)
    sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    _c = 0
    for i in tqdm(data.index, desc=f"[{name} embedding...]"):
        try:
            emb = sbert.encode(data.loc[i])
            emb = pd.DataFrame([emb[0]], index=[i], columns=cols)
            result = pd.concat([result, emb], axis=0)
        except ValueError:
            print(f"{name} in {i} has Error, Skipped.")
            continue
        if _c > 5 and break_flag:
            break
        _c += 1
    result.to_csv(f"{directory_path}/{name}_embeddings.csv")


def load_user_embeddings(
    directory_path: str,
    user_ids: list,
    col_range: int = 384
) -> pd.DataFrame:
    name = "users"
    cols = [f"e_{i}" for i in range(col_range)]
    file_path = f"{directory_path}/{name}_embeddings.csv"
    originarl_df = pd.DataFrame(user_ids, columns=["name"], index=user_ids)
    originarl_df = originarl_df.astype(str)
    if not os.path.isfile(file_path):
        text_to_embedding(
            directory_path,
            originarl_df,
            name,
            cols
        )
    df = pd.read_csv(file_path, index_col=0)
    df = df.fillna(0)
    return df


def load_recipe_cooking_directions_embeddings(
    directory_path: str,
    originarl_df: pd.DataFrame,
    col_range: int = 384
) -> pd.DataFrame:
    name = "recipe_cooking_directions"
    cols = [f"e_{i}" for i in range(col_range)]
    file_path = f"{directory_path}/{name}_embeddings.csv"
    if not os.path.isfile(file_path):
        text_to_embedding(
            directory_path,
            originarl_df,
            name,
            cols
        )
    df = pd.read_csv(file_path, index_col=0)
    df = df.fillna(0)
    return df


def load_recipe_image_vlm_caption_embeddings(
    directory_path: str,
    originarl_df: pd.DataFrame,
    col_range: int = 384
) -> pd.DataFrame:
    name = "recipe_image_vlm_caption"
    cols = [f"e_{i}" for i in range(col_range)]
    file_path = f"{directory_path}/{name}_embeddings.csv"
    if not os.path.isfile(file_path):
        text_to_embedding(
            directory_path,
            originarl_df,
            name,
            cols
        )
    df = pd.read_csv(file_path, index_col=0)
    df = df.fillna(0)
    return df


def preprocess_rating_data(data: pd.DataFrame):
    # data["interaction"] = data["rating"].apply(lambda x: 1 if x > rating_threshold else 0)
    return data


def core_file_loader(
    directory_path: str,
    rating_threshold: float = 3.5
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(f"{directory_path}/core-data-train_rating.csv")
    test = pd.read_csv(f"{directory_path}/core-data-test_rating.csv")
    val = pd.read_csv(f"{directory_path}/core-data-valid_rating.csv")
    print("一旦使わないけど：", rating_threshold)
    return (
        pd.read_csv(f"{directory_path}/core-data_recipe.csv", index_col=0),
        preprocess_rating_data(train),
        preprocess_rating_data(test),
        preprocess_rating_data(val)
    )


def load_recipe_nutrients(directory_path: str, originarl_df: pd.DataFrame):
    """
    アイテム - 栄養素の一次保存データ
    """

    file_path = f"{directory_path}/recipe_nutrients.csv"
    if not os.path.isfile(file_path):

        __recipes = originarl_df.copy()
        __recipes[use_nutritions] = 0.0
        __recipes["nutritions"] = __recipes["nutritions"].to_dict()

        for idx, row in tqdm(__recipes.iterrows(), total=__recipes.shape[0]):
            nutri_text = row["nutritions"]
            nutri_val: Dict[int, str] = parse_nutrient_json(nutri_text)
            for key, val in nutri_val.items():
                __recipes.loc[idx, key] = val

        __recipes.to_csv(file_path)

    recipe_nutrients = pd.read_csv(file_path, index_col=0)
    print("recipe_nutrients is loaded")
    return recipe_nutrients


def load_ingredients(directory_path: str, originarl_df: pd.DataFrame):
    """
    食材の一時保存データ。
    """
    if not os.path.isfile(f"{directory_path}/ingredients.csv"):
        __recipes = originarl_df.copy()
        ingredients = set([""])
        for idx, row in tqdm(__recipes.iterrows(), total=__recipes.shape[0]):
            for ing in row["ingredients"].split("^"):
                ingredients.add(ing)
        ingredients.remove("")
        res = pd.DataFrame(ingredients, columns=["name"])
        res.index.name = "ingredient_id"
        res.to_csv(f"{directory_path}/ingredients.csv")

    df = pd.read_csv(f"{directory_path}/ingredients.csv", index_col=0)
    print("ingredients is loaded")
    return df


def load_recipe_ingredients(directory_path: str, originarl_df: pd.DataFrame):
    """
    アイテム - 食材の一時保存データ
    """
    file_path = f"{directory_path}/recipe_ingredients.csv"
    if not os.path.isfile(file_path):

        __recipes = originarl_df.copy()
        recipe_ingredients = pd.DataFrame([], columns=["recipe_id",
                                                       "ingredient_id"])
        ingredients = load_ingredients(directory_path, originarl_df)
        all_ings = ingredients.values.reshape(-1)
        for recipe_id, recipe_row in tqdm(
            __recipes.iterrows(),
            total=__recipes.shape[0]
        ):
            for recip_ing in recipe_row["ingredients"].split("^"):
                if recip_ing in all_ings:
                    recipe_ingredients = pd.concat([
                        recipe_ingredients,
                        pd.DataFrame(
                            [[
                                recipe_id,
                                np.where(all_ings == recip_ing)[0][0]
                            ]],
                            columns=["recipe_id", "ingredient_id"])]
                        )
        recipe_ingredients.to_csv(file_path)

    recipe_ingredients = pd.read_csv(file_path, index_col=0)
    print("recipe_ingredients is loaded")
    return recipe_ingredients


def load_recipe_cooking_directions(
    directory_path: str,
    originarl_df: pd.DataFrame
) -> pd.DataFrame:
    """
    レシピ名、調理工程、レシピ画像の一時保存データ
    """
    file_path = f"{directory_path}/recipe_cooking_directions.csv"
    key = file_path
    if not os.path.isfile(file_path):

        __recipes = originarl_df.copy()
        __recipes[key] = __recipes[key].to_dict()

        recipe_cooking_directions = pd.DataFrame(
            [],
            index=__recipes.index,
            columns=["direction"]
        )
        for i in tqdm(__recipes.index.to_list(), total=__recipes.shape[0]):
            txt = __recipes.at[i, key]
            txt = eval(txt).get('directions')
            txts = txt.split("\n")
            txts = [t for t in txts if len(t) > 10]
            txt = "\n".join(txts)
            recipe_cooking_directions.at[i, "direction"] = txt
        recipe_cooking_directions.to_csv(file_path)

    recipe_cooking_directions = pd.read_csv(file_path, index_col=0)
    print("recipe_cooking_directions is loaded")
    return recipe_cooking_directions


def load_ingredients_with_embeddings(
    directory_path: str,
    originarl_df: pd.DataFrame,
    col_range: int = 384
):
    """
    食材の埋込データ
    """
    name = "ingredients"
    file_path = f"{directory_path}/{name}_embeddings.csv"
    if not os.path.isfile(file_path):
        text_to_embedding(
            directory_path,
            originarl_df,
            name,
            [f"e_{i}" for i in range(col_range)]
        )
    ingredients_with_embeddings = pd.read_csv(file_path, index_col=0)
    print("ingredients_with_embeddings is loaded")
    return ingredients_with_embeddings


def load_recipe_image_embeddings_ft(
    directory_path: str,
    originarl_df: pd.DataFrame,
    device
) -> pd.DataFrame:
    file_path = f"{directory_path}/recipe_image_embeddings_ft.csv"
    recipe_image_embeddings = pd.read_csv(file_path, index_col=0)
    print("recipe_image_embeddings_ft is loaded")
    return recipe_image_embeddings


def load_recipe_image_embeddings(
    directory_path: str,
    originarl_df: pd.DataFrame,
    device
) -> pd.DataFrame:
    """
    レシピ画像の特徴データ
    """
    file_path = f"{directory_path}/recipe_image_embeddings.csv"
    if not os.path.isfile(file_path):
        cnn_model = models.resnet50(pretrained=True)
        cnn = nn.Sequential(
            # 最後の平均プーリング層の手前まで取得
            *(list(cnn_model.children())[:-2]),

            # 1x1畳み込み層でoutput_dim次元に
            nn.Conv2d(2048, 1024, kernel_size=1),

            # 出力を1x1にプールしてoutput_dim次元ベクトルに変換
            nn.AdaptiveAvgPool2d((1, 1))
        ).to(device)

        # 評価モードに設定
        cnn.eval()
        for param in cnn.parameters():
            param.requires_grad = False

        # 画像の前処理を定義
        image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        recipe_ingredients = load_recipe_ingredients(
            directory_path, originarl_df)
        recipe_image_embeddings = pd.DataFrame(
            [],
            index=recipe_ingredients["recipe_id"].unique(),
            columns=[f"e_{i}" for i in range(1024)])
        errors = []

        def process_image(recipe_id):
            image_path = f"{directory_path}/core-data-images/core-data-images"
            image_path = f"{image_path}/{recipe_id}.jpg"

            """個別の画像の読み込みと特徴量抽出を行う"""
            try:
                image = io.read_image(image_path).to(device)
                image = image_transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = cnn(image)

                # 特徴量をCPUに移動してフラット化
            except Exception as e:
                errors.append(recipe_id)
                print(e)
                return recipe_id, np.zeros(1024)

            return recipe_id, output.view(-1).cpu().numpy()

        # 画像をバッチ処理
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(tqdm(
                executor.map(
                    process_image,
                    recipe_image_embeddings.index.tolist()
                ),
                total=len(recipe_image_embeddings.index.tolist()),
                desc="Image Embedding gen"))
        for recipe_id, embedding in tqdm(results, desc="save..."):
            recipe_image_embeddings.loc[recipe_id] = embedding

        # 結果をCSVに保存
        recipe_image_embeddings.to_csv(file_path)

    recipe_image_embeddings = pd.read_csv(file_path, index_col=0)
    print("recipe_image_embeddings is loaded")
    return recipe_image_embeddings


def load_recipe_image_vlm_caption(directory_path: str,) -> pd.DataFrame:
    """
    レシピ画像のVLMキャプションデータ
    https://colab.research.google.com/drive/1Z0EV32iNRv5SeAovl1tM-1YkM7jZdcgT?usp=sharing
    """
    return pd.read_csv(f"{directory_path}/recipe_image_vlm_caption.csv",
                       index_col=0)


def batch_cosine_similarity_torch(embeddings: torch.Tensor, batch_size: int = 1000) -> torch.Tensor:
    device = embeddings.device
    n, d = embeddings.shape

    # (1) 正規化
    norms = embeddings.norm(dim=1, keepdim=True) + 1e-12
    embeddings_normed = embeddings / norms

    # (2) 結果用テンソル作成
    #     大きいのでGPUで持つとメモリ不足になりやすい点に注意
    cos_sim_matrix = torch.zeros((n, n), dtype=embeddings.dtype, device=device)

    # (3) バッチ処理
    for i_start in range(0, n, batch_size):
        i_end = min(i_start + batch_size, n)
        batch_i = embeddings_normed[i_start:i_end]  # shape: (bs_i, D)

        for j_start in range(0, n, batch_size):
            j_end = min(j_start + batch_size, n)
            batch_j = embeddings_normed[j_start:j_end]  # shape: (bs_j, D)

            sim_block = torch.matmul(batch_i, batch_j.transpose(0, 1))
            cos_sim_matrix[i_start:i_end, j_start:j_end] = sim_block

    return cos_sim_matrix


def get_top50_indices_above_threshold(
    sim_matrix: torch.Tensor,
    threshold: float = 0.7,
    top_k: int = 50
) -> torch.Tensor:
    """
    sim_matrix: (N x N) のコサイン類似度行列 (PyTorch Tensor)
    threshold : 類似度のしきい値 (0.7)
    top_k     : しきい値を超えるインデックスを何件抽出するか (50)

    戻り値:
      shape: (N x top_k)
      各行 i について、類似度が threshold を超える列インデックスを昇順で格納したテンソル
      ただし 50件に満たない場合は -1 で埋める
    """
    n = sim_matrix.size(0)

    # 返り値用テンソルを -1 で初期化
    top_indices = torch.full((n, top_k), -1, dtype=torch.long, device=sim_matrix.device)

    for i in tqdm(range(n)):
        # 行 i に対して、類似度が threshold を超える j を抽出
        row = sim_matrix[i]  # shape: (N, )
        indices = torch.where(row > threshold)[0]  # 類似度が0.7超のインデックス

        # indices = indices.sort().values
        row_values = row[indices]                 # しきい値を超えた要素の「実際の値」を取り出す
        _, sort_idx = row_values.sort(descending=True)
        indices = indices[sort_idx]              # 値が大きい順に並び替えたインデックス

        # 先頭 top_k 件を取り出し、残りは無視
        length = min(top_k, indices.size(0))
        top_indices[i, :length] = indices[:length]

    return top_indices


def load_alternative_ingredients(directory_path: str, originarl_df: pd.DataFrame, device):
    file_path = f"{directory_path}/alternative_ingredients.csv"
    if os.path.isfile(file_path):
        return pd.read_csv(file_path, index_col=0)

    # SBERTモデルのロード
    model = SentenceTransformer('all-mpnet-base-v2')

    # バッチサイズ
    batch_size = 300

    ingredients = load_ingredients(directory_path, originarl_df)

    # ingredientsのベクトル化
    ingredient_embeddings = []
    for i in tqdm(range(0, len(ingredients), batch_size)):
        batch_names = ingredients["name"][i:i + batch_size].values
        embeddings = model.encode(batch_names)
        ingredient_embeddings.extend(embeddings)

    ingredient_embeddings = np.array(ingredient_embeddings)

    ingredient_embeddings_torch = torch.tensor(ingredient_embeddings, device=device)

    # バッチ計算
    result_matrix_torch = batch_cosine_similarity_torch(ingredient_embeddings_torch,
                                                        batch_size=1000)

    # しきい値0.7・上位50件を取得
    selected_indices = get_top50_indices_above_threshold(result_matrix_torch, 0.5, 50)

    alternative_ingredients = []

    for idx, i in tqdm(enumerate(ingredients.index)):
        for sim_ing in selected_indices[idx]:
            _i = sim_ing.item()
            if _i == -1:
                continue
            if _i == i:
                continue
            if _i < i:
                continue
            alternative_ingredients.append({
                "ingredient_id": _i,
                "alternative_ingredient": i,
                "score": result_matrix_torch[_i, i].item()
            })
    result = pd.DataFrame(alternative_ingredients)
    result.to_csv(file_path)
    return result


def train_dataclustering(train_data: pd.DataFrame, n_cluster: int, path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = train_data.copy()
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # 6クラスタでKMeansクラスタリング
    kmeans = KMeans(n_clusters=6, n_init=n_cluster)
    df['cluster'] = kmeans.fit_predict(df_scaled)
    joblib.dump(kmeans, f'{path}/recipe_kmeanth_model.joblib')

    return df


def calculate_cluster(data: pd.DataFrame, path: str) -> pd.DataFrame:
    loaded_kmeans_model = joblib.load(f'{path}/recipe_kmeanth_model.joblib')

    # テストデータなど (N=200, 次元=12) でクラスタIDを予測
    test_cluster_ids = loaded_kmeans_model.predict(data)
    data["cluster"] = test_cluster_ids
    return data
