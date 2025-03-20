from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd
from typing import Tuple, List
from torch_geometric.data import HeteroData


class ScalarPreprocess:
    def __init__(self, mapping: List[Tuple[str, str]]):
        """
        mapping_dict: Dict[str, str]
            ノード名と属性の対応を示す辞書
        """
        self.mapping = mapping
        self.standard_scaler = {
            map: StandardScaler()
            for map in mapping
        }

    def fit(self, data: HeteroData):
        for node, attr in self.mapping:
            self.standard_scaler[(node, attr)].fit(
                data[node][attr].cpu().numpy()
            )
        return self

    def transform(self, data: HeteroData):
        for node, attr in self.mapping:
            data[node][attr] = torch.tensor(
                self.standard_scaler[(node, attr)].transform(data[node][attr].cpu().numpy()),
                dtype=data[node][attr].dtype
            ).to(data[node][attr].device)
        return data


def filter_recipe_ingredient(
    recip_ing: pd.DataFrame,
    alternative_ing: pd.DataFrame,
    threshold: int
) -> pd.DataFrame:
    """
    レシピと食材のデータフレームを受け取り、代替食材の数が閾値を超えるレシピを返す
    """
    _key = 'ingredient_id'
    alternative = alternative_ing[alternative_ing["score"] > threshold]
    mapping_dict = dict(zip(alternative['alternative_ingredient'], alternative[_key]))
    recip_ing[_key] = recip_ing[_key].map(mapping_dict).fillna(recip_ing[_key]).astype(int)
    recip_ing = recip_ing.drop_duplicates(subset=['recipe_id', _key])
    return recip_ing


class NutrientStandardPreprocess():
    def __init__(self, use_nutrients):
        self.scalar = StandardScaler()
        self.use_nutrients = use_nutrients

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.scalar.fit_transform(df[self.use_nutrients])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.scalar.transform(df[self.use_nutrients])

    def transform_from_tensor(self, t: torch.Tensor) -> torch.Tensor:
        df = pd.DataFrame(t.cpu().numpy())
        return torch.tensor(df.to_numpy(), device=t.device)

    def do(
        self,
        train_recipe_ids: List,
        test_recipe_ids: List,
        val_recipe_ids: List,
        recipe_nutrients: pd.DataFrame,
    ):
        _rn = recipe_nutrients.copy()
        train = _rn.loc[_rn.index.isin(train_recipe_ids)]
        test = _rn.loc[_rn.index.isin(test_recipe_ids)]
        val = _rn.loc[_rn.index.isin(val_recipe_ids)]

        self.scalar.fit(train[self.use_nutrients])
        train = self.scalar.transform(train[self.use_nutrients])
        test = self.scalar.transform(test[self.use_nutrients])
        val = self.scalar.transform(val[self.use_nutrients])

        _rn.loc[train_recipe_ids, self.use_nutrients] = train
        _rn.loc[test_recipe_ids, self.use_nutrients] = test
        _rn.loc[val_recipe_ids, self.use_nutrients] = val

        return _rn
