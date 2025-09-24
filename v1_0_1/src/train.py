import pickle
from pathlib import Path

# import optuna.integration.lightgbm as lgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.metrics import log_loss

DATA_DIR = Path("..", "data")
INPUT_DIR = DATA_DIR / "02_feature"
OUPUT_DIR = DATA_DIR / "03_train"
OUPUT_DIR.mkdir(exist_ok=True, parents=True)

#リメイク#11 4:45
class Trainer:
    def __init__(
        self,
        features_filepath: Path = INPUT_DIR / "features.csv",
        config_filepath: Path = "config.yaml",
        output_dir: Path = OUPUT_DIR
    ):
        self.features = pd.read_csv(features_filepath, sep="\t")
        with open(config_filepath, "r") as f:
            config = yaml.safe_load(f)
            self.feature_cols = config["features"]
            self.params = config["params"]
        self.output_dir = output_dir

    def create_dataset(self, valid_start_date: str, test_start_date: str):
        """
        test_start_dateをYYYY-MM-DD形式で指定すると、
        その日付以降のデータをテストデータに、
        それより前のデータを学習データに分割する関数。
        """
        # 目的変数
        self.features["target"] = (self.features["rank"] == 1).astype(int)
        # 学習データとテストデータに分割
        self.train_df = self.features.query("date < @valid_start_date")
        self.valid_df = self.features.query(
            "date >= @valid_start_date and date < @test_start_date"
        )
        self.test_df = self.features.query("date >= @test_start_date")

    def train(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
        importance_filename: str,
        model_filename: str
    ) -> pd.DataFrame:
        # データセットの作成
        lgb_train = lgb.Dataset(train_df[self.feature_cols], train_df["target"])
        lgb_vaild = lgb.Dataset(
            valid_df[self.feature_cols], valid_df["target"], reference=lgb_train
        )
        # 学習の実行
        model = lgb.train(
            params=self.params,
            train_set=lgb_train,
            num_boost_round=10000,
            valid_sets=[lgb_vaild],
            callbacks=[
                lgb.log_evaluation(100),
                lgb.early_stopping(stopping_rounds=100)
            ]
        )
        self.best_params = model.params
        with open(self.output_dir / model_filename, "wb") as f:
            pickle.dump(model, f)
        # 特徴量重要度の可視化
        # 予測の精度にどの要素が一番効いてるか
        lgb.plot_importance(
            model, importance_type="gain", figsize=(30, 15), max_num_features=50
        )
        plt.savefig(self.output_dir / f"{importance_filename}.png")
        plt.close()
        importance_df = pd.DataFrame(
            {
                "feature": model.feature_name(),
                "importance": model.feature_importance(importance_type="gain")
            }
        ).sort_values("importance", ascending=False)
        importance_df.to_csv(
            self.output_dir / f"{importance_filename}.csv",
            index = False,
            sep="\t"
        )
        #テストデータに対してスコアリング
        evaluation_df = test_df[
            [
                "race_id",
                "horse_id",
                "target",
                "rank",
                "tansho_odds",
                "popularity",
                "umaban"
            ]
        ].copy()
        evaluation_df["pred"] = model.predict(test_df[self.feature_cols])
        return evaluation_df

    def run (
        self,
        valid_start_date: str,
        test_start_date: str,
        importance_filename: str = "importance",
        model_filename: str = "model.pickle",
        evaluation_filename: str = "evaluation.csv"
    ):
        """
        学習処理を実行する。
        test_start_dateをYYYY-MM-DD形式で指定すると、
        その日付以降のデータをテストデータに、
        それより前のデータを学習データに分割する関数。
        """
        self.create_dataset(valid_start_date, test_start_date)
        evaluation_df = self.train(
            self.train_df, self.valid_df, self.test_df, importance_filename, model_filename
        )
        evaluation_df.to_csv(
            self.output_dir / evaluation_filename, sep="\t", index=False
        )
        return evaluation_df