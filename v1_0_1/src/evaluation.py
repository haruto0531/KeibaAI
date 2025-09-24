import pandas as pd
from pathlib import Path
import math

DATA_DIR = Path("..", "data")
PREPROCESSED_DIR = DATA_DIR / "01_preprocessed"
TRAIN_DIR = DATA_DIR / "03_train"
OUTPUT_DIR = DATA_DIR / "04_evaluation"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

class Evaluator:
    def __init__(
        self,
        return_tables_filepath: Path = PREPROCESSED_DIR / "return_tables.pickle",
        evaluation_filepath: Path = TRAIN_DIR / "evaluation.csv",
        outdir: Path = OUTPUT_DIR
    ):
        self.return_tables = pd.read_pickle(return_tables_filepath)
        self.evaluation_df = pd.read_csv(evaluation_filepath, sep="\t")
        self.outdir = OUTPUT_DIR

    def box_top_n (
        self,
        sort_col: str = "pred",
        ascending: bool =False,
        n:int = 3,
        exp_name: str ="model"
    ):
        """
        sort_colで指定した列でソートし、上位n件のBOX馬券の的中率・回収率を
        シミュレーションする関数。
        """
        bet_df = (
            self.evaluation_df.sort_values(sort_col, ascending=ascending)
            .groupby("race_id")
            .head(n)
            .groupby("race_id")["umaban"]
            .apply(lambda x: list(x.astype(str)))
            .reset_index()
        )
        df = bet_df.merge(self.return_tables, on="race_id")
        # 的中判定。BOXで全通り賭けシミュレーションなので、集合の包含関係で判定できる。
        # applyはデフォルトで列ごとに処理される
        # axis=1にすると行ごとに処理されるようになる
        df["hit"] = df.apply(
            lambda  x: set(x["win_umaban"]).issubset(set(x["umaban"])), axis=1
        )
        # 馬券種ごとの的中率
        agg_hitrate = (
            df.groupby(["race_id", "bet_type"])["hit"]
            .max()
            .groupby("bet_type")
            .mean()
            .rename(f"hitrate_{exp_name}").to_frame()
        )
        # 馬券種ごとの回収率
        df["hit_return"] = df["return"] * df["hit"]
        n_bets_dict = {
            "単勝": n,
            "複勝": n,
            "馬連": math.comb(n, 2),
            "ワイド": math.comb(n, 2),
            "馬単": math.perm(n, 2),
            "三連複": math.comb(n, 3),
            "三連単": math.perm(n, 3)
        }
        agg_df = (
            df.groupby(["race_id", "bet_type"])["hit_return"]
            .sum()
            .reset_index()
        )
        agg_df["n_bets"] = agg_df["bet_type"].map(n_bets_dict)
        agg_df = (
            agg_df.query("n_bets > 0")
            .groupby("bet_type")[["hit_return", "n_bets"]]
            .sum()
        )
        agg_returnrate = (
            (agg_df["hit_return"] / agg_df["n_bets"] / 100)
            .rename(f"returnrate_{exp_name}").to_frame()
        )
        # left_indexとright_indexで結合する、両方ともTrue場合はinner join
        return pd.merge(agg_hitrate, agg_returnrate, left_index=True, right_index=True)

    def summarize_box_top_n(
        self,
        sort_col: str = "pred",
        ascending: bool =False,
        n:int = 3,
        exp_name: str ="model",
        save_filename: str = "box_summary.csv"
    ) -> pd.DataFrame:
        """
        topnの的中率・回収率を人気順モデルと比較してまとめる関数。
        """
        summary_df = pd.concat(
            [
                self.box_top_n(sort_col, ascending, n, exp_name), # 実験モデル
                self.box_top_n("popularity", True, n, "pop") # 人気順モデル
            ],
            axis=1
        ).sort_index(axis=1)
        summary_df.to_csv(OUTPUT_DIR / save_filename, sep="\t")
        return summary_df
