import pandas as pd
from pathlib import Path
import json
import numpy as np
import re

COMMON_DATA_DIR = Path("..", "..", "common", "data")
INPUT_DIR = COMMON_DATA_DIR / "rawdf"
MAPPING_DIR = COMMON_DATA_DIR / "mapping"
POPULATION_DIR = Path("..", "data", "00_population")
OUTPUT_DIR = Path("..", "data", "01_preprocessed")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# カテゴリ変数を数値に変換するためのマッピング
with open(MAPPING_DIR / "sex.json", "r", encoding="utf-8") as f:
    sex_mapping = json.load(f)
with open(MAPPING_DIR / "race_type.json", "r", encoding="utf-8") as f:
    race_type_mapping = json.load(f)
with open(MAPPING_DIR / "around.json", "r", encoding="utf-8") as f:
    around_mapping = json.load(f)
with open(MAPPING_DIR / "weather.json", "r", encoding="utf-8") as f:
    weather_mapping = json.load(f)
with open(MAPPING_DIR / "ground_state.json", "r", encoding="utf-8") as f:
    ground_state_mapping = json.load(f)
with open(MAPPING_DIR / "race_class.json", "r", encoding="utf-8") as f:
    race_class_mapping = json.load(f)
# with open(MAPPING_DIR / "place.json", "r", encoding="utf-8") as f:
#     place_mapping = json.load(f)

def process_results(
        input_dir: Path = INPUT_DIR,
        output_dir: Path = OUTPUT_DIR,
        save_filename: str = "results.csv",
        sex_mapping: dict = sex_mapping
) -> pd.DataFrame:
    """
    未加工のレース結果テーブルをinput_dirから読み込んで加工し、
    output_dirに保存する関数。
    """
    df = pd.read_csv(input_dir / save_filename, sep="\t")
    # errors="coerce"で変換できない場合は欠損値（NaN）で変換される
    # デフォルトはerrors="raise"で変換できない場合は例外スローされる
    df["rank"] = pd.to_numeric(df["着順"], errors="coerce")
    # subsetは欠損値を除外
    # inplace=Trueはdf変更が反映される
    df.dropna(subset=["rank"], inplace=True)
    df["rank"] = df["rank"].astype(int)
    df["umaban"] = df["馬番"].astype(int)
    df["tansho_odds"] = df["単勝"].astype(float)
    df["popularity"] = df["人気"].astype(int)
    df["impost"] = df["斤量"].astype(float)
    df["wakuban"] = df["枠番"].astype(int)
    df["sex"] = df["性齢"].str[0].map(sex_mapping)
    df["age"] = df["性齢"].str[1:].astype(int)
    # 「d+」は数字の繰り返しまでの部分
    df["weight"] = df["馬体重"].str.extract(r"(\d+)").astype(int)
    # 「.」は任意の文字列、エスケープ[\]で()の中に任意の文字列を取り出す
    df["weight_diff"] = df["馬体重"].str.extract(r"\((.+)\)").astype(int)
    # データが着順に並んでることによるリーク防止のため、核レースを馬番順にソートする
    df = df.sort_values(["race_id", "umaban"])
    # 使用する列を選択
    df = df[
        [
            "race_id",
            "horse_id",
            "jockey_id",
            "trainer_id",
            "owner_id",
            "rank",
            "umaban",
            "wakuban",
            "tansho_odds",
            "popularity",
            "impost",
            "sex",
            "age",
            "weight",
            "weight_diff"
        ]
    ]
    df.to_csv(output_dir / save_filename, sep="\t", index=False)
    return df

def process_race_info(
        population_dir: Path = POPULATION_DIR,
        population_filename: str = "population.csv",
        input_dir: Path = INPUT_DIR,
        output_dir: Path = OUTPUT_DIR,
        input_filename: str = "race_info.csv",
        output_filename: str = "race_info.csv",
        race_type_mapping: dict = race_type_mapping,
        around_mapping: dict = around_mapping,
        weather_mapping: dict = weather_mapping,
        ground_state_mapping: dict = ground_state_mapping,
        race_class_mapping: dict = race_class_mapping
) -> pd.DataFrame:
    """
    未加工のレース情報テーブルをinput_dirから読み込んで加工し、
    output_dirに保存する関数。
    """
    # population = pd.read_csv(population_dir / population_filename, sep="\t")
    # df = pd.read_csv(input_dir / input_filename, sep="\t").query(
    #     "race_id in @population['race_id']"
    # )
    df = pd.read_csv(input_dir / input_filename, sep="\t")
    # evalで文字列型の列をリスト型に変換し、一時的な列を作成
    df["tmp"] = df["info1"].map(lambda x: eval(x)[0])
    # ダートor芝or障害
    df["race_type"] = df["tmp"].str[0].map(race_type_mapping)
    # 右or左or直線
    df["around"] = df["tmp"].str[1].map(around_mapping)
    df["course_len"] = df["info1"].str.extract(r"(\d+)m")[0].astype(int)
    # 天候
    df["weather"] = df["info1"].str.extract(r"\天候:(\w+)")[0].map(weather_mapping)
    df["ground_state"] = (
        df["info1"].str.extract(r"(芝|ダート|障害):(\w+)")[1].map(ground_state_mapping)
    )
    df["date"] = pd.to_datetime(
        df["info2"].map(lambda x: eval(x)[0]), format='%Y年%m月%d日'
    )
    # 「race_class=(L)」は正規表現で認識されないので、エスケープ処理が必要
    regex_race_class = "|".join([re.escape(k) for k in race_class_mapping])
    df["race_class"] = (
        df["title"]
        .str.extract(rf"({regex_race_class})")
        # タイトルからレース階級情報が取れない場合はinfo2から取得
        .fillna(df["info2"].str.extract(rf"({regex_race_class})"))[0]
        .map(race_class_mapping)
    )
    df["place"] = df["race_id"].astype(str).str[4:6].astype(int)
    df["month"] = df["date"].dt.month
    # 約4年に一度のうるる年も考慮
    df["sin_date"] = np.sin(2 * np.pi * df["date"].dt.day_of_year / 365.25) + 1
    df["cos_date"] = np.cos(2 * np.pi * df["date"].dt.day_of_year / 365.25) + 1
    # 使用する列を選択
    df = df[
        [
            "race_id",
            "date",
            "race_type",
            "around",
            "course_len",
            "weather",
            "ground_state",
            "race_class",
            "place",
            "month",
            "sin_date",
            "cos_date"
        ]
    ]
    df.to_csv(output_dir / output_filename, sep="\t", index=False)
    return df

def process_return_tables(
        population_dir: Path = POPULATION_DIR,
        population_filename: str = "population.csv",
        input_dir: Path = INPUT_DIR,
        output_dir: Path = OUTPUT_DIR,
        input_filename: str = "return_tables.csv",
        output_filename: str = "return_tables.pickle"
) -> pd.DataFrame:
    """
    未加工の払い戻しテーブルをinput_dirから読み込んで加工し、
    output_dirに保存する関数。
    """
    # population = pd.read_csv(population_dir / population_filename, sep="\t")
    # df = pd.read_csv(input_dir / input_filename, sep="\t").query(
    #     "race_id in @population['race_id']"
    # )
    df = pd.read_csv(input_dir / input_filename, sep="\t", index_col=0)
    df = (
        df[["0", "1", "2"]]
        .replace(" (-|→) ", "-", regex=True)
        .replace(",", "", regex=True)
        .apply(lambda x: x.str.split())
        .explode(["1", "2"])
        .explode("0")
        .apply(lambda x: x.str.split("-"))
        .explode(["0", "2"])
    )
    df.columns = ["bet_type", "win_umaban", "return"]
    df["return"] = df["return"].astype(int)
    # リスト型を維持したいのでpickleで保存
    df.to_pickle(output_dir / output_filename)
    return df

def process_horse_results(
        population_dir: Path = POPULATION_DIR,
        population_filename: str = "population.csv",
        input_dir: Path = INPUT_DIR,
        output_dir: Path = OUTPUT_DIR,
        input_filename: str = "horse_results.csv",
        output_filename: str = "horse_results.csv",
        race_type_mapping: dict = race_type_mapping,
        weather_mapping: dict = weather_mapping,
        ground_state_mapping: dict = ground_state_mapping,
        race_class_mapping: dict = race_class_mapping,
        # place_mapping: dict = place_mapping
) -> pd.DataFrame:
    """
    未加工の馬の過去成績テーブルをinput_dirから読み込んで加工し、
    output_dirに保存する関数。
    """
    # population = pd.read_csv(population_dir / population_filename, sep="\t")
    # df = pd.read_csv(input_dir / input_filename, sep="\t").query(
    #     "horse_id in @population['horse_id'] and 着順 not in ['取', '除']"
    # )
    df = pd.read_csv(input_dir / input_filename, sep="\t")
    # errors="coerce"で変換できない場合は欠損値（NaN）で変換される
    # デフォルトはerrors="raise"で変換できない場合は例外スローされる
    df["rank"] = (
        pd.to_numeric(df["着順"].str.replace("(降)", ""), errors="coerce")
        .fillna(df["頭数"])
        .astype(int)
    )
    # subsetは欠損値を除外
    # inplace=Trueはdf変更が反映される
    # df.dropna(subset=["rank"], inplace=True)
    df["date"] = pd.to_datetime(df["日付"])
    df["weather"] = df["天気"].map(weather_mapping)
    df["race_type"] = df["距離"].str[0].map(race_type_mapping)
    df["course_len"] = df["距離"].str.extract(r"(\d+)").astype(int)
    df["ground_state"] = df["馬場"].map(ground_state_mapping)
    # 着差は1着以外は「1着との差」を表すが、1着のみ「2着との差」のデータが入っている
    df["rank_diff"] = df["着差"].map(lambda x: 0 if x < 0 else x)
    df["prize"] = df["賞金"].fillna(0)
    regex_race_class = "|".join([re.escape(k) for k in race_class_mapping])
    df["race_class"] = (
        df["レース名"].str.extract(rf"({regex_race_class})")[0].map(race_class_mapping)
    )
    df["time"] = pd.to_datetime(df["タイム"], format="%M:%S.%f", errors="coerce")
    df["time"] = (
        df["time"].dt.minute * 60
        + df["time"].dt.second
        + df["time"].dt.microsecond / 1000000
    )
    df["win"] = (df["rank"] == 1).astype(int)
    df["rentai"] = (df["rank"] <= 2).astype(int)
    df["show"] = (df["rank"] <= 3).astype(int)
    # df["place"] = df["開催"].str.extract(r"(\D+)")[0].map(place_mapping)
    df["place"] = df["開催"].str.extract(r"(\D+)")[0]
    df.rename(columns={"頭数": "n_horses"}, inplace=True)
    # 使用する列を選択
    df = df[
        [
            "horse_id",
            "date",
            "rank",
            "prize",
            "rank_diff",
            "weather",
            "race_type",
            "course_len",
            "ground_state",
            "race_class",
            "time",
            "win",
            "rentai",
            "show",
            "place",
            "n_horses"
        ]
    ]
    df.to_csv(output_dir / output_filename, sep="\t", index=False)
    return df
