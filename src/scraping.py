from urllib.request import urlopen
import pandas as pd
from bs4 import BeautifulSoup # Htmlの細かい要素を取り出したい場合に使うライブラリ
import re #正規表現を使いたい場合のライブラリ
import time
from tqdm.notebook import tqdm # 実行進捗を可視化できるライブラリ

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from tqdm.notebook import tqdm
import traceback
from pathlib import Path

HTML_RACE_DIR = Path("..", "data", "html", "race")

def scrape_kaisai_date(from_: str, to_: str) -> list[str]:
    """
    from_toとto_をyyyy-mmの形で取得すると、間の開催日一覧を取得する関数。
    """
    kaisai_data_list = []

    for date in tqdm(pd.date_range(from_, to_, freq="MS")):
        year = date.year
        month = date.month
        url = f"https://race.netkeiba.com/top/calendar.html?year={year}&month={month}"
        html = urlopen(url).read() # スクレイピング
        time.sleep(1) # スクレイピングをFor文使用する際、絶対忘れないように（パフォーマンス低下防ぎ）
        soup = BeautifulSoup(html, "lxml")
        a_list = soup.find("table", class_="Calendar_Table").find_all("a")
        for a in a_list:
            kaisai_data = re.findall(r"kaisai_date=(\d{8})", a["href"])[0]
            kaisai_data_list.append(kaisai_data)
    return kaisai_data_list

def scrape_race_id_list(kaisai_data_list: list[str]) -> list[str]:
    """
    開催日（yyyymmdd形式)をリストで入れると、レースid一覧が返ってくる関数。
    """
    options = Options()
    # バックグラウンドで実行
    options.add_argument("--headless")
    # 最新版のChromeをインストールし、インストール先のパスが返却される
    driver_path = ChromeDriverManager().install()
    race_id_list = []

    # インストールした最新版のChromeブラウザを起動する
    with webdriver.Chrome(service=Service(driver_path), options=options) as driver:
        for kaisai_data in tqdm(kaisai_data_list):
            url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={kaisai_data}"
            try:
                driver.get(url)
                time.sleep(1)
                li_list = driver.find_elements(By.CLASS_NAME, "RaceList_DataItem")
                for li in li_list:
                    href = li.find_element(By.TAG_NAME, "a").get_attribute("href")
                    race_id = re.findall(r"race_id=(\d{12})", href)[0]
                    race_id_list.append(race_id)
            except:
                print(f"stopped at {url}")
                print(traceback.format_exc())
                break
    return race_id_list

def scrap_html_race(race_id_list: list[str], save_dir: Path = HTML_RACE_DIR) -> list[Path]:
    """
    netkeiba.comのraceページのhtmlをスクレイピングして、save_dirに保存する関数。
    既にhtmlが存在する場合はスキップされ、新たに取得されたhtmlのパスだけが返ってくる。
    """
    html_path_list = []
    save_dir.mkdir(parents=True, exist_ok=True)
    for race_id in tqdm(race_id_list):
        filepath = save_dir / f"{race_id}.bin"
        # binファイルが既に存在する場合はスキップする
        if filepath.is_file():
            print(f"skipped:{race_id}")
            continue
        url = f"https://db.netkeiba.com/race/{race_id}"
        html = urlopen(url).read()
        time.sleep(1)
        with open(filepath, "wb") as f:
            f.write(html)
        html_path_list.append(filepath)
    return html_path_list