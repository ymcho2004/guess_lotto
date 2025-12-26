import requests
import pandas as pd
from tqdm import tqdm
import time

def get_lotto_numbers(start_round, end_round):
    lotto_list=[]

    print(f"**{start_round}회부터 {end_round}회까지 로또 데이터를 가져옵니다...**")

    for i in tqdm(range(start_round, end_round+1)):
        url=f"https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={i}"

        try:
            response = requests.get(url)
            data=response.json()

            if data["returnValue"]=="success":
                row={
                    "round": i,
                    "date": data["drwNoDate"],
                    "drwtNo1": data["drwtNo1"],
                    "drwtNo2": data["drwtNo2"],
                    "drwtNo3": data["drwtNo3"],
                    "drwtNo4": data["drwtNo4"],
                    "drwtNo5": data["drwtNo5"],
                    "drwtNo6": data["drwtNo6"],
                    "bonus": data["bnusNo"]
                }
                lotto_list.append(row)

            else:
                print(f"{i}회차의 정보가 없습니다!")

        except Exception as e:
            print(f"{i}회차의 정보를 불러오는데 오류가 발생했습니다!")

        time.sleep(0.1)

    return pd.DataFrame(lotto_list)

if __name__ == "__main__":
    df=get_lotto_numbers(1, 1203)

    df.to_csv("lotto_history.csv", index=False)
    print("\n✅ 'lotto_history.csv' 저장 완료! 데이터를 확인해보세요.")
    print(df.tail())
