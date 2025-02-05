import requests
from urllib import parse
import random
import csv
import os
from bs4 import BeautifulSoup

class NewsByTransfer():
    def __init__(self, from_area=None, to_area=None, date=None):
        self.fromArea = from_area
        self.toArea = to_area
        self.date = date

    def getOneJsUrl(self):
        fromArea = parse.quote(parse.quote(self.fromArea))
        departureStation = parse.quote(fromArea)
        toArea = parse.quote(parse.quote(self.toArea))
        arrivalStation = parse.quote(toArea)
        js_url = f"https://trains.ctrip.com/pages/booking/getTransferList?departureStation={departureStation}&arrivalStation={arrivalStation}&departDateStr={self.date}"
        print(js_url)
        return js_url

    def getOneNews(self, js_url):
        UA = [
            "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36",
            "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:30.0) Gecko/20100101 Firefox/30.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/537.75.14",
            "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Win64; x64; Trident/6.0)",
            'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11',
            'Opera/9.25 (Windows NT 5.1; U; en)',
            'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)',
            'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5 (like Gecko) (Kubuntu)',
            'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.8.0.12) Gecko/20070731 Ubuntu/dapper-security Firefox/1.5.0.12',
            'Lynx/2.8.5rel.1 libwww-FM/2.14 SSL-MM/1.4.1 GNUTLS/1.2.9',
            "Mozilla/5.0 (X11; Linux i686) AppleWebKit/535.7 (KHTML, like Gecko) Ubuntu/11.04 Chromium/16.0.912.77 Chrome/16.0.912.77 Safari/535.7",
            "Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:10.0) Gecko/20100101 Firefox/10.0 "
        ]
        user_agent = random.choice(UA)
        headers = {
            "User-Agent": user_agent,
            "Referer": "https://trains.ctrip.com/",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive"
        }

        try:
            response = requests.get(js_url, headers=headers, timeout=10)
            response.raise_for_status()  # 如果状态码不是200，将引发HTTPError异常
            
            # 打印响应内容，用于调试
            print("Response content:", response.text)
            
            text = response.json()
            
            if "data" not in text or "transferList" not in text["data"]:
                print("Unexpected JSON structure:", text)
                return []

            transferList = text["data"]["transferList"]
            csvList = []

            for oneTransfer in transferList:
                tranDict = {
                    "总出发站": oneTransfer["departStation"],
                    "总目的站": oneTransfer["arriveStation"],
                    "总信息": f"{oneTransfer['transferStation']}换乘 停留{oneTransfer['transferTakeTime']} 全程{oneTransfer['totalRuntime']} 价格{oneTransfer['showPriceText']}"
                }

                for i, trainInfo in enumerate(oneTransfer["trainTransferInfos"], 1):
                    tranDict[f"班次列车号{i}"] = trainInfo['trainNo']
                    tranDict[f"发车时间-到站时间{i}"] = f"{trainInfo['departDate']} {trainInfo['departTime']}---{trainInfo['arriveDate']} {trainInfo['arriveTime']}"
                    tranDict[f"发车站-目的站{i}"] = f"{trainInfo['departStation']}---{trainInfo['arriveStation']}"

                csvList.append(tranDict)

            return csvList

        except requests.exceptions.RequestException as e:
            print(f"请求错误: {e}")
        except ValueError as e:
            print(f"JSON解析错误: {e}")
        except KeyError as e:
            print(f"数据结构错误: {e}")
        
        return []

    def mkcsv(self, csvlist):
        if not csvlist:
            print("没有数据可写入CSV文件")
            return

        filename = f"/home/leon/agent/web_crawler/result/{csvlist[0]['总出发站']}到{csvlist[0]['总目的站']}转站查找.csv"
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csvlist[0].keys())
            writer.writeheader()
            writer.writerows(csvlist)
        print(f"CSV文件已保存: {filename}")

    def main(self):
        if not all([self.fromArea, self.toArea, self.date]):
            print("缺少必要的信息（出发站、目的站或日期）")
            return

        js_url = self.getOneJsUrl()
        csvList = self.getOneNews(js_url)
        if csvList:
            self.mkcsv(csvList)
        else:
            print("未能获取数据，请检查网络连接或网站结构是否发生变化")

if __name__ == "__main__":
    # 示例用法
    crawler = NewsByTransfer("上海", "北京", "2024-10-11")
    crawler.main()