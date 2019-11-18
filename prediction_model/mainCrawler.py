import requests
import time
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class GameResultCrawler():
    def __init__(self, name):
        self.dataset = list()
        self.apiKey = "RGAPI-e4809d00-01e5-458d-bf17-b85887800f86"
        self.myName = name.replace(' ', '-')
        self.headers = {
            "Origin": "https://developer.riotgames.com",
            "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Riot-Token": self.apiKey,
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6,zh;q=0.5",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36"
        }
        URL_id = "https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/{}".format(name)
        res_id = requests.get(URL_id, headers=self.headers)
        self.accountId = res_id.json()['accountId']

    def crawlMatchlists(self, counter, season, queue=420): # queue=420 for Rank game
        URL_matchlists = "https://kr.api.riotgames.com/lol/match/v4/matchlists/by-account/{0}?queue={1}&season={2}&endIndex={3}&beginIndex={4}".format(self.accountId, queue, season, (counter+1)*100+1, counter*100+1)
        res_matchlists = requests.get(URL_matchlists, headers=self.headers)
        
        matchlist = [mtc["gameId"] for mtc in res_matchlists.json()["matches"]]

        return matchlist

    def crawlMatch(self, match):
        URL_match = "https://kr.api.riotgames.com/lol/match/v4/matches/{}".format(match)
        res_match = requests.get(URL_match, headers=self.headers)
        data = res_match.json()

        if res_match.status_code == 429:
            print("pending")
            time.sleep(int(res_match.headers['Retry-After']))

            res_match = requests.get(URL_match, headers=self.headers)
            data = res_match.json()

        myPick = None     # id to identify myPick in match

        participantIdentities = data['participantIdentities']
        result = None       # win = 1
        for pi in participantIdentities:
            if pi['player']['accountId'] == self.accountId:
                myPick = pi['participantId']
                if pi['participantId'] <= 5:
                    if data['teams'][0]['win'] == 'Win': result = 1
                    else:  result = 0
                else:
                    if data['teams'][1]['win'] == 'Win': result = 1
                    else:  result = 0

        participants = data['participants']
        champions = list()
        for ptc in participants:
            champions.append(ptc['championId'])

        # [picks1~10], result, myPick
        dataset = list()
        dataset.extend(champions)
        dataset.append(result)
        dataset.append(myPick)

        
        return dataset

    def main_crawler(self, exist_matchlist=False):
        season = 13
        matchlist = list()
        dataset = list()

        if exist_matchlist:
            path = os.path.join(BASE_DIR, 'data', 'history', 'matchlist', 'matchlist_{}.json'.format(self.myName))
            m = open(path, 'r')
            matchlist = json.load(m)
        else:
            list_counter = 0        # matchlist counter
            try:
                print("Collecting Matchlist")
                while True:        
                    matchlist.extend(self.crawlMatchlists(list_counter, season))
                    list_counter += 1
            except Exception as e:
                print(e)
            finally:
                print("Complete collecting Matchlist")
                path = os.path.join(BASE_DIR, 'data', 'history', 'matchlist', 'matchlist_{}.json'.format(self.myName))
                with open(path, 'w') as mt:
                    json.dump(matchlist, mt, indent=4)

        try:
            print("Collecting Match results")
            for match in matchlist:
                try:
                    data = self.crawlMatch(match)
                    #print(data)
                    dataset.append(data)
                except Exception as e:
                    print("{0} : {1}".format(match, e))
        except Exception as e:
            print(e)
        finally:
            path = os.path.join(BASE_DIR, 'data', 'history', 'history_{}.json'.format(self.myName))
            with open(path, 'w') as f:
                json.dump(dataset, f, indent=4)
            return True

# For Frontend 
def crawl_splash():
    t = open(os.path.join(BASE_DIR, 'prediction_model', 'documents', 'champion_table.json'), 'r', encoding='UTF8')
    table = json.load(t)

    for champ in table:
        url = "http://ddragon.leagueoflegends.com/cdn/9.22.1/img/champion/{}.png".format(champ["name"])
        response = requests.get(url)
        if response.status_code == 200:
            with open(os.path.join(BASE_DIR, 'prediction_model', 'documents', 'splash', "{}.png".format(champ["id"])), 'wb') as i:
                i.write(response.content)

if __name__ == "__main__":
    #crawler = GameResultCrawler('laurelwoods')
    #crawler.main_crawler()
    crawl_splash()