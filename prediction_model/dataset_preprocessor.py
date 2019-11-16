import json
import numpy as np
import pandas as pd
from collections import defaultdict
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class DatasetPreprocessor:
    def __init__(self, name):
        self.name = name.replace(' ', '-')
        m = open(os.path.join(BASE_DIR, 'prediction_model', 'documents', 'mapping.json'), 'r')
        self.map = json.load(m)
        
        history = open(os.path.join(BASE_DIR, 'data', 'history', 'history_{}.json'.format(self.name)))
        self.history = json.load(history)

        columns = ['x' + str(i) for i in range(146)]
        columns.insert(0, 'y')
        self.columns = columns
    
    def mapping(self, data, result): 
        dataset = list()
        champion_picks = [0 for i in range(145)]
        
        for d in data:
            champion_picks[self.map[str(d)]] = 1

        dataset.append(result)
        dataset.append(1)   # Bias Column
        dataset.extend(champion_picks)
        
        return dataset

    def process_dataset(self):
        dataframe = defaultdict(list)

        for data in self.history:
            myPick = data.pop()
            result = data.pop()

            if myPick <= 5:                
                team_data = data[:5]
                enemy_data= data[5:]
            else:
                enemy_data = data[:5]
                team_data= data[5:]
            
            my_data = [result]
            mapped_my_pick = self.map[str(data[myPick-1])]
            my_data.append(mapped_my_pick)           

            dataframe["myPicks"].append(my_data)
            dataframe["team"].append(self.mapping(team_data, result))
            dataframe["enemy"].append(self.mapping(enemy_data, result))
        
        with open(os.path.join(BASE_DIR, 'data', 'dataset', 'dataset_{0}_{1}.json'.format(self.name, "pick_history")), 'w') as j:
            json.dump(dataframe["myPicks"], j, indent=4)
            
        dataset_team = np.array(dataframe["team"], dtype=np.float32)
        dataset_team = pd.DataFrame(dataset_team, columns=self.columns)

        dataset_enemy = np.array(dataframe["enemy"], dtype=np.float32)
        dataset_enemy = pd.DataFrame(dataset_enemy, columns=self.columns)

        
        dataset_team.to_csv(os.path.join(BASE_DIR, 'data', 'dataset', 'dataset_{0}_{1}.csv'.format(self.name, "team")), index=False)
        dataset_enemy.to_csv(os.path.join(BASE_DIR, 'data', 'dataset', 'dataset_{0}_{1}.csv'.format(self.name, "enemy")), index=False)


def process_input(model_input):
    m = open(os.path.join(BASE_DIR, 'prediction_model', 'documents', 'mapping.json'), 'r')
    mapping = json.load(m)

    processed_input = list()
    champion_picks = [float(0) for i in range(145)]
    
    for mi in model_input:
        champion_picks[mapping[str(mi)]] = 1.0

    processed_input.append(1.0)   #bias
    processed_input.extend(champion_picks)
    return processed_input


if __name__ == "__main__":
    preprocessor = DatasetPreprocessor("hide on bush")
    preprocessor.process_dataset()
    