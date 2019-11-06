import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open('{}/data/laurel.txt'.format(BASE_DIR), 'r') as t:
    print(t)