# Allison Bolen
# win 2019
# cis678
# Wolffe

import pandas as pd
import pickle, os, math
from pprint import pprint

# defaults:
def save_it_all(obj, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_objects(file):
    with open(file, 'rb') as input:
        return pickle.load(input)

def saveFrame(df, name):
    df.to_csv(name+".csv", index=False, sep=",", header=True)
    save_it_all(df, name+".pkl")

def classify(instance, tree):
    '''
    This takes in an instance of data for classification and a tree to classify against
    this method is recursive
    and returns the resulting instance predication
    '''
    result = None
    for k,v in tree.items():
        edge = instance[k]
        if type(tree[k][edge]) is dict:
            t = tree.copy()
            t = t[k][edge]
            result = classify(instance, t)
        else:
            return tree[k][edge]
    return result

def main():
    # #fish
    # tree = load_objects("DataFiles/fishData/fishTree.pkl")
    # dataframe = pd.read_csv("DataFiles/fishData/processedFishData.pkl")

    # # contacts
    # tree = load_objects("DataFiles/contactData/contactTree.pkl")
    # dataframe = pd.read_csv("DataFiles/contactData/contactData.pkl")

    # cars
    tree = load_objects("../DataFiles/carData/carTreeGini.pkl")
    dataframe = pd.read_csv("../DataFiles/carData/car_test.csv")

    # # hw2
    # test = load_objects("DataFiles/hw2set/trainTree.pkl")
    # dataframe = pd.read_csv("DataFiles/hw2set/test.csv")

    results = []
    count = 0
    for i in range(0,len(dataframe.index)):
        dataframeNoClass = dataframe.drop(columns=["Class"])
        prediction = classify(dataframeNoClass.iloc[i, : ], tree)
        results.append(prediction)
        try:
            if prediction == dataframe.iloc[i, : ]["Class"]:
                count = count + 1
            else:
                # print("-------WRONG--------")
                print(str(prediction)+"!")
                i = None
        except KeyError as e:
            print("Key Error")
    print()
    print("Right predictions: " + str(count) + ".\nWrong Predictions: "+str(len(dataframe.index)-count)+".\nOut of " + str(len(dataframe.index)) + " total intances")

if __name__ == "__main__": main()
