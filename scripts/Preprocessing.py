# Allison Bolen
# win 2019
# cis678
# Wolffe

import sys
import pandas as pd
import pickle, os


# In[71]:

def main():
    '''
        This is the main that converts teh data files into an info dictinary and a data frame
        it saves tehse for future use
    '''
    # # fish
    # dataFrame = pd.read_csv("../DataFiles/fishData/fishing.csv", header=0, sep=',')
    # dataFrame.head()
    # saveFrame(dataFrame, "../DataFiles/fishData/processedFishData")
    # info = readFile("../DataFiles/fishData/fishInfo.txt")
    # save = "../DataFiles/fishData/fishCounts.pkl"

    # # contacts
    # dataFrame = pd.read_csv("../DataFiles/contactData/contact-lenses.csv", header=0, sep=',')
    # dataFrame.head()
    # saveFrame(dataFrame, "../DataFiles/contactData/contactData")
    # info = readFile("../DataFiles/contactData/contactInfo.txt")
    # save = "../DataFiles/carData/contactCounts.pkl"

    # cars
    dataFrame = pd.read_csv("../DataFiles/carData/car_training.csv", header=0, sep=',')
    dataFrame.head()
    saveFrame(dataFrame, "../DataFiles/carData/carData")
    info = readFile("../DataFiles/carData/cartraininginfo.txt")
    save = "../DataFiles/carData/carCounts.pkl"

    print("Saving " + save)
    save_it_all(prep(info), save)

# defualts:
def save_it_all(obj, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, protocol=2)

def load_objects(file):
    with open(file, 'rb') as input:
        return pickle.load(input)

def saveFrame(df, name):
    df.to_csv(name+".csv", index=False, sep=",", header=True)
    save_it_all(df, name+".pkl")

# read info file:
def readFile(fname):
    with open(fname) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def prep(info):
    """
        This gets the info for the data set organized and in a dictionray
    """
    infoDict = {}
    for index in range(0, len(info)):
        if index == 0:
            # class values
            infoDict["classInfo"] = {"num": int(info[index]), "values": info[1].split(",")}

        if index == 2:
            infoDict["attributeInfo"] = {"num":int(info[index])}
            for attributeIndex in range(index+1, index+1+int(info[index])):
                attribute = info[attributeIndex].split(",")[0]
                numValues = info[attributeIndex].split(",")[1]
                values = info[attributeIndex].split(",")[2:]
                if RepresentsInt(values[0]):
                    values = list(map(int, values))
                infoDict["attributeInfo"][attribute] = {"num":numValues, "values":values}
        if index == len(info)-1:
            infoDict["total"] = int(info[index])

    return infoDict

if __name__ == "__main__": main()
