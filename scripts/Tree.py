# Allison Bolen
# win 2019
# cis678
# Wolffe

import pandas as pd
import pickle, os, math
from pprint import pprint

def main():

    # #fish
    # baseInfoDict = load_objects("DataFiles/fishData/fishCounts.pkl")
    # dataframe = load_objects("DataFiles/fishData/processedFishData.pkl")
    # saveFile = "DataFiles/fishData/fishTree.pkl"

    # # contacts
    # baseInfoDict = load_objects("DataFiles/contactData/contactCounts.pkl")
    # dataframe = load_objects("DataFiles/contactData/contactData.pkl")
    # saveFile = "DataFiles/contactData/contactTree.pkl"

    # cars
    baseInfoDict = load_objects("../DataFiles/carData/carCounts.pkl")
    dataframe = load_objects("../DataFiles/carData/carData.pkl")
    saveFile = "../DataFiles/carData/carTree.pkl"

    # # hw2
    # baseInfoDict = load_objects("DataFiles/hw2set/trainCounts.pkl")
    # dataframe = load_objects("DataFiles/hw2set/train.pkl")
    # saveFile = "DataFiles/hw2set/trainTree.pkl"

    # print(dataframe)

    tree = makeTree(baseInfoDict, dataframe, "")
    save_it_all(tree, saveFile)

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

# tree methods:
def setInfo(dataFrame, infoDict):
    '''
    This sets up the overall data of the dataframe the summary of the entire set
    '''
    roundXDict = {"setInfo":{}}
    for classValue in infoDict["classInfo"]["values"]:
        num = len(dataFrame["Class"][dataFrame["Class"]==classValue].tolist())
        roundXDict["setInfo"][classValue] = num/infoDict["total"]

    purity = []
    for key, value in roundXDict["setInfo"].items():
        if value == 0 or value == 1:
            purity.append(0)
        else:
            purity.append(-(value)*math.log(value,2))


    e = sum(purity)

    roundXDict["setInfo"]["purity"] = e
    return roundXDict

def entropy(val):
    return -(val)*math.log(val,2)

def gini(val):
    return val**2

def attributeInfo(dataFrame, infoDict, roundX):
    '''
    This populates a dictionary full of the data frame info
    this caclulates the number of instaces for each branch attribute
    the purity for each branch attribute
    and the class percentage
    '''
    roundX["attributes"] = {}
    # make dictionary
    for attribute in infoDict["attributeInfo"]:
        if attribute != "num":
            roundX["attributes"][attribute] = {"attrTypes":{}}
            for attributeValue in infoDict["attributeInfo"][attribute]["values"]:
                roundX["attributes"][attribute]["attrTypes"][attributeValue] = {"values":{"numInstance":0}}

    # do data gathering
    for classValue in infoDict["classInfo"]["values"]:
        for attribute in infoDict["attributeInfo"]:
            if attribute != "num":
                for attributeValue in infoDict["attributeInfo"][attribute]["values"]:
                    classFrame = dataFrame[dataFrame["Class"] == classValue]
                    numAttribute = len(dataFrame[dataFrame[attribute]==attributeValue])
                    if numAttribute != 0:
                        numAttributeWithClass = len(classFrame[classFrame[attribute] == attributeValue])
                        roundX["attributes"][attribute]["attrTypes"][attributeValue]["values"]["numInstance"] += numAttributeWithClass
                        roundX["attributes"][attribute]["attrTypes"][attributeValue]["values"][classValue] = numAttributeWithClass/numAttribute
                        roundX["attributes"][attribute]["attrTypes"][attributeValue]["values"]["purity"] = None
                    else:
                        roundX["attributes"][attribute]["attrTypes"][attributeValue]["values"]["numInstance"] += 0 # possible bug
                        roundX["attributes"][attribute]["attrTypes"][attributeValue]["values"][classValue] = 0
                        roundX["attributes"][attribute]["attrTypes"][attributeValue]["values"]["purity"] = None

    # setup purity
    for attribute in roundX["attributes"]:
        for key, value in roundX["attributes"][attribute]["attrTypes"].items():
                p = []
                for subkey, subvalue in value["values"].items():
                    if subkey != "purity" and subkey != "numInstance":
                        if subvalue == 0 or subvalue == 1:
                            p.append(0)
                        else:
                            # change for gini or entropy
                            p.append(entropy(subvalue))
                    purity = sum(p)
                    roundX["attributes"][attribute]["attrTypes"][key]["values"]["purity"] = purity

    return roundX

def gainCheck(dataFrame, infoDict, roundXDict):
    '''
        This function calculates the gain for each branch
    '''
    for attribute, value in roundXDict["attributes"].items():
        gain = []
        for types, vals in value['attrTypes'].items():
            typeInfo = vals["values"]
            # set purity  - sum((typecount/classcount)*purityoftype + ...) =
            gain.append((typeInfo["numInstance"]/infoDict["total"]*typeInfo["purity"]))

        finalGain = roundXDict["setInfo"]["purity"] - sum(gain)
        roundXDict["attributes"][attribute]["gain"] = finalGain
    return roundXDict

def setUpMax(roundX):
    """
    Get the maximum gain value of the set and return that as the new branch value
    """
    vals = []
    for attribute in roundX["attributes"]:
        vals.append(roundX["attributes"][attribute]["gain"])
    branch = []
    for attribute in roundX["attributes"]:
        if max(vals) == roundX["attributes"][attribute]["gain"]:
            branch.append(attribute)
    return branch

import collections
def findMajority(roundX, branch):
    """
    For attributes that dont have any existance in the test set please give them the majority value at the current branch
    """
    majority = []
    for attributeType, value in roundX["attributes"][branch[0]]["attrTypes"].items():
        # print("\n"+tabs+str(attributeType).upper())
        if value["values"]["purity"] == 0 and value["values"]["numInstance"] != 0: # we are at a termianl node, this node is pure
            # check the class values to get the class node value
            for key, val in value["values"].items():
                if key != "purity" and key != "numInstance": # for class values only
                    if val != 0:
                        majority.append(key)
    counter = collections.Counter(majority)
    return counter.most_common()[0][0]

def getBranch(infoDict, dataframe):
    '''
        Go through teh steps of getting the branch of the current set
    '''
    roundX = setInfo(dataframe, infoDict)
    roundX = attributeInfo(dataframe, infoDict, roundX)
    roundX = gainCheck(dataframe, infoDict, roundX)
    branch = setUpMax(roundX)
    return branch , roundX

def makeTree(infoDict, dataframe, tabs):
    '''
    This makes the tree for the data set
    it is recursive
    '''
    # return the next branch level
    branch, roundX = getBranch(infoDict, dataframe)

    tree = {branch[0]:{}}
    print(tabs+"This is the next branch: "+branch[0])
    if len(branch) > 1:
        print("YOU HAVE MORE THAN ONE CHOICE FOR THIS BRANCH")

    for attributeType, value in roundX["attributes"][branch[0]]["attrTypes"].items():
        tree[branch[0]][attributeType]= {}
        if value["values"]["purity"] == 0 and value["values"]["numInstance"] != 0: # we are at a termianl node, this node is pure
            # check the class values to get the class node value
            for key, val in value["values"].items():
                if key != "purity" and key != "numInstance": # for class values only
                    if val != 0:
                        # get the terminal leaf
                        print(tabs+"Terminal for branch " + branch[0] +" at value "+ str(attributeType) + " : "+key)
                        tree[branch[0]][attributeType] = key
        elif value["values"]["numInstance"] == 0:
            # if you dont have any occrances in the trainin set youll just be the majority result at this node
            majority = findMajority(roundX, branch)
            tree[branch[0]][attributeType] = findMajority(roundX, branch)
            print(tabs+"Terminal for branch " + branch[0] +" at value "+ str(attributeType) + " : "+majority)
        else:
            # look for the next branch level
            # edit dataframe
            dataFrameEdit = dataframe
            dataFrameUse = dataFrameEdit[dataFrameEdit[branch[0]]==attributeType]
            dataFrameUse = dataFrameUse.drop(columns=[branch[0]])
            # change info dictionary
            infoDictEdit = infoDict.copy()
            infoDictEdit["attributeInfo"] = infoDict["attributeInfo"].copy()
            del infoDictEdit["attributeInfo"][branch[0]]
            infoDictEdit["attributeInfo"]["num"] -= 1

            infoDictEdit["total"] = len(dataFrameUse.index)
            print(tabs+"New round branching off of "+ branch[0] + " at " + str(attributeType))
            # call recursive
            tree[branch[0]][attributeType] = makeTree(infoDictEdit, dataFrameUse, tabs+"\t")

    return tree

if __name__ == "__main__": main()
