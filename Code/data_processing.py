"""Data processing methods"""

import os

def preprocess(datapath,filepath):
    """
     Remove unwanted information and clean the files for use
    :param datapath: Reads files from data location
    :param filepath: Writes processed output files into a different location
    :return:
    """
    print("Processing the raw files")
    files = os.listdir(datapath)
    for file in files:
        with open(os.path.join(filepath, file), 'w',encoding="utf-8") as target:
            with open(os.path.join(datapath, file), 'r',encoding="utf-8") as doc:
                content = doc.read().split('\n')
                for line in content:
                    target.write(line.split('\t')[-1].lower())
        target.close()

def loadDocument(filepath):
    """
    Load the cleaned data into a list for vectorizers
    :param filepath: Read all files in this location
    :return: Return a list of all the documents' contents
    """
    documents=[]
    print("Loading document object from processed files")
    files = os.listdir(filepath)
    for file in files:
        with open(os.path.join(filepath, file), 'r', encoding="utf-8") as doc:
            documents.append(doc.read())
    return documents