import argparse
import os

import shared_utils as su
import classifier as clf
import data_processing as dp
import models as m

# Script parameters
parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="Enter the location of the dialogs4 folder",type=str)
parser.add_argument("final_log", help="Enter the location to store the processed files",type=str)

parser.add_argument("mode", help="Must be one of LDA and NMF",type=str)
parser.add_argument("numTopics", help="Select the number of distinct topics you want to find",type=int)
parser.add_argument("vocab_size", help="Enter size of vocabulary",type=int)
parser.add_argument("isProcessed", help="Enter size of vocabulary",type=bool)

args = parser.parse_args()


def detect_topics(args):
    """
    :param args:
    :return:
    """

    if args.isProcessed:
        # Check if directory exists
        if not os.path.exists(args.final_log):
            os.makedirs(args.final_log)
        # Create Vocab file
        #su.vocab(args.data_dir,args.final_log)
        dp.preprocess(args.data_dir,args.final_log)

    # Load the dataset as a list with document contents
    documents=dp.loadDocument(args.final_log)
    mode_modules = {
        "LDA": m.runLDA,
        "NMF": m.runNMF}
    # Launch chosen module
    if args.mode in mode_modules:
        print("Mode: ", args.mode)
        module = mode_modules[args.mode]
        module(documents,args.vocab_size,args.numTopics)



if __name__ == '__main__':
    detect_topics(args)