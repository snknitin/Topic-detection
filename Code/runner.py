import argparse
parser = argparse.ArgumentParser()
parser.add_argument("data_directory", help="Enter the location of the dialogs4 folder",type=str)
parser.add_argument("mode", help="Must be one of LDA and NMF",type=str)
parser.add_argument("numTopics", help="Select the number of distinct topics you want to find",type=str)
parser.add_argument("vocab_size", help="Enter size of vocabulary",type=str)

args = parser.parse_args()


def detect_topics(args):
    """
    :param args:
    :return:
    """
    #if args.len




if __name__ == '__main__':
    detect_topics(args)