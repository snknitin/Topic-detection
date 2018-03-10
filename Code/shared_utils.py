"""
Helper methods used across project
"""

import os

def vocab(dir,output_path):
    """
     Create a vocab file separately
    :param dir:
    :return:
    """
    pass

def display(model, feature_names, num_to_display):
    """
     Function to display the topics by their top words
    :param model:
    :param feature_names:
    :param num_to_display:
    :return:
    """
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-num_to_display - 1:-1]]))