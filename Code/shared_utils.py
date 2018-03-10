import os

def vocab(dir,output_path):
    """
    :param dir:
    :return:
    """
    pass

def display(model, feature_names, num_to_display):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-num_to_display - 1:-1]]))