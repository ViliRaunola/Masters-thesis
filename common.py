import os
import sys
from typing import Type

from transformers import Pipeline

import classifier
import tagger

_REPO_NAME_CLASS = "./model"
_REPO_NAME_NER = "./model_ner"

CLASSIFIER_LABELS = ["neg", "neut", "pos"]


class NlpTools:
    """
    Used to store the loaded pipelines
    """

    def __init__(
        self, sentiment_pipeline: Type[Pipeline], ner_pipeline: Type[Pipeline]
    ):
        self.sentiment_pipeline = sentiment_pipeline
        self.ner_pipeline = ner_pipeline


class colors:
    CEND = "\33[0m"
    CBOLD = "\33[1m"
    CITALIC = "\33[3m"
    CURL = "\33[4m"
    CBLINK = "\33[5m"
    CBLINK2 = "\33[6m"
    CSELECTED = "\33[7m"

    CBLACK = "\33[30m"
    CRED = "\33[31m"
    CGREEN = "\33[32m"
    CYELLOW = "\33[33m"
    CBLUE = "\33[34m"
    CVIOLET = "\33[35m"
    CBEIGE = "\33[36m"
    CWHITE = "\33[37m"

    CBLACKBG = "\33[40m"
    CREDBG = "\33[41m"
    CGREENBG = "\33[42m"
    CYELLOWBG = "\33[43m"
    CBLUEBG = "\33[44m"
    CVIOLETBG = "\33[45m"
    CBEIGEBG = "\33[46m"
    CWHITEBG = "\33[47m"

    CGREY = "\33[90m"
    CRED2 = "\33[91m"
    CGREEN2 = "\33[92m"
    CYELLOW2 = "\33[93m"
    CBLUE2 = "\33[94m"
    CVIOLET2 = "\33[95m"
    CBEIGE2 = "\33[96m"
    CWHITE2 = "\33[97m"

    CGREYBG = "\33[100m"
    CREDBG2 = "\33[101m"
    CGREENBG2 = "\33[102m"
    CYELLOWBG2 = "\33[103m"
    CBLUEBG2 = "\33[104m"
    CVIOLETBG2 = "\33[105m"
    CBEIGEBG2 = "\33[106m"
    CWHITEBG2 = "\33[107m"


def load_pipelines():
    """
    Loads the sentiment analysis and the named entity
    recognition pipeline to Tools class instance and returns it.
    """

    print(f"{colors.CBLINK2}Loading the models...{colors.CEND}")

    try:
        nlp_tools = NlpTools(
            classifier.get_sentiment_pipeline(), tagger.get_ner_pipeline()
        )
    except:
        print(f"{colors.CRED}Error while loading the pipelines!{colors.CEND}")
        exit_program("Exited in loading the pipelines")

    print(f"{colors.CGREEN}Pipelines are loaded and ready to use{colors.CEND}")

    return nlp_tools


def test_pipeline(sentiment_pipeline):
    test_sentenses = [
        "Tämä ML teknologia on aivan uskomatonta!",
        "Voi kunpa mieki osaisin koodata :D.",
        "Tämä lause on tosi vitun paska ja negaatiivinen, kys!",
        "Positiivinen lause.",
        "Mikään ei onnistu.",
    ]

    results = sentiment_pipeline(test_sentenses)
    print("The results from testing:")
    for index, sentense in enumerate(test_sentenses):
        print(sentense)
        print(results[index])
        print("\n\n")


def test_ner_pipeline(ner_pipeline):
    tulos = ner_pipeline(
        [
            "Tässä lauseessa kerrotaan, että Vilin syntymäpäivä on 19.06.1997",
            "Twitter on ollut vuosia ihan paska...",
            "Pakko antaa kyllä u/Pontus_Pilates :lle propsit käyttäjänimestä, ansaitsi nenä tuhahduksen",
        ]
    )

    for i in tulos:
        for j in i:
            print(j)
        print("\n")


def check_folder_ner():
    """
    True, if folder exist and has data --> model most likely trained
    """
    if not os.path.isdir(_REPO_NAME_NER):
        return False
    if not os.listdir(_REPO_NAME_NER):
        return False
    return True


def check_folder_class():
    """
    True, if folder exist and has data --> model most likely trained
    """
    if not os.path.isdir(_REPO_NAME_CLASS):
        return False
    if not os.listdir(_REPO_NAME_CLASS):
        return False
    return True


def check_model_folders():
    """
    Check if the folders are created.
    Returns true if both folders exist.
    """

    is_ner_trained = check_folder_ner()
    is_class_trained = check_folder_class()

    if is_class_trained & is_ner_trained:
        return True
    if not is_ner_trained:
        print(
            f"{colors.CYELLOW}Ner folder doesn't exist or is empty. Please train the model first.{colors.CEND}"
        )
    if not is_class_trained:
        print(
            f"{colors.CYELLOW}Classifier folder doesn't exist or is empty. Please train the model first.{colors.CEND}"
        )
    return False


def exit_program(exit_message=None):
    """
    Exiting the program. If exit_message is provided,
    this message is displayed instead of the default exit message.
    """

    message = f"""
    {colors.CBLINK2}Exiting...{colors.CEND}
    {colors.CBLUE}Thank you for using the program!{colors.CEND}
    """

    if exit_message is None:
        print(message)
        sys.exit()
    else:
        sys.exit(exit_message)
