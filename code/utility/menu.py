from typing import Type

import analyzer.reddit as reddit
import models.classifier as classifier
import models.tagger as tagger
import utility.globals as globals
from praw import Reddit
from utility.common import NlpTools, colors, exit_program


def start_menu():
    """
    Prints menu and returns the user input.

    0 - exit
    1 - train NER model
    2 - train classifier model
    """

    menu = f"""
{"-" * globals.MENU_BORDER_SIZE}
| You need to have neural networks created on your machine before analyzing posts.
| 0 - exit the program
| 1 - train NER model
| 2 - train classifier model
{"-" * globals.MENU_BORDER_SIZE}
    """
    print(menu)

    userinput = input(">: ")
    return userinput


def main_menu():
    """
    Prints menu and returns the user input
    1 - analyze reddit full
    2 - post only
    3 - post + root comments
    0 - exit
    """

    menu = f"""
{"-" * globals.MENU_BORDER_SIZE}
| Please select the next operation:
| 0) exit
| 1) analyse reddit post
| 2) analyse reddit post and root comments
| 3) analyse reddit post, root comments and comment replies (full)
{"-" * globals.MENU_BORDER_SIZE}
        """
    print(menu)

    userinput = input(">: ")
    return userinput


def switch_main(input: str, prawn_connection: Type[Reddit], nlp_tools: Type[NlpTools]):
    """
    Selecting the operation to execute based on user input of the main menu
    """

    if input == "1":
        reddit.start_reddit_analyzer_post(nlp_tools=nlp_tools, reddit=prawn_connection)
    elif input == "2":
        reddit.start_reddit_analyzer_post_root_comments(
            nlp_tools=nlp_tools, reddit=prawn_connection
        )
    elif input == "3":
        reddit.start_reddit_analyzer_full(nlp_tools=nlp_tools, reddit=prawn_connection)
    elif input == "0":
        exit_program()
    else:
        print(f"{colors.CYELLOW}Unknown input, try again{colors.CEND}")


def switch_start(input: str):
    """
    Selecting the operation to execute based on user input of the first menu
    """

    if input == "1":
        tagger.create_ner_finbert()
    elif input == "2":
        classifier.create_finbert()
    elif input == "0":
        exit_program()
    else:
        print(f"{colors.CYELLOW}Unknown input, try again{colors.CEND}")
