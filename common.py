import os
import sys

import classifier
import tagger

MENU_BORDER_SIZE = 20
_REPO_NAME_CLASS = "./model"
_REPO_NAME_NER = "./model_ner"


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


def print_start_menu():
    """
    Prints menu and returns the user input.

    0 - exit
    1 - train NER model
    2 - train classifier model
    """

    menu = f"""
    {"-" * MENU_BORDER_SIZE}
    You need to have neural networks created on your machine before analyzing posts.
    0 - exit the program
    1 - train NER model
    2 - train classifier model
    {"-" * MENU_BORDER_SIZE}
    """
    print(menu)

    userinput = input("Your selection: ")
    return userinput


def print_main_menu():
    """
    Prints menu and returns the user input
    """
    menu = f"""
        {"-" * MENU_BORDER_SIZE}
        | Please select the next operation:
        | 1) Create
        | 
        {"-" * MENU_BORDER_SIZE}
        """
    print(menu)

    userinput = input("Your selection: ")
    return userinput


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


def switch_start(input):
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


def exit_program():
    message = f"""
    {colors.CBLINK2}Exiting...{colors.CEND}
    {colors.CBLUE}Thank you for using the program!{colors.CEND}
    """
    print(message)

    sys.exit()
