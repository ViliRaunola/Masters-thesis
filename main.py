import classifier
import common
import menu
import tagger


class Tools:
    def __init__(self, sentiment_pipeline, ner_pipeline):
        self.sentiment_pipeline = sentiment_pipeline
        self.ner_pipeline = ner_pipeline


def check_model_folders():
    """
    Check if the folders are created.
    Returns true if both folders exist.
    """

    is_ner_trained = common.check_folder_ner()
    is_class_trained = common.check_folder_class()

    if is_class_trained & is_ner_trained:
        return True
    if not is_ner_trained:
        print(
            f"{common.colors.CYELLOW}Ner folder doesn't exist or is empty. Please train the model first.{common.colors.CEND}"
        )
    if not is_class_trained:
        print(
            f"{common.colors.CYELLOW}Classifier folder doesn't exist or is empty. Please train the model first.{common.colors.CEND}"
        )
    return False


def main():

    print("Hello World!")

    while True:
        if check_model_folders():
            break

        userinput = menu.start_menu()
        menu.switch_start(userinput)

    tools = Tools(classifier.get_sentiment_pipeline(), tagger.get_ner_pipeline())

    common.test_ner_pipeline(tools.ner_pipeline)
    common.test_pipeline(tools.sentiment_pipeline)

    while True:
        userinput = menu.main_menu()
        menu.switch_main(userinput)

    # classifier.create_finbert()
    # tagger.create_ner_finbert()

    # sentiment_pipeline = classifier.get_sentiment_pipeline()
    # if sentiment_pipeline == None:
    #     print(f"{common.colors.CRED}Couldn't load the pipeline{common.colors.CEND}")
    #     return 0
    # print(f"{common.colors.CGREEN}Loaded the sentiment pipeline!{common.colors.CEND}")

    # ner_pipeline = tagger.get_ner_pipeline()
    # if ner_pipeline == None:
    #     print(f"{common.colors.CRED}Couldn't load the pipeline{common.colors.CEND}")
    #     return 0
    # print(f"{common.colors.CGREEN}Loaded the ner pipeline!{common.colors.CEND}")

    # test_pipeline(sentiment_pipeline)
    # test_ner_pipeline(ner_pipeline)


if __name__ == "__main__":
    main()
