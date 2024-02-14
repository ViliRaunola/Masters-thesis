import common
import menu
import reddit


def main():

    print("Hello World!")

    # Let user train the models
    while True:
        if common.check_model_folders():
            break

        userinput = menu.start_menu()
        menu.switch_start(userinput)

    nlp_tools = common.load_pipelines()
    prawn_connection = reddit.create_praw_instance()

    # common.test_ner_pipeline(tools.ner_pipeline)
    # common.test_pipeline(tools.sentiment_pipeline)

    # Let user to use the main functionalities
    while True:
        userinput = menu.main_menu()
        menu.switch_main(
            input=userinput, prawn_connection=prawn_connection, nlp_tools=nlp_tools
        )

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
