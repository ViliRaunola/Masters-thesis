import common


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


class tools:
    def __init__(self, sentiment_pipeline, ner_pipeline):
        self.sentiment_pipeline = sentiment_pipeline
        self.ner_pipeline = ner_pipeline


def main():

    print("Hello World!")

    while True:
        is_ner_trained = common.check_folder_ner()
        is_class_trained = common.check_folder_class()

        if is_class_trained & is_ner_trained:
            break
        if not is_ner_trained:
            print(
                f"{common.colors.CYELLOW}Ner folder doesn't exist or is empty. Please train the model first.{common.colors.CEND}"
            )
        if not is_class_trained:
            print(
                f"{common.colors.CYELLOW}Classifier folder doesn't exist or is empty. Please train the model first.{common.colors.CEND}"
            )

        userinput = common.print_start_menu()
        common.switch_start(userinput)

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
