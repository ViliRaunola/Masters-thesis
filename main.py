import transformers

import classifier
import tagger


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
        ]
    )

    for i in tulos:
        for j in i:
            print(j)
        print("\n")


def main():

    print("Hello World!")
    classifier.create_finbert()
    tagger.create_ner_finbert()

    sentiment_pipeline = classifier.get_sentiment_pipeline()
    if sentiment_pipeline == None:
        print("Couldn't load the pipeline")
        return 0
    print("Loaded the sentiment pipeline!")

    ner_pipeline = tagger.get_ner_pipeline()
    if ner_pipeline == None:
        print("Couldn't load the pipeline")
        return 0
    print("Loaded the ner pipeline!")

    test_pipeline(sentiment_pipeline)
    test_ner_pipeline(ner_pipeline)


if __name__ == "__main__":
    main()
