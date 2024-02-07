import classifier


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


def main():
    print("Hello World!")
    # classifier.create_finbert()
    sentiment_pipeline = classifier.get_sentiment_pipeline()
    test_pipeline(sentiment_pipeline)


if __name__ == "__main__":
    main()
