import os
import re
from typing import Type

import dotenv
import praw
from praw import exceptions as praw_exceptions

import text_processing
from common import NlpTools, colors


def create_praw_instance():
    dotenv.load_dotenv()

    reddit = praw.Reddit(
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),
        user_agent=os.getenv("USER_AGENT"),
    )

    return reddit


def _get_post_by_url(reddit: Type[praw.Reddit], url: str):
    """
    Returns reddit submission
    """

    post = reddit.submission(url=url)
    return post


def _get_reddit_post_url():
    print("Give the Reddit post URL")
    userinput = input(">: ")
    return userinput


def _title_sentiment_analysis(nlp_tools: Type[NlpTools], title: str):
    title_sentiment = nlp_tools.sentiment_pipeline(title)[0]
    print("Title:", title)
    print("Label:", title_sentiment["label"])
    print("Score: {:.5}".format(title_sentiment["score"]))


def _root_comment_analysis(
    nlp_tools: Type[NlpTools], post: Type[praw.Reddit.submission]
):
    for comment in post.comments:
        comment_text = text_processing.remove_emojis(comment.body)
        comment_text = text_processing.remove_url(comment_text)
        comment_text = text_processing.remove_reddit_quotation(comment_text)
        print(comment_text)


def start_reddit_analyzer(nlp_tools: Type[NlpTools], reddit: Type[praw.Reddit]):

    while True:
        try:
            url = _get_reddit_post_url()
            post = _get_post_by_url(reddit=reddit, url=url)
            break
        except (praw_exceptions.InvalidURL, ValueError) as error:
            print(
                f"{colors.CYELLOW}The provided link is invalid, please try again!{colors.CEND}"
            )
            print(f"{colors.CYELLOW}{error}{colors.CEND}")

    # Processing the title data
    title = post.title
    title = text_processing.remove_emojis(title)
    title = text_processing.remove_url(title)
    _title_sentiment_analysis(title=title, nlp_tools=nlp_tools)

    #!TODO Add sentiment analysis for comments, and sub comments
    #!TODO Add the ner tagging, combine the results somehow... maybe use the text highlight


# reddit = create_praw_instance()

# url = "https://www.reddit.com/r/Suomi/comments/1apt9gp/ty%C3%B6paikkakiusaaminen/"

# post = get_post_by_url(reddit=reddit, url=url)

# text = "Ukraina ja georgia. 😭😭 Kaks maata jotka ovat todella hyviä internet-sympatian saamisessa ja pillittämisessä muuttamatta tosiasiaa. Yawn, next"
# text = text_processing.remove_emojis(text)

# text2 = ">Eikö nää tyypit tajua"
# text2 = text_processing.remove_reddit_quotation(text2)

# text3 = "[Linkki](https://twitter.com/Ukraine/status/1468206078940823554?t=MOgBs7L-AbAsktl8pgcSKQ&s=19)"
# text3 = text_processing.remove_url(text3)

# print(text)
# print(text2)
# print(text3)
