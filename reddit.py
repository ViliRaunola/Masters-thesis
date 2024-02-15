import os
import re
from collections import Counter
from typing import Type

import dotenv
import praw
import tabulate
from praw import exceptions as praw_exceptions

import text_processing
from common import CLASSIFIER_LABELS, NlpTools, colors


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
    header = ["Label", "Score", "Title after processing"]
    rows = [[title_sentiment["label"], title_sentiment["score"], title]]
    print("\n")
    print(f"{colors.CBLUEBG}Title{colors.CEND}")
    print(tabulate.tabulate(tabular_data=rows, headers=header))


def _post_sentiment_analysis(nlp_tools: Type[NlpTools], post: str):
    post_sentiment = nlp_tools.sentiment_pipeline(post)[0]
    header = ["Label", "Score", "Post after processing"]
    rows = [[post_sentiment["label"], post_sentiment["score"], post]]
    print("\n")
    print(f"{colors.CBLUEBG}Post{colors.CEND}")
    print(tabulate.tabulate(tabular_data=rows, headers=header))


def _count_sent_label_occurances(tags_list: list):
    counter = Counter()
    for dictionary in tags_list:
        for value in dictionary.values():
            if value in CLASSIFIER_LABELS:
                counter[value] += 1
    return counter


def _root_comment_analysis(
    nlp_tools: Type[NlpTools], post: Type[praw.Reddit.submission]
):
    tags_list = []
    rows = []
    for comment in post.comments:
        comment_text = text_processing.remove_emojis(comment.body)
        comment_text = text_processing.remove_url(comment_text)
        comment_text = text_processing.remove_reddit_quotation(comment_text)
        comment_sentiment = nlp_tools.sentiment_pipeline(comment_text)[0]

        # Save results
        tags_list.append(
            {"label": comment_sentiment["label"], "score": comment_sentiment["score"]}
        )
        rows.append(
            [comment_sentiment["label"], comment_sentiment["score"], comment_text]
        )

    # Individual results
    header = ["Label", "Score", "Comment after preprocessing"]
    print(f"\n{colors.CBLUEBG}Root comments{colors.CEND}")
    print(tabulate.tabulate(tabular_data=rows, headers=header, tablefmt="grid"))

    # Results combined
    counter = _count_sent_label_occurances(tags_list)
    print(f"\n{colors.CBLUEBG}Post's root comments combined labels{colors.CEND}")
    header = ["Negative", "Neutral", "Positive"]
    rows = [[counter["neg"], counter["neut"], counter["pos"]]]
    print(tabulate.tabulate(tabular_data=rows, headers=header))


def _recursion_on_comments(
    comment,
    depth: int,
    result_list: list,
    nlp_tools: Type[NlpTools],
):
    """
    Preprocesses the comment's body then does the sentiment analysis.
    If the comment has replies, calls the function recursively
    """
    depth += 1

    if comment.body:
        # Pre processing the text data
        comment_text = text_processing.remove_emojis(comment.body)
        comment_text = text_processing.remove_url(comment_text)
        comment_text = text_processing.remove_reddit_quotation(comment_text)

        comment_sentiment = nlp_tools.sentiment_pipeline(comment_text)[0]
        result_list.append(
            {"label": comment_sentiment["label"], "score": comment_sentiment["score"]}
        )

    if comment.replies:
        for reply in comment.replies:
            _recursion_on_comments(
                comment=reply, depth=depth, result_list=result_list, nlp_tools=nlp_tools
            )


def _sentiment_analysis_all_comments(post, nlp_tools: Type[NlpTools]):
    depth = 0
    root_comment_children_sentiments = []
    root_comment_tags_list = []
    root_comment_afer_preprocessing = []
    total_comments_sentiment = {"neg": 0, "neut": 0, "pos": 0}
    for index, comment in enumerate(post.comments):

        # Root comment analysis
        comment_text = text_processing.remove_emojis(comment.body)
        comment_text = text_processing.remove_url(comment_text)
        comment_text = text_processing.remove_reddit_quotation(comment_text)
        root_comment_afer_preprocessing.append(comment_text)
        comment_sentiment = nlp_tools.sentiment_pipeline(comment_text)[0]

        # Save result of root comment analysis
        root_comment_tags_list.append(
            {"label": comment_sentiment["label"], "score": comment_sentiment["score"]}
        )

        # Adding new list for each root comment
        root_comment_children_sentiments.append([])

        # Analysing the replies to the root comment
        for relply in comment.replies:
            _recursion_on_comments(
                comment=relply,
                depth=depth,
                result_list=root_comment_children_sentiments[index],
                nlp_tools=nlp_tools,
            )

    header = [
        "Label",
        "Score",
        "# of neg replies",
        "# of neut replies",
        "# of pos replies",
        "Comment after preprocessing",
    ]
    rows = []

    # Creting the data rows for the table
    for index, root_comment_results in enumerate(root_comment_children_sentiments):
        counter = _count_sent_label_occurances(root_comment_results)

        # From a root comment's replies
        total_comments_sentiment["neg"] += counter["neg"]
        total_comments_sentiment["neut"] += counter["neut"]
        total_comments_sentiment["pos"] += counter["pos"]

        # From the root comment
        if root_comment_tags_list[index]["label"] == "neg":
            total_comments_sentiment["neg"] += 1
        elif root_comment_tags_list[index]["label"] == "neut":
            total_comments_sentiment["neut"] += 1
        elif root_comment_tags_list[index]["label"] == "pos":
            total_comments_sentiment["pos"] += 1

        rows.append(
            [
                root_comment_tags_list[index]["label"],
                root_comment_tags_list[index]["score"],
                counter["neg"],
                counter["neut"],
                counter["pos"],
                root_comment_afer_preprocessing[index],
            ]
        )

    print(
        f"\n{colors.CBLUEBG}Post's root comment and root comment's replies{colors.CEND}"
    )
    print(tabulate.tabulate(tabular_data=rows, headers=header, tablefmt="grid"))

    header = [
        "# of neg replies",
        "# of neut replies",
        "# of pos replies",
    ]
    rows = [
        [
            total_comments_sentiment["neg"],
            total_comments_sentiment["neut"],
            total_comments_sentiment["pos"],
        ]
    ]
    print(f"\n{colors.CBLUEBG}Post's all comments (root + root replies){colors.CEND}")
    print(tabulate.tabulate(tabular_data=rows, headers=header, tablefmt="grid"))


def _print_counter_results(counter: Type[Counter]):
    for label, value in counter.items():
        print("{:<40}".format(f"Number of {label} comments:"), value)


def start_reddit_analyzer_post(nlp_tools: Type[NlpTools], reddit: Type[praw.Reddit]):
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
    _post_sentiment_analysis(post=post.selftext, nlp_tools=nlp_tools)


def start_reddit_analyzer_post_root_comments(
    nlp_tools: Type[NlpTools], reddit: Type[praw.Reddit]
):
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
    _post_sentiment_analysis(post=post.selftext, nlp_tools=nlp_tools)
    _root_comment_analysis(nlp_tools=nlp_tools, post=post)


def start_reddit_analyzer_full(nlp_tools: Type[NlpTools], reddit: Type[praw.Reddit]):

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
    _post_sentiment_analysis(post=post.selftext, nlp_tools=nlp_tools)
    # _root_comment_analysis(nlp_tools=nlp_tools, post=post)
    _sentiment_analysis_all_comments(post=post, nlp_tools=nlp_tools)

    #!TODO Add sentiment analysis for comments, and sub comments
    #!TODO Add the ner tagging, combine the results somehow... maybe use the text highlight
