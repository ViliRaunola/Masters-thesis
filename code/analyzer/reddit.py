import os
from collections import Counter
from typing import Type

import dotenv
import pandas as pd
import praw
import tabulate
import utility.globals as globals
import utility.text_processing as text_processing
from praw import exceptions as praw_exceptions
from utility import common
from utility.common import NlpTools, colors


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
    try:
        title_sentiment = nlp_tools.sentiment_pipeline(title)[0]
    except RuntimeError as err:
        print(err)
        print("using the version for longer inputs")
        title_sentiment = nlp_tools.sentiment_pipeline_long(title)
    header = ["Label", "Score", "Title after processing"]
    rows = [[title_sentiment["label"], title_sentiment["score"], title]]
    print("\n")
    print(f"{colors.CBLUEBG}Title{colors.CEND}")
    print(tabulate.tabulate(tabular_data=rows, headers=header))


def _post_sentiment_analysis(nlp_tools: Type[NlpTools], post: str):
    try:
        post_sentiment = nlp_tools.sentiment_pipeline(post)[0]
    except RuntimeError as err:
        print(err)
        print("using the version for longer inputs")
        post_sentiment = nlp_tools.sentiment_pipeline_long(post)

    header = ["Label", "Score", "Post after processing"]
    rows = [[post_sentiment["label"], post_sentiment["score"], post]]
    print("\n")
    print(f"{colors.CBLUEBG}Post{colors.CEND}")
    print(tabulate.tabulate(tabular_data=rows, headers=header))


def _count_sent_label_occurances(tags_list: list):
    counter = Counter()
    for dictionary in tags_list:
        for value in dictionary.values():
            if value in globals.CLASSIFIER_LABELS:
                counter[value] += 1
    return counter


def _root_comment_analysis(
    nlp_tools: Type[NlpTools], post: Type[praw.Reddit.submission]
):
    tags_list = []
    rows = []
    ner_saved = []
    for comment in post.comments:
        comment_text = text_processing.remove_emojis(comment.body)
        comment_text = text_processing.remove_url(comment_text)
        comment_text = text_processing.remove_reddit_quotation(comment_text)
        comment_text = text_processing.remove_deleted_and_removed_tags(comment_text)
        comment_text = text_processing.remove_new_lines(comment_text)

        try:
            comment_sentiment = nlp_tools.sentiment_pipeline(comment_text)[0]
        except RuntimeError as err:
            print(err)
            print("using the version for longer inputs")
            comment_sentiment = nlp_tools.sentiment_pipeline_long(post)

        # Save results
        tags_list.append(
            {"label": comment_sentiment["label"], "score": comment_sentiment["score"]}
        )

        temp = _text_ner_analysis(
            nlp_tools=nlp_tools,
            text=comment_text,
            context_score=tags_list[-1],
        )
        ner_saved.append(temp)

        rows.append(
            [comment_sentiment["label"], comment_sentiment["score"], comment_text]
        )

    # Individual results
    header = ["Label", "Score", "Comment after preprocessing"]
    file_name = "table_root_comments_sentiment.txt"
    file_name = common.create_unique_file_name(file_name)
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(tabulate.tabulate(tabular_data=rows, headers=header, tablefmt="grid"))
    print(
        f"\nRoot comments sentiment analysis saved to file: {colors.CBLUE}{file_name}{colors.CEND}\n"
    )

    # Results combined
    counter = _count_sent_label_occurances(tags_list)
    print(f"\n{colors.CBLUEBG}Post's root comments combined labels{colors.CEND}")
    header = ["Negative", "Neutral", "Positive"]
    rows = [[counter["neg"], counter["neut"], counter["pos"]]]
    print(tabulate.tabulate(tabular_data=rows, headers=header))

    ner_results = _combine_ner_results(ner_saved)
    _print_ner_results(ner_results, save_to_file=True)
    _most_common_ner_tags_in_results(ner_results)


def _combine_ner_results(ner_saved: list):
    temp = {
        "PERSON": [],
        "LOC": [],
        "ORG": [],
        "PRODUCT": [],
        "EVENT": [],
        "DATE": [],
        "GPE": [],
    }

    for comment_ner_results in ner_saved:
        for tag in comment_ner_results:
            if tag in temp:
                for result in comment_ner_results[tag]:
                    result["context_label"] = comment_ner_results["context_label"]
                    result["context_score"] = comment_ner_results["context_score"]
                    result["context_text"] = comment_ner_results["context_text"]

    for comment_ner_results in ner_saved:
        for tag in comment_ner_results:
            if tag in temp:
                temp[tag].extend(comment_ner_results[tag])

    return temp


def _recursion_on_comments(
    comment,
    result_list: list,
    nlp_tools: Type[NlpTools],
    ner_results_all: list,
    all_comments_sentiment_results: list,
):
    """
    Preprocesses the comment's body then does the sentiment analysis.
    If the comment has replies, calls the function recursively
    """

    if hasattr(comment, "body"):
        # Pre processing the text data
        comment_text = text_processing.remove_emojis(comment.body)
        comment_text = text_processing.remove_url(comment_text)
        comment_text = text_processing.remove_reddit_quotation(comment_text)
        comment_text = text_processing.remove_deleted_and_removed_tags(comment_text)
        comment_text = text_processing.remove_new_lines(comment_text)

        try:
            comment_sentiment = nlp_tools.sentiment_pipeline(comment_text)[0]
        except RuntimeError as err:
            print(err)
            print("using the version for longer inputs")
            comment_sentiment = nlp_tools.sentiment_pipeline_long(comment_text)

        result_list.append(
            {"label": comment_sentiment["label"], "score": comment_sentiment["score"]}
        )

        all_comments_sentiment_results.append(
            {
                "label": comment_sentiment["label"],
                "score": comment_sentiment["score"],
                "text": comment_text,
            }
        )

        unpacked_results = _text_ner_analysis(
            nlp_tools=nlp_tools,
            text=comment_text,
            context_score=result_list[-1],
        )
        ner_results_all.append(unpacked_results)

    if hasattr(comment, "replies"):
        for reply in comment.replies:
            _recursion_on_comments(
                comment=reply,
                result_list=result_list,
                nlp_tools=nlp_tools,
                ner_results_all=ner_results_all,
                all_comments_sentiment_results=all_comments_sentiment_results,
            )


def _analyse_root_and_replies(
    post: Type[praw.Reddit.submission],
    root_comment_after_preprocessing: list,
    root_comment_tags_list: list,
    root_comment_children_sentiments: list,
    nlp_tools: Type[NlpTools],
    ner_results_all: list,
    all_comments_sentiment_results: list,
):
    for index, comment in enumerate(post.comments):

        # Root comment analysis
        comment_text = text_processing.remove_emojis(comment.body)
        comment_text = text_processing.remove_url(comment_text)
        comment_text = text_processing.remove_reddit_quotation(comment_text)
        comment_text = text_processing.remove_deleted_and_removed_tags(comment_text)
        comment_text = text_processing.remove_new_lines(comment_text)

        root_comment_after_preprocessing.append(comment_text)

        try:
            comment_sentiment = nlp_tools.sentiment_pipeline(comment_text)[0]
        except RuntimeError as err:
            print(err)
            print("using the version for longer inputs")
            comment_sentiment = nlp_tools.sentiment_pipeline_long(comment_text)[0]

        # Save result of root comment analysis
        root_comment_tags_list.append(
            {"label": comment_sentiment["label"], "score": comment_sentiment["score"]}
        )

        all_comments_sentiment_results.append(
            {
                "label": comment_sentiment["label"],
                "score": comment_sentiment["score"],
                "text": comment_text,
            }
        )

        unpacked_results = _text_ner_analysis(
            nlp_tools=nlp_tools,
            text=comment_text,
            context_score=root_comment_tags_list[-1],
        )
        ner_results_all.append(unpacked_results)

        # Adding new list for each root comment
        root_comment_children_sentiments.append([])

        # Analysing the replies to the root comment
        for relply in comment.replies:
            _recursion_on_comments(
                comment=relply,
                result_list=root_comment_children_sentiments[index],
                nlp_tools=nlp_tools,
                ner_results_all=ner_results_all,
                all_comments_sentiment_results=all_comments_sentiment_results,
            )
    return (
        root_comment_after_preprocessing,
        root_comment_tags_list,
        root_comment_children_sentiments,
        ner_results_all,
        all_comments_sentiment_results,
    )


def _print_root_comment_results(
    total_comments_sentiment: dict,
    root_comment_children_sentiments: list,
    root_comment_tags_list: list,
    root_comment_after_preprocessing: list,
):
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
                root_comment_after_preprocessing[index],
            ]
        )

    file_name = "table_comments_sentiment.txt"
    file_name = common.create_unique_file_name(file_name)
    print(
        f"\nPost's root comment and root comment's replies are saved to file: {colors.CBLUE}{file_name}{colors.CEND}"
    )
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(tabulate.tabulate(tabular_data=rows, headers=header, tablefmt="grid"))


def _print_all_replies_sentiment(total_comments_sentiment: dict):
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


def _sentiment_analysis_all_comments(post, nlp_tools: Type[NlpTools]):
    root_comment_children_sentiments = []
    root_comment_tags_list = []
    root_comment_after_preprocessing = []
    ner_results_all = []
    total_comments_sentiment = {"neg": 0, "neut": 0, "pos": 0}
    all_comments_sentiment_results = []

    (
        root_comment_after_preprocessing,
        root_comment_tags_list,
        root_comment_children_sentiments,
        ner_results_all,
        all_comments_sentiment_results,
    ) = _analyse_root_and_replies(
        post=post,
        root_comment_after_preprocessing=root_comment_after_preprocessing,
        root_comment_tags_list=root_comment_tags_list,
        nlp_tools=nlp_tools,
        root_comment_children_sentiments=root_comment_children_sentiments,
        ner_results_all=ner_results_all,
        all_comments_sentiment_results=all_comments_sentiment_results,
    )

    # Individual results
    header = ["Label", "Score", "Comment after preprocessing"]
    rows = []
    for result in all_comments_sentiment_results:
        rows.append([result["label"], result["score"], result["text"]])
    file_name = "table_all_comments_sentiment.txt"
    file_name = common.create_unique_file_name(file_name)
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(
            tabulate.tabulate(
                tabular_data=rows,
                headers=header,
                tablefmt="grid",
            )
        )
    print(
        f"\nAll comments sentiment analysis results saved to file: {colors.CBLUE}{file_name}{colors.CEND}\n"
    )

    _print_root_comment_results(
        total_comments_sentiment=total_comments_sentiment,
        root_comment_after_preprocessing=root_comment_after_preprocessing,
        root_comment_children_sentiments=root_comment_children_sentiments,
        root_comment_tags_list=root_comment_tags_list,
    )

    _print_all_replies_sentiment(total_comments_sentiment=total_comments_sentiment)

    combined_ner_results = _combine_ner_results(ner_results_all)
    _print_ner_results(combined_ner_results, save_to_file=True)
    _most_common_ner_tags_in_results(combined_ner_results)


#!TODO refactor to reduce complexity
def _unpack_ner_results(results: list):
    parsed_results = {
        "PERSON": [],
        "LOC": [],
        "ORG": [],
        "PRODUCT": [],
        "EVENT": [],
        "DATE": [],
        "GPE": [],
    }
    enitity_count = 0

    for result in results:
        if result["entity"][0] == "B":
            enitity_count = 1
            for tag in parsed_results:
                if tag == result["entity"][2:]:
                    list_to_append = parsed_results[tag]
                    temp = {"word": result["word"], "score": result["score"]}
                    list_to_append.append(temp)
        elif result["entity"][0] == "I":
            for tag in parsed_results:
                if tag == result["entity"][2:]:
                    enitity_count += 1
                    list_to_append = parsed_results[tag]
                    # Special case in which the I tag is found before B. Can happen since the model isn't perfect.
                    if len(list_to_append) == 0:
                        temp = {"word": result["word"], "score": result["score"]}
                        list_to_append.append(temp)
                    else:
                        list_to_append[-1]["word"] = (
                            list_to_append[-1]["word"] + " " + result["word"]
                        )
                        # Calculating the average score of the words
                        list_to_append[-1]["score"] = (
                            list_to_append[-1]["score"] + result["score"]
                        ) / enitity_count

    return parsed_results


def _print_ner_results(results: dict, save_to_file: bool = False):
    header = [
        "Label",
        "Score",
        "Word",
        "Context label",
        "Context score",
        "Context text after preprocessing",
    ]
    rows = []
    for tag in results:
        for list_values in results[tag]:

            if (
                (list_values.get("context_score") is not None)
                or (list_values.get("context_label") is not None)
                or (list_values.get("context_text") is not None)
            ):
                rows.append(
                    [
                        tag,
                        list_values["score"],
                        list_values["word"],
                        list_values["context_label"],
                        list_values["context_score"],
                        list_values["context_text"],
                    ]
                )
            else:
                rows.append(
                    [
                        tag,
                        list_values["score"],
                        list_values["word"],
                    ]
                )

    if not save_to_file:
        print(tabulate.tabulate(tabular_data=rows, headers=header, tablefmt="grid"))
        return

    file_name = "table_ner_tags_comments.txt"
    file_name = common.create_unique_file_name(file_name)
    print(f"\nSaved NER results to file: {colors.CBLUE}{file_name}{colors.CEND}\n")
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(tabulate.tabulate(tabular_data=rows, headers=header, tablefmt="grid"))


def _create_data_frame_for_ner_sentiment_analysis(results: dict):
    header = [
        "Label",
        "Score",
        "Word",
        "Context label",
        "Context score",
        "Context text after preprocessing",
    ]
    rows = []
    #!TODO this has been done earlier, no need to compute again!
    for tag in results:
        for list_values in results[tag]:
            rows.append(
                [
                    tag,
                    list_values["score"],
                    list_values["word"],
                    list_values["context_label"],
                    list_values["context_score"],
                    list_values["context_text"],
                ]
            )

    df = pd.DataFrame(data=rows, columns=header)
    return df


def _get_top_amount_from_each_tag(
    tags: set, df: Type[pd.DataFrame], top_amount: int, most_common_tags: dict
):
    for tag in tags:
        counts = df.loc[df["Label"] == tag]["Word"].value_counts()
        top_ten = counts.head(top_amount)
        most_common_tags[tag].extend(top_ten.keys())

    return most_common_tags


def _reorder_ner_tag_to_context_value(
    most_common_tags: dict, df: Type[pd.DataFrame], tags: set
):
    for tag in tags:
        most_common_tags[tag] = {value: [] for value in most_common_tags[tag]}

    for tag in tags:
        for index, row in df.iterrows():
            for key, value in most_common_tags[tag].items():
                if key == row["Word"]:
                    value.append(row["Context label"])

    return most_common_tags


def _calculate_sentiment_occurances_to_ner_tags(most_common_tags: dict, tags: set):
    results = {}

    for tag in tags:
        for key, value in most_common_tags[tag].items():
            counter_neg = 0
            counter_neut = 0
            counter_pos = 0
            for sen_label in value:
                if sen_label == "neg":
                    counter_neg += 1
                elif sen_label == "neut":
                    counter_neut += 1
                elif sen_label == "pos":
                    counter_pos += 1
            results.setdefault(tag, {}).setdefault(key, {})["neg"] = counter_neg
            results.setdefault(tag, {}).setdefault(key, {})["neut"] = counter_neut
            results.setdefault(tag, {}).setdefault(key, {})["pos"] = counter_pos

    return results


def _write_ner_tag_with_sentiment_score_to_file(
    top_amount: int, tags: set, results: dict
):
    header = ["Word", "Neg", "Neut", "Pos", "Total"]
    rows = []

    file_name = "ner_context.txt"
    file_name = common.create_unique_file_name(file_name)

    with open(file_name, "w", encoding="utf-8") as f:
        f.write("")

    for tag in tags:
        if tag in results:
            for key, values in results[tag].items():
                rows.append(
                    [
                        key,
                        values["neg"],
                        values["neut"],
                        values["pos"],
                        values["neg"] + values["neut"] + values["pos"],
                    ]
                )

            with open(file_name, "a", encoding="utf-8") as f:
                f.write(f"The context of TOP {top_amount} most {tag} labels\n")
                f.write(tabulate.tabulate(tabular_data=rows, headers=header))
                f.write("\n\n")

            rows.clear()

    print(
        f"The TOP {top_amount} NER tags for each category and their context sentiment has been saved to file: {colors.CBLUE}{file_name}{colors.CEND}"
    )


def _most_common_ner_tags_in_results(results: dict):
    tags = {
        "PERSON",
        "LOC",
        "ORG",
        "PRODUCT",
        "EVENT",
        "DATE",
        "GPE",
    }

    top_amount = 10

    df = _create_data_frame_for_ner_sentiment_analysis(results=results)

    most_common_tags = {
        "PERSON": [],
        "LOC": [],
        "ORG": [],
        "PRODUCT": [],
        "EVENT": [],
        "DATE": [],
        "GPE": [],
    }

    most_common_tags = _get_top_amount_from_each_tag(
        tags=tags, most_common_tags=most_common_tags, df=df, top_amount=top_amount
    )

    most_common_tags = _reorder_ner_tag_to_context_value(
        most_common_tags=most_common_tags, df=df, tags=tags
    )

    results = _calculate_sentiment_occurances_to_ner_tags(
        most_common_tags=most_common_tags, tags=tags
    )

    _write_ner_tag_with_sentiment_score_to_file(
        tags=tags, top_amount=top_amount, results=results
    )


def _text_ner_analysis(
    nlp_tools: Type[NlpTools],
    text: str,
    context_score: dict = None,
):
    results = nlp_tools.ner_pipeline(text)
    unpacked_results = _unpack_ner_results(results)

    if context_score is not None:
        unpacked_results["context_label"] = context_score["label"]
        unpacked_results["context_score"] = context_score["score"]
        unpacked_results["context_text"] = text

    return unpacked_results


def _title_analysis(post: Type[praw.Reddit.post], nlp_tools: Type[NlpTools]):
    # Processing the title data
    title = post.title
    title = text_processing.remove_emojis(title)
    title = text_processing.remove_url(title)
    title = text_processing.remove_deleted_and_removed_tags(title)
    title = text_processing.remove_new_lines(title)

    _title_sentiment_analysis(title=title, nlp_tools=nlp_tools)

    print(f"\n{colors.CBLUEBG}Identified labels from Title{colors.CEND}")
    unpacked_results = _text_ner_analysis(nlp_tools=nlp_tools, text=title)
    _print_ner_results(unpacked_results)


def _post_analysis(post: Type[praw.Reddit.post], nlp_tools: Type[NlpTools]):
    post_text = post.selftext
    post_text = text_processing.remove_emojis(post_text)
    post_text = text_processing.remove_url(post_text)
    post_text = text_processing.remove_deleted_and_removed_tags(post_text)
    post_text = text_processing.remove_new_lines(post_text)
    _post_sentiment_analysis(post=post_text, nlp_tools=nlp_tools)
    print(f"\n{colors.CBLUEBG}Identified labels from Post's body{colors.CEND}")
    unpacked_results = _text_ner_analysis(nlp_tools=nlp_tools, text=post_text)
    _print_ner_results(unpacked_results)


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

    _title_analysis(nlp_tools=nlp_tools, post=post)
    _post_analysis(nlp_tools=nlp_tools, post=post)


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

    _title_analysis(nlp_tools=nlp_tools, post=post)
    _post_analysis(nlp_tools=nlp_tools, post=post)
    _root_comment_analysis(nlp_tools=nlp_tools, post=post)


def _all_comments_analysis(post: Type[praw.Reddit.post], nlp_tools: Type[NlpTools]):
    _sentiment_analysis_all_comments(post=post, nlp_tools=nlp_tools)


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

    _title_analysis(nlp_tools=nlp_tools, post=post)
    _post_analysis(nlp_tools=nlp_tools, post=post)
    _all_comments_analysis(nlp_tools=nlp_tools, post=post)
