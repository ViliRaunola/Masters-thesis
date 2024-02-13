import os
import re

import dotenv
import praw

import text_process


def create_praw_instance():
    dotenv.load_dotenv()

    reddit = praw.Reddit(
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),
        user_agent=os.getenv("USER_AGENT"),
    )

    return reddit


def get_post_by_url(reddit, url):
    post = reddit.submission(url=url)
    return post


# reddit = create_praw_instance()

# url = "https://www.reddit.com/r/Suomi/comments/1apt9gp/ty%C3%B6paikkakiusaaminen/"

# post = get_post_by_url(reddit=reddit, url=url)

text = "Ukraina ja georgia. 😭😭 Kaks maata jotka ovat todella hyviä internet-sympatian saamisessa ja pillittämisessä muuttamatta tosiasiaa. Yawn, next"
text = text_process.remove_emojis(text)

text2 = ">Eikö nää tyypit tajua"
text2 = text_process.remove_reddit_quotation(text2)

text3 = "[Linkki](https://twitter.com/Ukraine/status/1468206078940823554?t=MOgBs7L-AbAsktl8pgcSKQ&s=19)"
text3 = text_process.remove_url(text3)

print(text)
print(text2)
print(text3)
