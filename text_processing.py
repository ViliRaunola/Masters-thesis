import re


def remove_url(text: str):
    regex_pattern = re.compile(pattern="https?:[^\s]+")
    return regex_pattern.sub(r"", text)


def remove_reddit_quotation(text: str):
    regex_pattern = re.compile(pattern="\s*(?=>)(?!\s*>)|^>")
    return regex_pattern.sub(r"", text)


# Source: https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
# / https://stackoverflow.com/questions/69554621/remove-emoji-from-string-doesnt-works-for-some-cases
def remove_emojis(text: str):
    regrex_pattern = re.compile(
        pattern="["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U0001F1F2"
        "\U0001F1F4"
        "\U0001F620"
        "\u200d"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "\U00002500-\U00002BEF"  # Chinese char
        "\U00010000-\U0010ffff"
        "]+",
        flags=re.UNICODE,
    )

    return regrex_pattern.sub(r"", text)
