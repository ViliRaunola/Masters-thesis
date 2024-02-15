import re


def remove_url(text: str):
    regex_pattern = re.compile(pattern="https?:[^\s]+")
    return regex_pattern.sub(r"", text)


def remove_reddit_quotation(text: str):
    regex_pattern = re.compile(pattern="\s*(?=>)(?!\s*>)|^>")
    return regex_pattern.sub(r"", text)


# Source: https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emojis(text: str):
    regrex_pattern = re.compile(
        pattern="["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+",
        flags=re.UNICODE,
    )

    return regrex_pattern.sub(r"", text)
