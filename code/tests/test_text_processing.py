import unittest

import utility.text_processing as text_processing


class TestTestProcessing(unittest.TestCase):

    def test_remove_url(self):
        string = "Tässä on eka url https://www.reddit.com/r/Suomi/comments/1ar8957/osa_toistaa_putinin_propagandaa_yle_selvitti_mit%C3%A4/ ja toka https://www.reddit.com/r/Suomi/comments/1ar8957/osa_toistaa_putinin_propagandaa_yle_selvitti_mit%C3%A4/"
        string_after = "Tässä on eka url  ja toka "
        string = text_processing.remove_url(string)
        self.assertEqual(string, string_after)

    def test_remove_emoji(self):
        string = "Asia 1 ja 2. 😭😭 Yawn, next"
        string_after = "Asia 1 ja 2.  Yawn, next"
        string = text_processing.remove_emojis(string)
        self.assertEqual(string, string_after)
