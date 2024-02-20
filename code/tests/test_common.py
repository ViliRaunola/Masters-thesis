import datetime
import unittest

import utility.common as common


class TestUtility(unittest.TestCase):

    def test_create_unique_file_name(self):
        file_name = "test.txt"
        ct = datetime.datetime.now()
        expected_file_name = f"test_{ct.day:02d}_{ct.month:02d}_{ct.year:04d}_{ct.hour:02d}_{ct.minute:02d}_{ct.second:02d}.txt"
        file_name = common.create_unique_file_name(file_name)
        self.assertEqual(expected_file_name, file_name)
