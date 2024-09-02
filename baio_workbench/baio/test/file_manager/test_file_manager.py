import os
import shutil
import tempfile
import unittest

from baio.st_app.file_manager import FileManager


class TestFileManager(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()

        # Create a test file
        self.test_file_path = os.path.join(self.test_dir, "test_file.txt")
        with open(self.test_file_path, "w") as f:
            f.write("Test content")

        # Create a FileManager instance
        self.file_manager = FileManager(self.test_dir)

    def tearDown(self):
        # Clean up the temporary directory after the test
        shutil.rmtree(self.test_dir)

    def test_file_download_button(self):
        button_html = self.file_manager.file_download_button(self.test_file_path)

        # Check if button_html is not None
        self.assertIsNotNone(
            button_html, "file_download_button returned None for a file"
        )

        # Check if the generated HTML contains expected elements
        self.assertIn('href="data:file/octet-stream;base64,', button_html)
        self.assertIn(f'download="test_file.txt"', button_html)
        self.assertIn('style="display:inline-block;', button_html)
        self.assertIn("Download", button_html)

    def test_directory_download_button(self):
        button_html = self.file_manager.file_download_button(self.test_dir)

        # Check if button_html is not None
        self.assertIsNotNone(
            button_html, "file_download_button returned None for a directory"
        )

        # Check if the generated HTML contains expected elements
        self.assertIn('onclick="downloadDirectory(', button_html)
        self.assertIn(".zip", button_html)
        self.assertIn('style="display:inline-block;', button_html)
        self.assertIn("Download", button_html)


if __name__ == "__main__":
    unittest.main()
