import requests
import os
import unittest
from pathlib import Path

API_URL = "http://127.0.0.1:5000/analyze"

class TestMalwareAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create test files before running tests"""
        # Create a valid small EXE (dummy file)
        Path("valid.exe").write_bytes(os.urandom(1024))  # 1KB
        # Create an oversized EXE (401MB)
        Path("huge.exe").write_bytes(os.urandom(401 * 1024 * 1024)) 
        Path("invalid.txt").write_text("Not a PE file")

    def test_valid_exe(self):
        """Test with valid small EXE"""
        with open("valid.exe", "rb") as f:
            response = requests.post(API_URL, files={"file": f})
        
        self.assertEqual(response.status_code, 200)
        json = response.json()
        self.assertIn("score", json)
        self.assertIn("label", json)
        print(f"✓ Valid EXE test passed (score: {json['score']})")

    def test_invalid_extension(self):
        """Test with non-EXE/DLL file"""
        with open("invalid.txt", "rb") as f:
            response = requests.post(API_URL, files={"file": f})
        
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid file type", response.json()["error"])
        print("✓ Invalid extension test passed")

    def test_oversize_file(self):
        """Test with EXE >400MB"""
        file_size = 401 * 1024 * 1024
        with open("huge.exe", "rb") as f:
            response = requests.post(
                API_URL,
                files={"file": ("huge.exe", f, "application/octet-stream")},
                headers={"Content-Length": str(file_size)}  # Force-set size
            )
        
        self.assertEqual(response.status_code, 400)
        self.assertIn("File too large", response.json()["error"])
        print("✓ Oversize file test passed")

    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        for f in ["valid.exe", "huge.exe", "invalid.txt"]:
            Path(f).unlink(missing_ok=True)

if __name__ == "__main__":
    unittest.main(verbosity=2)