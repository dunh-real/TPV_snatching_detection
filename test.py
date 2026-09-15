"""Trình chạy kiểm thử của dự án dùng trong lệnh khởi động nhanh ở README."""

import sys
import unittest


def main() -> int:
    suite = unittest.defaultTestLoader.discover(".", pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
