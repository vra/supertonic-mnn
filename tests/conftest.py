import sys
from unittest.mock import MagicMock

# Global mocks for dependencies that might not be present in the test environment
if 'MNN' not in sys.modules:
    mnn_mock = MagicMock()
    sys.modules['MNN'] = mnn_mock
