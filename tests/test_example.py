
import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

@pytest.fixture
def mock_dependencies_example():
    with patch('supertonic_mnn.wrapper.ensure_models') as mock_ensure, \
         patch('supertonic_mnn.wrapper.load_text_to_speech') as mock_load_tts, \
         patch('supertonic_mnn.wrapper.load_voice_style') as mock_load_style, \
         patch('supertonic_mnn.wrapper.get_voice_style_path') as mock_get_style_path, \
         patch('supertonic_mnn.wrapper.SupertonicTTS.save') as mock_save:

        mock_tts_engine = MagicMock()
        mock_tts_engine.sample_rate = 24000
        # Return (wav, key, rtf)
        # Note: wrapper.py usage: wav, duration, rtf = engine(...)
        # engine returns [array], array, float
        # wav_data = wav[0]
        # So mocks should return ([data], data, rtf)
        import numpy as np
        dummy_wav = np.zeros(100)
        mock_tts_engine.return_value = ([dummy_wav], np.zeros(1), 0.1)
        mock_load_tts.return_value = mock_tts_engine

        yield {
            'ensure': mock_ensure,
            'load_tts': mock_load_tts,
            'tts': mock_tts_engine,
            'save': mock_save
        }

def test_basic_usage_example(mock_dependencies_example):
    # Import the example module. 
    # Since basic_usage.py is a script, we can import it and run main() if it has one.
    # But basic_usage.py adds to sys.path which might be messy.
    # Instead, let's just test the logic by writing a similar test here 
    # or invoking main from the file if we can import it.
    
    # We dynamically load the module to avoid path issues
    import importlib.util
    spec = importlib.util.spec_from_file_location("basic_usage", os.path.join(os.path.dirname(__file__), "../examples/basic_usage.py"))
    basic_usage = importlib.util.module_from_spec(spec)
    sys.modules["basic_usage"] = basic_usage
    spec.loader.exec_module(basic_usage)
    
    # Run main
    basic_usage.main()
    
    # Verify calls
    mock_dependencies_example['ensure'].assert_called()
    mock_dependencies_example['load_tts'].assert_called()
    mock_dependencies_example['tts'].assert_called()
    # The example calls output_file="output_simple.wav", so save should be called.
    mock_dependencies_example['save'].assert_called()
