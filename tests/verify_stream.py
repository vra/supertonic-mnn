
import time
import numpy as np
from supertonic_mnn import SupertonicTTS

def read_file_to_string(filename):
    """
    Reads the entire content of a specified text file into a single string.
    
    Args:
        filename (str): The path to the text file.

    Returns:
        str: The content of the file, or an error message if the file is not found.
    """
    try:
        # 'with open(...) as file:' ensures the file is automatically closed
        # 'r' mode opens the file for reading (default)
        with open(filename, 'r', encoding='utf-8') as file:
            # Read the entire file content into a single string variable
            file_contents = file.read()
        return file_contents
    except FileNotFoundError:
        return f"Error: The file '{filename}' was not found."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def test_streaming():
    print("Initializing SupertonicTTS...")
    tts = SupertonicTTS()
    
    # A text long enough to be chunked (punctuation usually triggers chunking)
    file_name = "tts.txt"
    text = read_file_to_string(file_name)
    
    print("\n--- Starting Stream Synthesis ---")
    print(f"Text: {text}")
    
    start_time = time.time()
    chunk_count = 0
    total_audio_len = 0
    
    try:
        # Use the NEW synthesize_stream method
        stream = tts.synthesize_stream(text, voice="F2", steps=9) # Low steps for speed
        
        for audio_chunk, sr in stream:
            chunk_count += 1
            chunk_duration = audio_chunk.shape[0] / sr
            total_audio_len += chunk_duration
            
            elapsed = time.time() - start_time
            print(f"Received Chunk #{chunk_count}: Duration={chunk_duration:.2f}s, Time from start={elapsed:.2f}s")
            
            # Simple check to ensure we got valid data
            if not isinstance(audio_chunk, np.ndarray):
                print("ERROR: Chunk is not a numpy array!")
            if sr <= 0:
                print("ERROR: Invalid sample rate!")

        print("\n--- Stream Finished ---")
        print(f"Total Chunks: {chunk_count}")
        print(f"Total Audio Duration: {total_audio_len:.2f}s")
        
        if chunk_count > 1:
            print("SUCCESS: Multiple chunks received, streaming is working effectively.")
        else:
            print("WARNING: Only 1 chunk received. While valid, it doesn't demonstrate streaming benefits (text might be too short).")

    except Exception as e:
        print(f"\nFATAL ERROR during streaming: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_streaming()
