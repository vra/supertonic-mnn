from supertonic_mnn import SupertonicTTS
import sounddevice as sd
import numpy as np

def read_file_to_string(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            # Read the entire file content into a single string variable
            file_contents = file.read()
        return file_contents
    except FileNotFoundError:
        return f"Error: The file '{filename}' was not found."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

file_name = 'tts.txt'
data_string = read_file_to_string(file_name)

# Initialize
tts = SupertonicTTS()

# Synthesize using Stream
# Models will be downloaded automatically if not present
print(f"Synthesizing and playing text from {file_name}...")

import queue
import threading

# Queue to hold audio chunks
audio_queue = queue.Queue()
playback_finished = threading.Event()

def playback_thread_func(sample_rate):
    stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype='float32')
    stream.start()
    
    try:
        while True:
            chunk = audio_queue.get()
            if chunk is None:  # Sentinel value to signal end of stream
                break

            # Prepare data and write to stream (this call blocks if buffer full)
            audio_data = chunk.flatten().astype('float32')
            stream.write(audio_data)
    except Exception as e:
        print(f"Playback error: {e}")
    finally:
        stream.stop()
        stream.close()
        playback_finished.set()

stream_gen = tts.synthesize_stream(data_string, voice="F1", steps=9)
audio_thread = None

try:
    first_chunk = True
    for audio_chunk, sample_rate in stream_gen:
        # On first chunk, start the playback thread
        if first_chunk:
            audio_thread = threading.Thread(target=playback_thread_func, args=(sample_rate,))
            audio_thread.daemon = True # Daemon thread dies if main program exits
            audio_thread.start()
            print("Playback started...")
            first_chunk = False
        
        # Put generated chunk into the queue for the consumer thread
        audio_queue.put(audio_chunk)

    # Signal end of playback
    if audio_thread:
        audio_queue.put(None)
        # Wait for playback to finish
        playback_finished.wait()
        print("Playback finished.")

except Exception as e:
    print(f"An error occurred: {e}")
    # Ensure thread exits if there's a main thread error
    audio_queue.put(None)

