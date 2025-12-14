from supertonic_mnn import SupertonicTTS

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

# Synthesize
# Models will be downloaded automatically if not present
audio, sample_rate = tts.synthesize(data_string, voice="F1", output_file="hello.wav", steps=9)

