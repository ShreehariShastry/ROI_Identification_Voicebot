import os
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import sounddevice as sd
import wavio
import textwrap
import threading
import speech_recognition as sr
import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import pyglet
import joblib

# import warnings
# warnings.filterwarnings("ignore")


# Create a recognizer instance
recognizer_instance = sr.Recognizer()


def evaluate_model_new(svm_classifier, X_test):
    y_pred = svm_classifier.predict(X_test)
    return y_pred


def load_svm_model(load_model_path):
    return joblib.load(load_model_path)


def play_mp3(file_path):
    music = pyglet.resource.media(file_path)
    music.play()
    # # Keep the program running while the music plays
    # pyglet.app.run()


def text_to_speech(text, output_mp3):
    tts = gTTS(text)
    tts.save(output_mp3)


def preprocess_sentence(sentence):
    # Remove punctuation and convert to lowercase
    translator = str.maketrans('', '', string.punctuation)
    processed_sentence = sentence.translate(translator).lower().strip()
    return processed_sentence


def find_most_similar_sentence(target_sentence, sentence_list):
    # Preprocess the target sentence
    target_sentence = preprocess_sentence(target_sentence)

    # Preprocess each sentence in the list and convert to a common data type
    preprocessed_sentence_list = [sentence for sentence in sentence_list]
    preprocessed_sentence_list = np.array(preprocessed_sentence_list, dtype=object)

    # Combine the target sentence and preprocessed sentence list into a single array
    all_sentences = np.concatenate(([target_sentence], preprocessed_sentence_list))

    # Create a CountVectorizer to convert sentences into BoW representations
    vectorizer = CountVectorizer()

    # Calculate the BoW matrix for all sentences
    bow_matrix = vectorizer.fit_transform(all_sentences)

    # Get the BoW representation of the target sentence
    target_bow = bow_matrix[0]

    # Calculate the cosine similarity between the target sentence and each sentence in the list
    similarity_scores = cosine_similarity(target_bow, bow_matrix[1:])

    # Find the index of the sentence with the highest similarity score
    most_similar_index = similarity_scores.argmax()

    # Return the most similar sentence
    return sentence_list[most_similar_index], most_similar_index


# Function to get the screen width
def get_screen_width(root):
    return root.winfo_screenwidth()


class VoiceRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Recorder")
        self.root.geometry("800x400")  # Set the window size
        self.root.configure(bg="#F0F0F0")  # Set the background color

        # Define custom styles for themed buttons
        self.style = ttk.Style()
        self.style.configure("Start.TButton", foreground="green", font=("Arial", 16))
        self.style.configure("Stop.TButton", foreground="red", font=("Arial", 16))
        self.style.configure("Process.TButton", foreground="purple", font=("Arial", 16))

        self.recording_var = tk.BooleanVar()
        self.recording_var.set(False)

        # Center the buttons on the screen
        frame = tk.Frame(self.root, bg="#F0F0F0")
        frame.pack(expand=True)

        self.record_button = ttk.Button(frame, text="Start Recording", style="Start.TButton",
                                        command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=20, pady=20)

        self.voice_button = ttk.Button(frame, text="Process", style="Process.TButton", command=self.convert_to_text)
        self.voice_button.pack(side=tk.LEFT, padx=20, pady=20)

        # Add Output Text Boxes
        self.output1_text = tk.Text(self.root, height=5, width=50, wrap=tk.WORD)
        self.output1_text.pack(pady=10)

    def toggle_recording(self):
        # Delete the 'output.mp3' file if it exists
        if os.path.exists("output.mp3"):
            os.remove("output.mp3")
        self.start_recording()

    def clear_text_boxes(self):
        self.output1_text.delete(1.0, tk.END)

    def start_recording(self):
        self.recording_var.set(True)

        # Clear the contents of the text boxes
        self.clear_text_boxes()

        threading.Thread(target=self.record_audio).start()

    def record_audio(self):
        samplerate = 44100  # You can adjust the samplerate (samples per second)
        duration = 5  # You can adjust the recording duration (in seconds)
        recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()

        # Save the recording to a WAV file
        filename = "user_voice.wav"
        wavio.write(filename, recording, samplerate, sampwidth=2)

        print(f"Recording saved as {filename}")

    def convert_to_text(self):
        filename = "user_voice.wav"
        Indent = np.load('Indents.npy', allow_pickle=True)
        Target = np.load('Targets.npy', allow_pickle=True)
        sum_along_axis_1 = np.sum(Target, axis=0)
        ind = np.where(sum_along_axis_1 == max(sum_along_axis_1))

        title = "."
        messagebox.showinfo(title, 'The most frequently asked\nQuery is:'+Indent[ind[0][0]])

        print('Processing.....')
        # Read the audio file
        with sr.AudioFile(filename) as audio_file:
            # Adjust for ambient noise, if necessary
            recognizer_instance.adjust_for_ambient_noise(audio_file)

            # Read the audio data from the file
            audio_data = recognizer_instance.record(audio_file)

            try:
                Answer = np.load('Answer.npy', allow_pickle=True)
                # Recognize the speech using the default API (Google Web Speech API)
                text = recognizer_instance.recognize_google(audio_data)

                sentence_list = np.load('Preprocessed.npy', allow_pickle=True)
                most_similar_sentence, ind = find_most_similar_sentence(text, sentence_list)
                # print(most_similar_sentence)
                # ind = np.where(sentence_list == most_similar_sentence)
                out_answer = Answer[ind]
                file_path = "output.mp3"

                text_to_speech(out_answer, file_path)

                self.output1_text.delete(1.0, tk.END)
                # self.output1_text.insert(tk.END, f"Most similar is : {out_answer}")
                print('Done')
                self.output1_text.insert(tk.END, f"{out_answer}")

                play_mp3(file_path)

            except sr.UnknownValueError:
                messagebox.showerror("Speech recognition could not understand the audio.")
            # print("Speech recognition could not understand the audio.")
            except sr.RequestError as e:
                messagebox.showerror("Error during speech recognition; {0}".format(e))
                # print("Error during speech recognition; {0}".format(e))


def GUI():
    root = tk.Tk()
    app = VoiceRecorderApp(root)

    # Get the screen width and window width
    screen_width = get_screen_width(root)
    window_width = 800

    # Set the window position to the top-right corner
    x_position = screen_width - window_width
    root.geometry(f"800x400+{x_position}+0")
    root.mainloop()



if __name__ == "__main__":
    GUI()
