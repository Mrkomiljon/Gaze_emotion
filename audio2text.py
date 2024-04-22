import speech_recognition as sr
import pyttsx3
import time

# Initialize the recognizer
r = sr.Recognizer()

# Function to convert text to speech
def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

# Loop infinitely for user to speak
while True:
    try:
        st = time.time()
        # Use the microphone as source for input.
        with sr.Microphone() as source:
            # Adjust the energy threshold based on the surrounding noise level
            r.adjust_for_ambient_noise(source, duration=0.2)
            print("Speak Now...")
            # Listen for the user's input
            audio = r.listen(source, timeout=5)

            # Using Google to recognize audio
            recognized = r.recognize_google(audio, show_all=True)

            if recognized:
                # Get the most confident result
                best_guess = recognized['alternative'][0]['transcript']
                best_guess = best_guess.lower()
                print("You said:", best_guess)
                SpeakText(best_guess)
            print(time.time() - st)

    except sr.WaitTimeoutError:
        print("Timeout! No speech detected.")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
    except sr.UnknownValueError:
        print("Sorry, I did not understand what you said.")
