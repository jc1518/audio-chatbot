import threading
import sys
import wave
import os
import random
import json
import time

from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

from lib.web_search import web_search

# Bedrock settings
MODELS = [
    # "us.amazon.nova-micro-v1:0",
    # "us.amazon.nova-lite-v1:0",
    # "us.amazon.nova-pro-v1:0",
    # "anthropic.claude-3-5-haiku-20241022-v1:0",
    # "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
]

INFERENCE_CONFIG = {
    "maxTokens": 1000,
    "temperature": 0.2,
    "topP": 0.9,
}

SYSTEM = [
    {
        "text": """You are a highly intelligent and engaging virtual assistant designed to assist users with a wide range of inquiries and tasks.
          Your primary goal is to provide accurate, informative, concise and helpful responses while ensuring a positive user experience. Keep your
          answer within 5 sentences."""
    }
]

TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "web_search",
                "description": "Perform a web search to find relevant information",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "The maximum number of results to return",
                            },
                        },
                        "required": ["query"],
                    }
                },
            }
        },
    ]
}


class NullDevice:
    def write(self, *args, **kwargs):
        pass

    def flush(self):
        pass


class TranscriptHandler(TranscriptResultStreamHandler):
    def __init__(
        self,
        bedrock_runtime,
        transcript_result_stream,
        polly_client,
        language_code,
        converstation_history,
    ):

        super().__init__(transcript_result_stream)
        self.bedrock_runtime = bedrock_runtime
        self.polly_client = polly_client
        self.language_code = language_code
        self.listening = True  # Always listening
        self.polly_finished = threading.Event()
        self.conversation_history = (
            converstation_history  # Initialize conversation history
        )

    def speak_response(self, text):
        mixer = None
        mixer_lock = threading.Lock()
        if self.language_code == "zh-CN":
            voice_id = "Zhiyu"
            engine = "neural"
        else:
            voice_id = "Joanna"
            engine = "generative"
        try:
            response = self.polly_client.synthesize_speech(
                Text=text, OutputFormat="pcm", VoiceId=voice_id, Engine=engine
            )

            if "AudioStream" in response:
                original_stdout = sys.stdout
                sys.stdout = NullDevice()
                from pygame import mixer

                sys.stdout = original_stdout

                with wave.open("response.wav", "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16000)
                    wav_file.writeframes(response["AudioStream"].read())

                print("\nPress Enter to stop the voice playback...")

                # Flag for controlling playback
                should_stop = threading.Event()

                def wait_for_input():
                    try:
                        input()  # Wait for Enter key
                        should_stop.set()
                        with mixer_lock:
                            if mixer and mixer.get_init():
                                mixer.music.stop()
                    except Exception as e:
                        print(f"\nError stopping playback: {e}")

                # Initialize mixer
                with mixer_lock:
                    mixer.init()
                    mixer.music.load("response.wav")
                    mixer.music.play()

                # Start input thread
                input_thread = threading.Thread(target=wait_for_input)
                input_thread.daemon = True
                input_thread.start()

                # Wait for audio to finish or stop signal
                while mixer.music.get_busy() and not should_stop.is_set():
                    time.sleep(0.1)

                if should_stop.is_set():
                    print("\nVoice playback stopped.")

                # Clean up
                with mixer_lock:
                    if mixer and mixer.get_init():
                        mixer.quit()

                # Remove temporary file
                try:
                    os.remove("response.wav")
                except Exception as e:
                    print(f"Error removing temporary file: {e}")

        except Exception as e:
            print(f"Error in text-to-speech: {e}")
        finally:
            # Final cleanup
            try:
                with mixer_lock:
                    if mixer and mixer.get_init():
                        mixer.quit()
            except Exception:
                pass
            self.polly_finished.set()

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results

        for result in results:
            if result.alternatives and self.listening:
                transcript = result.alternatives[0].transcript
                if result.is_partial:
                    print(f"\rUser: {transcript}", end="", flush=True)
                else:
                    print(f"\rUser: {transcript}")  # Final transcript

                    try:
                        # Process response
                        self.listening = False  # Temporarily stop listening
                        self.polly_finished.clear()

                        # Add user message to conversation history
                        self.conversation_history.append(
                            {"role": "user", "content": [{"text": transcript}]}
                        )  # Wrap in list

                        modelId = random.choice(MODELS)
                        print(f"\n\r(calling {modelId})")
                        # Include conversation history in messages
                        messages = self.conversation_history.copy()  # Copy the history
                        response = self.bedrock_runtime.converse_stream(
                            modelId=modelId,
                            inferenceConfig=INFERENCE_CONFIG,
                            system=SYSTEM,
                            messages=messages,
                        )

                        print("\nAssistant: ", end="", flush=True)
                        full_response = ""
                        response_stream = response.get("stream")
                        if response_stream:
                            for event in response_stream:
                                try:
                                    if "contentBlockDelta" in event:
                                        text = event["contentBlockDelta"]["delta"][
                                            "text"
                                        ]
                                        full_response += text
                                        print(text, end="", flush=True)
                                except Exception as chunk_error:
                                    print(f"\nError processing chunk: {chunk_error}")
                                    continue

                        print("\n")  # New line after response

                        # Add assistant response to conversation history
                        self.conversation_history.append(
                            {"role": "assistant", "content": [{"text": full_response}]}
                        )  # Wrap in list

                        # Speak the response
                        self.speak_response(full_response)

                        print("-" * 50 + "\n")

                        # Resume listening
                        self.listening = True
                        print(
                            "Listening... You can start speaking now! (Press Ctrl+C to stop)\n"
                        )
                        return True

                    except Exception as e:
                        print(f"\nError: {e}")
                        self.listening = True
                        print(
                            "Listening... You can start speaking now! (Press Ctrl+C to stop)\n"
                        )
                        return False
