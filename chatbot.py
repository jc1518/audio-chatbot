import pyaudio
import boto3
import json
import sys
import random
import asyncio
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from botocore.exceptions import BotoCoreError, ClientError
import threading
import time
import os
import wave

# Audio input configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000


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
        else:
            voice_id = "Joanna"
        try:
            response = self.polly_client.synthesize_speech(
                Text=text, OutputFormat="pcm", VoiceId=voice_id, Engine="neural"
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

                        inferenceConfig = {
                            "maxTokens": 1000,
                            "temperature": 0.2,
                            "topP": 0.9,
                        }

                        system = [
                            {
                                "text": """You are a highly intelligent and engaging virtual assistant designed to assist users with a wide range of inquiries and tasks.
                                  Your primary goal is to provide accurate, informative, concise and helpful responses while ensuring a positive user experience. Keep your
                                  answer within 5 sentences."""
                            }
                        ]

                        models = [
                            "us.amazon.nova-micro-v1:0",
                            "us.amazon.nova-lite-v1:0",
                            "us.amazon.nova-pro-v1:0",
                            "anthropic.claude-3-5-haiku-20241022-v1:0",
                            "anthropic.claude-3-5-sonnet-20240620-v1:0",
                            "anthropic.claude-3-5-sonnet-20241022-v2:0",
                        ]

                        modelId = random.choice(models)
                        print(f"\n\r(calling {modelId})")
                        # Include conversation history in messages
                        messages = self.conversation_history.copy()  # Copy the history
                        response = self.bedrock_runtime.converse_stream(
                            modelId=modelId,
                            inferenceConfig=inferenceConfig,
                            system=system,
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


async def write_chunks(stream, audio_stream):
    try:
        while True:
            data = audio_stream.read(CHUNK, exception_on_overflow=False)
            await stream.input_stream.send_audio_event(audio_chunk=data)
    except asyncio.CancelledError:
        # Handle task cancellation gracefully
        raise
    except Exception as e:
        print(f"Error writing chunks: {e}")


async def main():
    supported_languages = {
        "1": "en-US",
        "2": "zh-CN",
        # "3": "ja-JP",
        # "4": "es-ES",
    }

    # Prompt user to select a language
    print("Select a language for transcription:")
    for key, value in supported_languages.items():
        if value == "en-US":
            print(f"{key}: English")
        if value == "zh-CN":
            print(f"{key}: 中文")

    language_choice = input("Enter the number corresponding to your choice: ")
    selected_language = supported_languages.get(language_choice, "en-US")
    conversation_history = []  # Initialize conversation history
    while True:  # Loop to allow restarting on timeout
        transcribe_client = None
        bedrock_runtime = None
        polly_client = None
        audio = None
        audio_stream = None
        tasks = []

        try:

            transcribe_client = TranscribeStreamingClient(region="us-west-2")
            bedrock_runtime = boto3.client(
                service_name="bedrock-runtime", region_name="us-west-2"
            )
            polly_client = boto3.client("polly", region_name="us-west-2")

            audio = pyaudio.PyAudio()
            audio_stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )

            print("Listening... You can start speaking now! (Press Ctrl+C to stop)\n")

            # Update stream_config with the selected language
            stream_config = {
                "media_sample_rate_hz": RATE,
                "media_encoding": "pcm",
                "language_code": selected_language,  # Use the selected language
                "enable_partial_results_stabilization": True,
            }

            stream = await transcribe_client.start_stream_transcription(**stream_config)

            handler = TranscriptHandler(
                bedrock_runtime,
                stream.output_stream,
                polly_client,
                selected_language,
                conversation_history,
            )
            handler_task = asyncio.create_task(handler.handle_events())
            writer_task = asyncio.create_task(write_chunks(stream, audio_stream))

            tasks = [handler_task, writer_task]

            await asyncio.gather(*tasks)

        except KeyboardInterrupt:
            break  # Exit the loop on manual interrupt
        except Exception as e:
            print(f"\nError: {e}")
            # Optionally, you can add a delay before restarting
            await asyncio.sleep(1)  # Delay before restarting
        finally:
            # Cancel tasks first
            for task in tasks:
                if not task.done():
                    task.cancel()

            # Wait for tasks to complete their cancellation
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Clean up audio resources
            if audio_stream:
                audio_stream.stop_stream()
                audio_stream.close()
            if audio:
                audio.terminate()

            print("\nThank you for using the chatbot!")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass  # Already handled in main()
    finally:
        try:
            # Clean up any remaining tasks
            tasks = asyncio.all_tasks(loop)
            for task in tasks:
                task.cancel()

            # Give tasks a chance to complete
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        except Exception:
            pass
        finally:
            loop.close()
