import pyaudio
import boto3
import json
import sys
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


def speak_response(text, polly_client):
    try:
        # Request speech synthesis
        response = polly_client.synthesize_speech(
            Text=text, OutputFormat="pcm", VoiceId="Joanna", Engine="neural"
        )

        if "AudioStream" in response:
            from pygame import mixer

            # Create a temporary wave file
            with wave.open("response.wav", "wb") as wav_file:
                wav_file.setnchannels(1)  # mono
                wav_file.setsampwidth(2)  # 2 bytes per sample
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes(response["AudioStream"].read())

            print("\nPress Enter to stop the voice playback...")

            # Flag for controlling playback
            should_stop = threading.Event()

            def wait_for_input():
                input()  # Wait for Enter key
                should_stop.set()
                mixer.music.stop()

            # Play the audio
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

            mixer.quit()

            # Clean up the temporary file
            os.remove("response.wav")

    except Exception as e:
        print(f"Error in text-to-speech: {e}")


class TranscriptHandler(TranscriptResultStreamHandler):
    def __init__(self, bedrock_runtime, transcript_result_stream, polly_client):
        super().__init__(transcript_result_stream)
        self.bedrock_runtime = bedrock_runtime
        self.polly_client = polly_client
        self.listening = True  # Always listening
        self.polly_finished = threading.Event()

    def speak_response(self, text):
        mixer = None
        mixer_lock = threading.Lock()
        try:
            response = self.polly_client.synthesize_speech(
                Text=text, OutputFormat="pcm", VoiceId="Joanna", Engine="neural"
            )

            if "AudioStream" in response:
                from pygame import mixer

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

                        # Get Claude's response
                        body = json.dumps(
                            {
                                "anthropic_version": "bedrock-2023-05-31",
                                "max_tokens": 1000,
                                "messages": [{"role": "user", "content": transcript}],
                            }
                        )

                        response = (
                            self.bedrock_runtime.invoke_model_with_response_stream(
                                modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                                body=body,
                                contentType="application/json",
                                accept="application/json",
                            )
                        )

                        print("\nClaude: ", end="", flush=True)
                        full_response = ""
                        response_stream = response.get("body")
                        if response_stream:
                            for event in response_stream:
                                try:
                                    chunk_bytes = event.get("chunk", {}).get(
                                        "bytes", b""
                                    )
                                    if chunk_bytes:
                                        chunk = json.loads(chunk_bytes.decode("utf-8"))
                                        if chunk.get("type") == "content_block_delta":
                                            text = chunk.get("delta", {}).get(
                                                "text", ""
                                            )
                                            full_response += text
                                            print(text, end="", flush=True)
                                except Exception as chunk_error:
                                    print(f"\nError processing chunk: {chunk_error}")
                                    continue

                        print("\n")  # New line after response

                        # Speak the response
                        self.speak_response(full_response)

                        print("-" * 50 + "\n")

                        # Resume listening
                        self.listening = True
                        print("Listening... (Press Ctrl+C to stop)\n")
                        return True

                    except Exception as e:
                        print(f"\nError: {e}")
                        self.listening = True
                        print("Listening... (Press Ctrl+C to stop)\n")
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
    transcribe_client = None
    bedrock_runtime = None
    polly_client = None
    audio = None
    audio_stream = None
    tasks = []

    try:
        print("Initializing services...")
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

        print("\nListening... Press Ctrl+C to stop.")
        print("You can start speaking now!\n")

        stream_config = {
            "media_sample_rate_hz": RATE,
            "media_encoding": "pcm",
            "language_code": "en-US",
            "enable_partial_results_stabilization": True,
        }

        stream = await transcribe_client.start_stream_transcription(**stream_config)

        handler = TranscriptHandler(bedrock_runtime, stream.output_stream, polly_client)
        handler_task = asyncio.create_task(handler.handle_events())
        writer_task = asyncio.create_task(write_chunks(stream, audio_stream))

        tasks = [handler_task, writer_task]

        await asyncio.gather(*tasks)

    except KeyboardInterrupt:
        print("\nGracefully shutting down...")
    except Exception as e:
        print(f"\nError: {e}")
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
