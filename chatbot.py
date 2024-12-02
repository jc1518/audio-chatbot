import pyaudio
import boto3
import json
import sys
import asyncio
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from botocore.exceptions import BotoCoreError, ClientError
import itertools
import threading
import time

# Audio input configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000


def spinning_cursor():
    spinner = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
    while True:
        yield next(spinner)


class TranscriptHandler(TranscriptResultStreamHandler):
    def __init__(self, bedrock_runtime, transcript_result_stream):
        super().__init__(transcript_result_stream)
        self.bedrock_runtime = bedrock_runtime
        self.spinner = spinning_cursor()
        self.spinner_running = False

    def start_spinner(self):
        self.spinner_running = True
        while self.spinner_running:
            print(f"\rProcessing your input {next(self.spinner)}", end="")
            time.sleep(0.1)

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results

        for result in results:
            if result.alternatives:
                transcript = result.alternatives[0].transcript
                if not result.is_partial:
                    print(f"\nUser: {transcript}")
                    spinner_thread = threading.Thread(target=self.start_spinner)
                    spinner_thread.start()

                    try:
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

                        # Stop spinner
                        self.spinner_running = False
                        spinner_thread.join()
                        print("\r", end="")  # Clear the spinner line

                        print("\nClaude: ", end="", flush=True)
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
                                            print(text, end="", flush=True)
                                except Exception as chunk_error:
                                    print(f"\nError processing chunk: {chunk_error}")
                                    continue

                        print("\n")  # New line after response
                        print("-" * 50 + "\n")

                    except Exception as e:
                        self.spinner_running = False
                        spinner_thread.join()
                        print("\r", end="")  # Clear the spinner line
                        print(f"\nError: {e}")
                        # Print full error details for debugging
                        import traceback

                        print(traceback.format_exc())


async def write_chunks(stream, audio_stream):
    while True:
        try:
            data = audio_stream.read(CHUNK, exception_on_overflow=False)
            await stream.input_stream.send_audio_event(audio_chunk=data)
        except Exception as e:
            print(f"Error writing chunks: {e}")
            break


async def main():
    # Initialize AWS clients
    transcribe_client = None
    bedrock_runtime = None
    audio = None
    audio_stream = None

    try:
        transcribe_client = TranscribeStreamingClient(region="us-west-2")
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime", region_name="us-west-2"
        )

        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        audio_stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        print("Listening... Press Ctrl+C to stop.")
        print("You can start speaking now!\n")

        # Start transcription stream
        stream_config = {
            "media_sample_rate_hz": RATE,
            "media_encoding": "pcm",
            "language_code": "en-US",
            "enable_partial_results_stabilization": True,
        }

        stream = await transcribe_client.start_stream_transcription(**stream_config)

        # Create and start the handler task
        handler = TranscriptHandler(bedrock_runtime, stream.output_stream)
        handler_task = asyncio.create_task(handler.handle_events())

        # Start the audio streaming task
        writer_task = asyncio.create_task(write_chunks(stream, audio_stream))

        # Wait for both tasks to complete
        await asyncio.gather(handler_task, writer_task)

    except KeyboardInterrupt:
        print("\nGracefully shutting down...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # Cleanup
        if audio_stream:
            audio_stream.stop_stream()
            audio_stream.close()
        if audio:
            audio.terminate()
        print("\nThank you for using the chatbot!")


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        loop.close()
