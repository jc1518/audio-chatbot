import asyncio

import pyaudio
import boto3

from amazon_transcribe.client import TranscribeStreamingClient

from lib.transcript_handler import TranscriptHandler

# AWS region
REGION = "us-west-2"

# Audio input configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000


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
        "3": "es-ES",
    }

    # Prompt user to select a language
    print("Select a language for transcription:")
    for key, value in supported_languages.items():
        if value == "en-US":
            print(f"{key}: English")
        if value == "zh-CN":
            print(f"{key}: 中文")
        if value == "es-ES":
            print(f"{key}: Español")

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

            transcribe_client = TranscribeStreamingClient(region=REGION)
            bedrock_runtime = boto3.client(
                service_name="bedrock-runtime", region_name=REGION
            )
            polly_client = boto3.client("polly", region_name=REGION)

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
