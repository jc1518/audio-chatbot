import threading
import sys
import random
import json
import time

from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

from lib.web_search import web_search

# Audio output
SIZE = -16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

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
    "stopSequences": [],
}

SYSTEM = [
    {
        "text": f"""
          ## Core Directive
          You are a highly intelligent virtual assistant committed to providing accurate, informative, and concise responses to user inquiries.

          ## Current Context Awareness
          - Today's date is {time.strftime("%Y-%m-%d")}
          - My location is Sydney, Australia
          - When ask weather,  use the Bureau of Meteorology (BOM) website
    
          ## Knowledge and Search Protocol
          - Use web_search ONLY when you are CERTAIN that:
            a) The information is not within your current knowledge
            b) You cannot confidently construct a response using existing information
            c) The query requires verified, up-to-date information
          - Response Strategy:
            a) First, attempt to answer using internal knowledge
            b) If knowledge is insufficient, clearly communicate the limitation
            c) Synthesize search results with existing understanding
            d) Maintain response conciseness

          ## Operational Principles
          - Prioritize accuracy over volume of information
          - Be transparent about information sources
          - Ensure user satisfaction through precise, helpful responses
          - Avoid unnecessary web searches
          - Maintain a helpful and engaging communication style
        """
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
                                "default": 3,
                            },
                        },
                        "required": ["query"],
                    }
                },
            }
        }
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
        self.listening = True
        self.polly_finished = threading.Event()
        self.conversation_history = converstation_history

        # Pre-initialize pygame mixer
        original_stdout = sys.stdout
        sys.stdout = NullDevice()
        from pygame import mixer

        sys.stdout = original_stdout
        self.mixer = mixer
        self.mixer.init(frequency=RATE, size=SIZE, channels=CHANNELS)
        self.mixer_lock = threading.Lock()

    def handle_tool_use(self, tool_use):
        try:
            if tool_use["name"] == "web_search":
                input_data = tool_use["input"]
                query = input_data["query"]
                max_results = input_data.get("max_results", 5)
                search_results = web_search(query, max_results)

                text_results = []
                for result in search_results:
                    text_results.append(
                        {
                            "text": f"Title: {result['title']}\nLink: {result['link']}\nContent: {result['body']}"
                        }
                    )

                return {"toolUseId": tool_use["toolUseId"], "content": text_results}
        except Exception as e:
            print(f"Error executing web_search: {e}")
            return {
                "toolUseId": tool_use["toolUseId"],
                "content": [{"text": "Error performing web search"}],
            }

    def process_response_stream(self, response_stream, modelId, initial_response=""):
        full_response = initial_response
        current_tool_use = None
        tool_use_input_parts = []

        for event in response_stream:
            try:
                if "contentBlockStart" in event:
                    start = event["contentBlockStart"].get("start", {})
                    if "toolUse" in start:
                        current_tool_use = start["toolUse"]
                        tool_use_input_parts = []

                elif "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"]["delta"]
                    if "text" in delta:
                        text = delta["text"]
                        full_response += text
                        print(text, end="", flush=True)
                    elif "toolUse" in delta and current_tool_use:
                        tool_use_input_parts.append(delta["toolUse"]["input"])

                elif "contentBlockStop" in event:
                    if current_tool_use and tool_use_input_parts:
                        # Handle tool use and get new response
                        tool_input_str = "".join(tool_use_input_parts)
                        tool_input = json.loads(tool_input_str)
                        current_tool_use["input"] = tool_input

                        # Add the current response and tool use to conversation history
                        self.conversation_history.append(
                            {
                                "role": "assistant",
                                "content": [
                                    {"text": full_response},
                                    {
                                        "toolUse": {
                                            "toolUseId": current_tool_use["toolUseId"],
                                            "name": current_tool_use["name"],
                                            "input": tool_input,
                                        }
                                    },
                                ],
                            }
                        )

                        # Execute tool and add result to conversation history
                        tool_result = self.handle_tool_use(current_tool_use)
                        self.conversation_history.append(
                            {
                                "role": "user",
                                "content": [{"toolResult": tool_result}],
                            }
                        )

                        # Get new response after tool use
                        new_response = self.bedrock_runtime.converse_stream(
                            modelId=modelId,
                            inferenceConfig=INFERENCE_CONFIG,
                            toolConfig=TOOL_CONFIG,
                            system=SYSTEM,
                            messages=self.conversation_history,
                        )

                        # Process the new response stream recursively
                        if "stream" in new_response:
                            return self.process_response_stream(
                                new_response["stream"], modelId, ""
                            )

                elif "messageStop" in event:
                    if event["messageStop"]["stopReason"] == "end_turn":
                        break

            except Exception as chunk_error:
                print(f"\nError processing chunk: {chunk_error}")
                continue

        return full_response

    def speak_response(self, text):
        try:
            if self.language_code == "zh-CN":
                voice_id = "Zhiyu"
                engine = "neural"
            else:
                voice_id = "Joanna"
                engine = "generative"

            response = self.polly_client.synthesize_speech(
                Text=text, OutputFormat="pcm", VoiceId=voice_id, Engine=engine
            )

            if "AudioStream" in response:
                import pyaudio

                # Initialize PyAudio
                p = pyaudio.PyAudio()

                # Open stream
                stream = p.open(
                    format=p.get_format_from_width(2),  # 16-bit PCM
                    channels=CHANNELS,
                    rate=RATE,  # Sample rate
                    output=True,
                )

                print("\nPress Enter to stop the voice playback...")
                should_stop = threading.Event()

                def wait_for_input():
                    try:
                        input()
                        should_stop.set()
                    except Exception as e:
                        print(f"\nError stopping playback: {e}")

                input_thread = threading.Thread(target=wait_for_input)
                input_thread.daemon = True
                input_thread.start()

                # Stream the audio data
                audio_stream = response["AudioStream"]
                chunk_size = CHUNK
                while not should_stop.is_set():
                    data = audio_stream.read(chunk_size)
                    if not data:
                        break
                    stream.write(data)

                if should_stop.is_set():
                    print("\nVoice playback stopped.")

                # Clean up
                stream.stop_stream()
                stream.close()
                p.terminate()

        except Exception as e:
            print(f"Error in text-to-speech: {e}")
        finally:
            self.polly_finished.set()

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results

        for result in results:
            if result.alternatives and self.listening:
                transcript = result.alternatives[0].transcript
                if result.is_partial:
                    print(f"\rUser: {transcript}", end="", flush=True)
                else:
                    print(f"\rUser: {transcript}")

                    try:
                        self.listening = False
                        self.polly_finished.clear()

                        self.conversation_history.append(
                            {"role": "user", "content": [{"text": transcript}]}
                        )

                        modelId = random.choice(MODELS)
                        print(f"\n\r(calling {modelId})")
                        print("\nAssistant: ", end="", flush=True)

                        response = self.bedrock_runtime.converse_stream(
                            modelId=modelId,
                            inferenceConfig=INFERENCE_CONFIG,
                            system=SYSTEM,
                            messages=self.conversation_history,
                            toolConfig=TOOL_CONFIG,
                        )

                        if "stream" in response:
                            full_response = self.process_response_stream(
                                response["stream"], modelId
                            )

                            if full_response:
                                self.conversation_history.append(
                                    {
                                        "role": "assistant",
                                        "content": [{"text": full_response}],
                                    }
                                )

                                print("\n")
                                self.speak_response(full_response)

                        print("-" * 50 + "\n")
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
