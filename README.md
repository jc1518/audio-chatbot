# Audio Chatbot

This chatbot application uses AWS services to enable voice interaction with LLM in Bedrock. It captures voice input, transcribes it using AWS Transcribe Streaming, and gets responses from model using AWS Bedrock, then uses Polly to convert text to speech.

## Prerequisites

1. AWS Account with access to:
   - AWS Transcribe
   - AWS Polly
   - AWS Bedrock model access
2. AWS credentials configured locally
3. Python 3.11+

## Installation

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Make sure you have PortAudio installed for PyAudio to work:

- On Ubuntu/Debian: `sudo apt-get install portaudio19-dev`
- On macOS: `brew install portaudio`
- On Windows: No additional installation needed

## Usage

```bash
python chatbot.py
```

or create a bash alias to run [launch_chatbot.sh](./launch_chatbot.sh)

## Demo

[![Watch the video](https://img.youtube.com/vi/JQwRPY6b3Ec//maxresdefault.jpg)](https://youtu.be/JQwRPY6b3Ec)
