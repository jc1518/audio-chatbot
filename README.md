# Voice-enabled Chatbot with AWS Services

This chatbot application uses AWS services to enable voice interaction with Claude 3 Sonnet. It captures voice input, transcribes it using AWS Transcribe Streaming, and gets responses from Claude using AWS Bedrock.

## Prerequisites

1. AWS Account with access to:
   - AWS Transcribe
   - AWS Bedrock (with Claude 3 Sonnet model access)
   
2. AWS credentials configured locally
3. Python 3.7+

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

1. Run the chatbot:
```bash
python chatbot.py
```

2. Speak into your microphone
3. The transcribed text will appear on screen
4. Claude's response will be displayed
5. Press Ctrl+C to stop the program

## Note

Make sure you have proper AWS credentials configured with access to AWS Transcribe and Bedrock services in the us-west-2 region.