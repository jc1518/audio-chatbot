#!/bin/bash

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment if you're using one
source $DIR/venv/bin/activate

# Run the chatbot
python3 "$DIR/chatbot.py"

# Keep the terminal window open
# read -p "Press Enter to close..." 