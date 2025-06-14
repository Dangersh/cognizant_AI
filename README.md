# Cohere Text Generation App

This is a simple terminal-based text generation application using Cohere's `generate()` API. The app allows users to input custom prompts, receive dynamic text completions, and control generation settings such as temperature and output length.

## Features

- Interactive prompt/response interface in the terminal
- Customizable temperature and max_tokens
- Input validation and error handling
- Supports a wide variety of prompt styles (creative, instructional, factual)

## Requirements

- Python 3.7 or higher
- Cohere Python SDK

Install dependencies using pip:

```bash
pip install cohere

## Example Prompts

Here are some example prompts to try:

- Explain recursion like I’m five.
- Write a haiku about the ocean.
- Summarize: Photosynthesis is the process by which...
- Continue this story: Once upon a time, there was a robot who...

## Customization

The app can be customized to adjust the behavior of the AI:

- **Model version**: Change `"command"` to `"command-nightly"` for the latest available model.
- **Creativity**: Adjust the `temperature` parameter. Set closer to `0` for more factual responses, or closer to `1` for more creative output.
- **Length**: Use `max_tokens` to define the maximum length of the generated response._

