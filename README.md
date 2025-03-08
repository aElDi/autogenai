# AI Application Generator

An intelligent application generator that leverages two neural networks (teacher and generator) to create custom software applications through an interactive dialog process.

## Overview

This tool automates the software development process by using advanced AI models to:

1. **Refine Requirements**: A teacher model helps analyze and enhance your project requirements through interactive dialogue.
2. **Generate Code**: A generator model creates working code based on the refined requirements.
3. **Improve Implementation**: Through iterative feedback, the code is continuously improved.

The system uses the OpenRouter API to access powerful AI models like Claude and GPT-4.

## Features

- Interactive dialogue-based requirements refinement
- Automatic code generation from detailed requirements
- Iterative code improvement cycle
- Automatic file extraction and organization
- Comprehensive error handling and retry mechanisms
- API connection testing
- Detailed logging

## Installation

### Prerequisites

- Python 3.8 or higher
- An OpenRouter API key

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-application-generator.git
   cd ai-application-generator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API key:
   
   Either set it as an environment variable:
   ```bash
   export OPENROUTER_API_KEY="your_api_key_here"
   ```
   
   Or create a `.env` file in the project directory:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

## Usage

### Basic Usage

Run the application in interactive mode:

```bash
python main.py
```

### Command Line Options

```bash
python main.py [options]
```

Available options:
- `--api-key KEY`: Your OpenRouter API key (alternatively use environment variable)
- `--teacher-model MODEL`: Model for the teacher role (default: anthropic/claude-3-opus)
- `--generator-model MODEL`: Model for the generator role (default: openai/gpt-4o)
- `--site-url URL`: Your site URL for OpenRouter rankings
- `--site-name NAME`: Your site name for OpenRouter rankings
- `--verbose`: Enable detailed logging
- `--test-api`: Test API connection and exit

### Example Session

1. Start the application:
   ```bash
   python main.py --verbose
   ```

2. Enter your initial project description when prompted.

3. Engage in a dialogue with the teacher to refine requirements.

4. Type 'DONE' when you're satisfied with the requirements.

5. Provide final confirmation to generate the detailed requirements document.

6. The initial code will be generated automatically.

7. Provide feedback for code improvements, or type 'DONE' to finalize.

8. The project files will be saved to the 'output' directory.

## Output Structure

After running the application, you'll find the following in the `output` directory:

- `requirements.md`: Detailed requirements document
- Source code files extracted from the generated code
- `teacher_conversation.json`: Record of the teacher dialogue
- `generator_conversation.json`: Record of the generator dialogue

## Advanced Configuration

### Using Different Models

You can specify different models for the teacher and generator roles:

```bash
python main.py --teacher-model anthropic/claude-3-sonnet --generator-model openai/gpt-4-turbo
```

Available models depend on your OpenRouter subscription and the models they support.

### Troubleshooting

If you encounter connection issues, test your API connection:

```bash
python main.py --test-api
```

## Development

### Project Structure

- `main.py`: Main application file
- `requirements.txt`: Python dependencies
- `output/`: Generated code and documentation
- `.env`: Environment variables (create this yourself, not in repo)

### Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenRouter](https://openrouter.ai/) for providing access to various AI models
- All the AI models that power this application

## Disclaimer

The code generated by this tool should be reviewed before use in production environments. While the AI models strive to create high-quality code, they may not always produce optimal or secure implementations.