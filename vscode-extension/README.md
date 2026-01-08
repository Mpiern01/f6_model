# F6 Jan AI VSCode Extension

VSCode extension for integrating Jan AI with F6 StreamTrain models.

## Features

- **Code Explanation**: Explain selected code in detail
- **Code Improvement**: Get suggestions for code improvements
- **Test Generation**: Generate comprehensive tests for code
- **Bug Fixing**: Identify and fix bugs in code
- **Vision Support**: Use vision capabilities for code analysis

## Installation

1. Install the extension from the VSCode marketplace (or build from source)
2. Configure Jan AI API settings in VSCode settings

## Configuration

Open VSCode settings and configure:

```json
{
  "f6JanAI.apiBase": "http://localhost:1337/v1",
  "f6JanAI.apiKey": "your-api-key",
  "f6JanAI.model": "Jan-v2-VL-high",  // Options: "Jan-v2-VL-high", "GLM-4.6V-Flash"
  "f6JanAI.enableVision": true
}
```

## Usage

### Commands

- `F6 Jan AI: Ask Jan AI` - Open chat interface
- `F6 Jan AI: Explain Code` - Explain selected code
- `F6 Jan AI: Improve Code` - Get code improvements
- `F6 Jan AI: Generate Tests` - Generate tests for code
- `F6 Jan AI: Fix Bugs` - Fix bugs in code

### Keyboard Shortcuts

- `Cmd+Shift+M` (Mac) / `Ctrl+Shift+M` (Windows/Linux) - Ask Jan AI
- `Cmd+Shift+E` (Mac) / `Ctrl+Shift+E` (Windows/Linux) - Explain Code

## Requirements

- Jan AI server running locally (default: http://localhost:1337)
- F6 StreamTrain model loaded in Jan AI

## Development

```bash
cd vscode-extension
npm install
npm run compile
```

## License

Apache 2.0

