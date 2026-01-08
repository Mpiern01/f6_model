/**
 * F6 Jan AI VSCode Extension
 * Integrates Jan AI with VSCode for code assistance
 */

import * as vscode from 'vscode';
import { JanAIClient } from './janai-client';
import { CodeAssistant } from './code-assistant';

let janAIClient: JanAIClient;
let codeAssistant: CodeAssistant;

export function activate(context: vscode.ExtensionContext) {
    console.log('F6 Jan AI extension is now active!');

    // Initialize Jan AI client
    const config = vscode.workspace.getConfiguration('f6JanAI');
    janAIClient = new JanAIClient({
        apiBase: config.get<string>('apiBase', 'http://localhost:1337/v1'),
        apiKey: config.get<string>('apiKey', ''),
        // OPTION 1: Jan-v2-VL-high (default)
        // OPTION 2: GLM-4.6V-Flash (change in settings)
        model: config.get<string>('model', 'Jan-v2-VL-high'),
        enableVision: config.get<boolean>('enableVision', true)
    });

    // Initialize code assistant
    codeAssistant = new CodeAssistant(janAIClient);

    // Register commands
    const askCommand = vscode.commands.registerCommand('f6JanAI.ask', async () => {
        await handleAskCommand();
    });

    const explainCodeCommand = vscode.commands.registerCommand('f6JanAI.explainCode', async () => {
        await handleExplainCodeCommand();
    });

    const improveCodeCommand = vscode.commands.registerCommand('f6JanAI.improveCode', async () => {
        await handleImproveCodeCommand();
    });

    const generateTestsCommand = vscode.commands.registerCommand('f6JanAI.generateTests', async () => {
        await handleGenerateTestsCommand();
    });

    const fixBugsCommand = vscode.commands.registerCommand('f6JanAI.fixBugs', async () => {
        await handleFixBugsCommand();
    });

    context.subscriptions.push(
        askCommand,
        explainCodeCommand,
        improveCodeCommand,
        generateTestsCommand,
        fixBugsCommand
    );
}

export function deactivate() {
    // Cleanup
}

async function handleAskCommand() {
    const question = await vscode.window.showInputBox({
        prompt: 'Ask Jan AI',
        placeHolder: 'Enter your question...'
    });

    if (!question) {
        return;
    }

    await showResponse(question, async () => {
        return await janAIClient.chat(question);
    });
}

async function handleExplainCodeCommand() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor');
        return;
    }

    const selectedText = editor.document.getText(editor.selection);
    if (!selectedText) {
        vscode.window.showWarningMessage('No code selected');
        return;
    }

    await showResponse('Explain this code', async () => {
        return await codeAssistant.explainCode(selectedText, editor.document.languageId);
    });
}

async function handleImproveCodeCommand() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor');
        return;
    }

    const selectedText = editor.document.getText(editor.selection);
    if (!selectedText) {
        vscode.window.showWarningMessage('No code selected');
        return;
    }

    await showResponse('Improve this code', async () => {
        return await codeAssistant.improveCode(selectedText, editor.document.languageId);
    });
}

async function handleGenerateTestsCommand() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor');
        return;
    }

    const selectedText = editor.document.getText(editor.selection);
    if (!selectedText) {
        vscode.window.showWarningMessage('No code selected');
        return;
    }

    await showResponse('Generate tests', async () => {
        return await codeAssistant.generateTests(selectedText, editor.document.languageId);
    });
}

async function handleFixBugsCommand() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor');
        return;
    }

    const selectedText = editor.document.getText(editor.selection);
    if (!selectedText) {
        vscode.window.showWarningMessage('No code selected');
        return;
    }

    await showResponse('Fix bugs', async () => {
        return await codeAssistant.fixBugs(selectedText, editor.document.languageId);
    });
}

async function showResponse(title: string, responseFn: () => Promise<string>) {
    const panel = vscode.window.createWebviewPanel(
        'f6JanAI',
        title,
        vscode.ViewColumn.Beside,
        {
            enableScripts: true,
            retainContextWhenHidden: true
        }
    );

    panel.webview.html = getLoadingHtml();

    try {
        const response = await responseFn();
        panel.webview.html = getResponseHtml(response);
    } catch (error: any) {
        panel.webview.html = getErrorHtml(error.message || 'Unknown error');
    }
}

function getLoadingHtml(): string {
    return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {
                    font-family: var(--vscode-font-family);
                    padding: 20px;
                }
                .loading {
                    text-align: center;
                    padding: 40px;
                }
            </style>
        </head>
        <body>
            <div class="loading">
                <p>Loading response from Jan AI...</p>
            </div>
        </body>
        </html>
    `;
}

function getResponseHtml(response: string): string {
    const escaped = response.replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {
                    font-family: var(--vscode-font-family);
                    padding: 20px;
                    line-height: 1.6;
                }
                pre {
                    background: var(--vscode-text-block-background);
                    padding: 10px;
                    border-radius: 4px;
                    overflow-x: auto;
                }
                code {
                    font-family: var(--vscode-editor-font-family);
                }
            </style>
        </head>
        <body>
            <pre><code>${escaped}</code></pre>
        </body>
        </html>
    `;
}

function getErrorHtml(error: string): string {
    return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {
                    font-family: var(--vscode-font-family);
                    padding: 20px;
                    color: var(--vscode-errorForeground);
                }
            </style>
        </head>
        <body>
            <p>Error: ${error}</p>
            <p>Make sure Jan AI server is running at the configured API base URL.</p>
        </body>
        </html>
    `;
}

