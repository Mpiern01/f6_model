/**
 * Code Assistant
 * High-level code assistance functions using Jan AI
 */

import { JanAIClient } from './janai-client';

export class CodeAssistant {
    constructor(private client: JanAIClient) {}

    async explainCode(code: string, language: string): Promise<string> {
        const prompt = `Explain this ${language} code in detail:

\`\`\`${language}
${code}
\`\`\`

Provide:
1. What the code does
2. How it works
3. Key concepts and patterns
4. Potential improvements`;

        return await this.client.chat(prompt);
    }

    async improveCode(code: string, language: string): Promise<string> {
        const prompt = `Improve this ${language} code:

\`\`\`${language}
${code}
\`\`\`

Provide:
1. Improved version of the code
2. Explanation of improvements
3. Best practices applied`;

        return await this.client.chat(prompt);
    }

    async generateTests(code: string, language: string): Promise<string> {
        const prompt = `Generate comprehensive tests for this ${language} code:

\`\`\`${language}
${code}
\`\`\`

Include:
1. Unit tests
2. Edge cases
3. Integration tests if applicable
4. Test fixtures and setup`;

        return await this.client.chat(prompt);
    }

    async fixBugs(code: string, language: string): Promise<string> {
        const prompt = `Fix any bugs in this ${language} code:

\`\`\`${language}
${code}
\`\`\`

Provide:
1. Fixed code
2. Explanation of bugs found
3. How they were fixed`;

        return await this.client.chat(prompt);
    }

    async refactorCode(code: string, language: string, goal?: string): Promise<string> {
        const prompt = `Refactor this ${language} code${goal ? ` to ${goal}` : ''}:

\`\`\`${language}
${code}
\`\`\`

Provide:
1. Refactored code
2. Explanation of refactoring changes
3. Benefits of the refactoring`;

        return await this.client.chat(prompt);
    }

    async generateCode(description: string, language: string): Promise<string> {
        const prompt = `Generate ${language} code for:

${description}

Provide:
1. Complete, working code
2. Comments explaining key parts
3. Usage examples`;

        return await this.client.chat(prompt);
    }
}

