/**
 * Jan AI Client
 * Communicates with Jan AI API server
 */

import axios, { AxiosInstance } from 'axios';

export interface JanAIConfig {
    apiBase: string;
    apiKey?: string;
    model: string;
    enableVision?: boolean;
}

export class JanAIClient {
    private client: AxiosInstance;
    private config: JanAIConfig;

    constructor(config: JanAIConfig) {
        this.config = config;
        
        this.client = axios.create({
            baseURL: config.apiBase,
            headers: {
                'Content-Type': 'application/json',
                ...(config.apiKey && { 'Authorization': `Bearer ${config.apiKey}` })
            }
        });
    }

    async chat(message: string, context?: string): Promise<string> {
        try {
            const response = await this.client.post('/chat/completions', {
                model: this.config.model,
                messages: [
                    ...(context ? [{ role: 'system', content: context }] : []),
                    { role: 'user', content: message }
                ],
                temperature: 0.7,
                max_tokens: 2048
            });

            return response.data.choices[0]?.message?.content || 'No response';
        } catch (error: any) {
            throw new Error(`Jan AI API error: ${error.message}`);
        }
    }

    async chatWithVision(message: string, imageData?: string): Promise<string> {
        if (!this.config.enableVision) {
            return this.chat(message);
        }

        try {
            const messages: any[] = [
                {
                    role: 'user',
                    content: [
                        { type: 'text', text: message },
                        ...(imageData ? [{ type: 'image_url', image_url: { url: imageData } }] : [])
                    ]
                }
            ];

            const response = await this.client.post('/chat/completions', {
                model: this.config.model,
                messages: messages,
                temperature: 0.7,
                max_tokens: 2048
            });

            return response.data.choices[0]?.message?.content || 'No response';
        } catch (error: any) {
            throw new Error(`Jan AI Vision API error: ${error.message}`);
        }
    }

    async healthCheck(): Promise<boolean> {
        try {
            await this.client.get('/health');
            return true;
        } catch {
            return false;
        }
    }
}

