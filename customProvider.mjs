import logger from './node_modules/promptfoo/dist/src/logger.js';
import { fetchWithCache } from './node_modules/promptfoo/dist/src/cache.js';
import { parseChatPrompt, REQUEST_TIMEOUT_MS } from './node_modules/promptfoo/dist/src/providers/shared.js';

function getTokenUsage(data, cached) {
    if (data.usage) {
        if (cached) {
            return { cached: data.usage.total_tokens, total: data.usage.total_tokens };
        }
        else {
            return {
                total: data.usage.total_tokens,
                prompt: data.usage.prompt_tokens || 0,
                completion: data.usage.completion_tokens || 0,
            };
        }
    }
    return {};
}
class CustomGenericProvider {
    constructor(options = {}) {
        const { config, id, env } = options;
        this.env = env;
        this.modelName = options.modelName || 'gpt-3.5-turbo';
        this.config = config || {};
        this.id = id ? () => id : this.id;
    }
    id() {
        return `custom:${this.modelName}`;
    }
    toString() {
        return `[Custom Provider ${this.modelName}]`;
    }
    getOrganization() {
        return (this.config.organization || this.env?.OPENAI_ORGANIZATION || process.env.OPENAI_ORGANIZATION);
    }
    getApiUrlDefault() {
        return 'https://api.openai.com/v1';
    }
    getApiUrl() {
        const apiHost = this.config.apiHost || this.env?.OPENAI_API_HOST || process.env.OPENAI_API_HOST;
        if (apiHost) {
            return `https://${apiHost}/v1`;
        }
        return (this.config.apiBaseUrl ||
            this.env?.OPENAI_API_BASE_URL ||
            process.env.OPENAI_API_BASE_URL ||
            this.getApiUrlDefault());
    }
    getApiKey() {
        return (this.config.apiKey ||
            (this.config?.apiKeyEnvar
                ? process.env[this.config.apiKeyEnvar] ||
                    this.env?.[this.config.apiKeyEnvar]
                : undefined) ||
            this.env?.OPENAI_API_KEY ||
            process.env.OPENAI_API_KEY);
    }
    // @ts-ignore: Params are not used in this implementation
    async callApi(prompt, context, callApiOptions) {
        throw new Error('Not implemented');
    }
}

class CustomEmbeddingProvider extends CustomGenericProvider {
    async callEmbeddingApi(text) {
        if (!this.getApiKey()) {
            throw new Error('OpenAI API key must be set for similarity comparison');
        }
        const body = {
            input: text,
            model: this.modelName,
        };
        let data, cached = false;
        try {
            ({ data, cached } = (await (0, fetchWithCache)(`${this.getApiUrl()}/embeddings`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${this.getApiKey()}`,
                    ...(this.getOrganization() ? { 'OpenAI-Organization': this.getOrganization() } : {}),
                    ...(this.config?.apiHeaders || {}),
                },
                body: JSON.stringify(body),
            }, REQUEST_TIMEOUT_MS)));
        }
        catch (err) {
            logger.default.error(`API call error: ${err}`);
            throw err;
        }
        logger.default.debug(`\tOpenAI embeddings API response: ${JSON.stringify(data)}`);
        try {
            const embedding = data?.data?.[0]?.embedding;
            if (!embedding) {
                throw new Error('No embedding found in OpenAI embeddings API response');
            }
            return {
                embedding,
                tokenUsage: getTokenUsage(data, cached),
            };
        }
        catch (err) {
            logger.default.error(data.error.message);
            throw err;
        }
    }
}
class CustomCompletionProvider extends CustomGenericProvider {
    constructor(options = {}) {
        super(options);
        if (!CustomCompletionProvider.OPENAI_COMPLETION_MODEL_NAMES.includes(this.modelName) &&
            this.getApiUrl() === this.getApiUrlDefault()) {
            logger.default.warn(`FYI: Using unknown OpenAI completion model: ${this.modelName}`);
        }
    }
    async callApi(prompt, context, callApiOptions) {
        if (!this.getApiKey()) {
            throw new Error('OpenAI API key is not set. Set the OPENAI_API_KEY environment variable or add `apiKey` to the provider config.');
        }
        let stop;
        try {
            stop = process.env.OPENAI_STOP
                ? JSON.parse(process.env.OPENAI_STOP)
                : this.config?.stop || ['<|im_end|>', '<|endoftext|>'];
        }
        catch (err) {
            throw new Error(`OPENAI_STOP is not a valid JSON string: ${err}`);
        }
        const body = {
            model: this.modelName,
            prompt,
            seed: this.config.seed || 0,
            max_tokens: this.config.max_tokens ?? parseInt(process.env.OPENAI_MAX_TOKENS || '1024'),
            temperature: this.config.temperature ?? parseFloat(process.env.OPENAI_TEMPERATURE || '0'),
            top_p: this.config.top_p ?? parseFloat(process.env.OPENAI_TOP_P || '1'),
            presence_penalty: this.config.presence_penalty ?? parseFloat(process.env.OPENAI_PRESENCE_PENALTY || '0'),
            frequency_penalty: this.config.frequency_penalty ?? parseFloat(process.env.OPENAI_FREQUENCY_PENALTY || '0'),
            best_of: this.config.best_of ?? parseInt(process.env.OPENAI_BEST_OF || '1'),
            ...(callApiOptions?.includeLogProbs ? { logprobs: callApiOptions.includeLogProbs } : {}),
            ...(stop ? { stop } : {}),
            ...(this.config.passthrough || {}),
        };
        logger.default.debug(`Calling OpenAI API: ${JSON.stringify(body)}`);
        let data, cached = false;
        try {
            ({ data, cached } = (await (0, fetchWithCache)(`${this.getApiUrl()}/completions`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${this.getApiKey()}`,
                    ...(this.getOrganization() ? { 'OpenAI-Organization': this.getOrganization() } : {}),
                    ...(this.config?.apiHeaders || {}),
                },
                body: JSON.stringify(body),
            }, REQUEST_TIMEOUT_MS)));
        }
        catch (err) {
            return {
                error: `API call error: ${String(err)}`,
            };
        }
        logger.default.debug(`\tOpenAI completions API response: ${JSON.stringify(data)}`);
        if (data.error) {
            return {
                error: formatCustomError(data),
            };
        }
        try {
            return {
                output: data.choices[0].text,
                tokenUsage: getTokenUsage(data, cached),
                cached,
                cost: calculateCost(this.modelName, this.config, data.usage?.prompt_tokens, data.usage?.completion_tokens),
            };
        }
        catch (err) {
            return {
                error: `API error: ${String(err)}: ${JSON.stringify(data)}`,
            };
        }
    }
}
CustomCompletionProvider.OPENAI_COMPLETION_MODELS = [
    {
        id: 'gpt-3.5-turbo-instruct',
        cost: {
            input: 0.0015 / 1000,
            output: 0.002 / 1000,
        },
    },
    {
        id: 'gpt-3.5-turbo-instruct-0914',
        cost: {
            input: 0.0015 / 1000,
            output: 0.002 / 1000,
        },
    },
    {
        id: 'text-davinci-003',
    },
    {
        id: 'text-davinci-002',
    },
    {
        id: 'text-curie-001',
    },
    {
        id: 'text-babbage-001',
    },
    {
        id: 'text-ada-001',
    },
];
CustomCompletionProvider.OPENAI_COMPLETION_MODEL_NAMES = CustomCompletionProvider.OPENAI_COMPLETION_MODELS.map((model) => model.id);
export default class CustomChatCompletionProvider extends CustomGenericProvider {
    constructor(options = {}) {
        super(options);
        if (!CustomChatCompletionProvider.OPENAI_CHAT_MODEL_NAMES.includes(this.modelName)) {
            logger.default.warn(`Using unknown OpenAI chat model: ${this.modelName}`);
        }
    }
    async callApi(prompt, context, callApiOptions) {
        if (!this.getApiKey()) {
            throw new Error('OpenAI API key is not set. Set the OPENAI_API_KEY environment variable or add `apiKey` to the provider config.');
        }
        const messages = (0, parseChatPrompt)(prompt, [{ role: 'user', content: prompt }]);
        let stop;
        try {
            stop = process.env.OPENAI_STOP
                ? JSON.parse(process.env.OPENAI_STOP)
                : this.config?.stop || [];
        }
        catch (err) {
            throw new Error(`OPENAI_STOP is not a valid JSON string: ${err}`);
        }
        const body = {
            model: this.modelName,
            messages: messages,
            seed: this.config.seed || 0,
            max_tokens: this.config.max_tokens ?? parseInt(process.env.OPENAI_MAX_TOKENS || '1024'),
            temperature: this.config.temperature ?? parseFloat(process.env.OPENAI_TEMPERATURE || '0'),
            top_p: this.config.top_p ?? parseFloat(process.env.OPENAI_TOP_P || '1'),
            presence_penalty: this.config.presence_penalty ?? parseFloat(process.env.OPENAI_PRESENCE_PENALTY || '0'),
            frequency_penalty: this.config.frequency_penalty ?? parseFloat(process.env.OPENAI_FREQUENCY_PENALTY || '0'),
            ...(this.config.functions ? { functions: this.config.functions } : {}),
            ...(this.config.function_call ? { function_call: this.config.function_call } : {}),
            ...(this.config.tools ? { tools: this.config.tools } : {}),
            ...(this.config.tool_choice ? { tool_choice: this.config.tool_choice } : {}),
            ...(this.config.response_format ? { response_format: this.config.response_format } : {}),
            ...(callApiOptions?.includeLogProbs ? { logprobs: callApiOptions.includeLogProbs } : {}),
            ...(this.config.stop ? { stop: this.config.stop } : {}),
            ...(this.config.passthrough || {}),
        };
        logger.default.debug(`Calling OpenAI API: ${JSON.stringify(body)}`);
        let data, cached = false;
        try {
            ({ data, cached } = (await (0, fetchWithCache)(`${this.getApiUrl()}/chat/completions`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${this.getApiKey()}`,
                    ...(this.getOrganization() ? { 'OpenAI-Organization': this.getOrganization() } : {}),
                    ...(this.config?.apiHeaders || {}),
                },
                body: JSON.stringify(body),
            }, REQUEST_TIMEOUT_MS)));
        }
        catch (err) {
            return {
                error: `API call error: ${String(err)}`,
            };
        }
        logger.default.debug(`\tOpenAI chat completions API response: ${JSON.stringify(data)}`);
        if (data.error) {
            return {
                error: formatCustomError(data),
            };
        }
        try {
            const message = data.choices[0].message;
            const output = message.content === null ? message.function_call || message.tool_calls : message.content;
            const logProbs = data.choices[0].logprobs?.content?.map((logProbObj) => logProbObj.logprob);
            return {
                output,
                tokenUsage: getTokenUsage(data, cached),
                cached,
                logProbs,
                cost: calculateCost(this.modelName, this.config, data.usage?.prompt_tokens, data.usage?.completion_tokens),
            };
        }
        catch (err) {
            return {
                error: `API error: ${String(err)}: ${JSON.stringify(data)}`,
            };
        }
    }
}
CustomChatCompletionProvider.OPENAI_CHAT_MODELS = [
    ...['gpt-4', 'gpt-4-0314', 'gpt-4-0613'].map((model) => ({
        id: model,
        cost: {
            input: 0.03 / 1000,
            output: 0.06 / 1000,
        },
    })),
    ...[
        'gpt-4-1106-preview',
        'gpt-4-1106-vision-preview',
        'gpt-4-0125-preview',
        'gpt-4-turbo-preview',
    ].map((model) => ({
        id: model,
        cost: {
            input: 0.01 / 1000,
            output: 0.03 / 1000,
        },
    })),
    ...['gpt-4-32k', 'gpt-4-32k-0314'].map((model) => ({
        id: model,
        cost: {
            input: 0.06 / 1000,
            output: 0.12 / 1000,
        },
    })),
    ...[
        'gpt-3.5-turbo',
        'gpt-3.5-turbo-0301',
        'gpt-3.5-turbo-0613',
        'gpt-3.5-turbo-1106',
        'gpt-3.5-turbo-0125',
        'gpt-3.5-turbo-16k',
        'gpt-3.5-turbo-16k-0613',
    ].map((model) => ({
        id: model,
        cost: {
            input: 0.0005 / 1000,
            output: 0.0015 / 1000,
        },
    })),
];
CustomChatCompletionProvider.OPENAI_CHAT_MODEL_NAMES = CustomChatCompletionProvider.OPENAI_CHAT_MODELS.map((model) => model.id);
function formatCustomError(data) {
    return (`API error: ${data.error.message}` +
        (data.error.type ? `, Type: ${data.error.type}` : '') +
        (data.error.code ? `, Code: ${data.error.code}` : ''));
}
function calculateCost(modelName, config, promptTokens, completionTokens) {
    if (!promptTokens || !completionTokens) {
        return undefined;
    }
    const model = [
        ...CustomChatCompletionProvider.OPENAI_CHAT_MODELS,
        ...CustomCompletionProvider.OPENAI_COMPLETION_MODELS,
    ].find((m) => m.id === modelName);
    if (!model || !model.cost) {
        return undefined;
    }
    const inputCost = config.cost ?? model.cost.input;
    const outputCost = config.cost ?? model.cost.output;
    return inputCost * promptTokens + outputCost * completionTokens || undefined;
}