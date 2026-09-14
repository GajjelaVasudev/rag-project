require('dotenv').config();
const OpenAI = require('openai');
const Anthropic = require('@anthropic-ai/sdk');

// Groq and OpenRouter both speak the OpenAI API format, so the same
// OpenAI SDK works for both — just point it at a different baseURL.
const groq = new OpenAI({
  apiKey: process.env.GROQ_API_KEY,
  baseURL: 'https://api.groq.com/openai/v1',
});

const openrouter = new OpenAI({
  apiKey: process.env.OPENROUTER_API_KEY,
  baseURL: 'https://openrouter.ai/api/v1',
});

// Anthropic uses its own SDK and message format, not OpenAI-shaped.
const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

async function generateWithGroq(prompt) {
  const res = await groq.chat.completions.create({
    model: 'openai/gpt-oss-120b',
    messages: [{ role: 'user', content: prompt }],
  });
  return res.choices[0].message.content;
}

async function generateWithOpenRouter(prompt) {
  const res = await openrouter.chat.completions.create({
    // Check openrouter.ai/models (filter: Free) if this ID stops working —
    // the free model lineup rotates.
    model: 'inclusionai/ling-3.0-flash-vl:free',
    messages: [{ role: 'user', content: prompt }],
  });
  return res.choices[0].message.content;
}

async function generateWithAnthropic(prompt) {
  const res = await anthropic.messages.create({
    model: 'claude-3-5-haiku-20241022',
    max_tokens: 1024,
    messages: [{ role: 'user', content: prompt }],
  });
  return res.content[0].text;
}

// Order = priority. Groq first (most reliable free tier), OpenRouter
// second, Anthropic last (finite trial credit, not a renewing free tier).
const providers = [
  { name: 'Groq', fn: generateWithGroq },
  { name: 'OpenRouter', fn: generateWithOpenRouter },
  { name: 'Anthropic', fn: generateWithAnthropic },
];

async function generateAnswer(prompt) {
  for (const provider of providers) {
    try {
      const text = await provider.fn(prompt);
      return { text, usedProvider: provider.name };
    } catch (err) {
      console.warn(`[fallback] ${provider.name} failed (${err.message}). Trying next...`);
    }
  }
  throw new Error('All generation providers failed.');
}

module.exports = { generateAnswer };