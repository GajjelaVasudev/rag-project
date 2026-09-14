require("dotenv").config();

const express = require("express");
const { GoogleGenAI } = require("@google/genai");
const { Pinecone } = require("@pinecone-database/pinecone");
const { generateAnswer } = require('./generation');


// ========================================
// Gemini Configuration (embeddings only now)
// ========================================

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY
});


// ========================================
// Pinecone Configuration
// ========================================

const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY
});

const index = pc.index(
  process.env.PINECONE_INDEX_NAME
);


// ========================================
// Express Configuration
// ========================================

const app = express();

app.use(express.json());
app.use(express.static("public"));


// ========================================
// Conversation Memory
// ========================================

// Simple in-memory conversation history.
// Fine for a single-user demo; for multiple users we'd key this
// by session/user ID instead of one shared array.

let conversationHistory = [];


// ========================================
// Create Embedding
// ========================================

async function embedQuery(text) {

  const response = await ai.models.embedContent({

    model: "gemini-embedding-001",

    contents: [text],

    config: {
      // Must match the Pinecone index and the embeddings used in ingest.js.
      outputDimensionality: 1536
    }

  });

  return response.embeddings[0].values;
}


// ========================================
// Chat API
// ========================================

app.post("/api/chat", async (req, res) => {

  try {

    // ------------------------------------
    // Get question from frontend
    // ------------------------------------

    const { question } = req.body;

    if (!question || !question.trim()) {
      return res.status(400).json({
        error: "Question is required"
      });
    }


    // ------------------------------------
    // 1. Convert question into embedding
    // ------------------------------------

    const queryVector = await embedQuery(question);

    console.log("Query vector dimension:", queryVector.length);


    // ------------------------------------
    // 2. Search Pinecone
    // ------------------------------------

    const results = await index.query({
      vector: queryVector,
      topK: 3,
      includeMetadata: true
    });


    // ------------------------------------
    // 3. Extract relevant document text
    // ------------------------------------

    const context = results.matches
      .map((match) => match.metadata.text)
      .join("\n\n---\n\n");


    // ------------------------------------
    // 4. Get recent conversation history
    // ------------------------------------

    const historyText = conversationHistory
      .slice(-3)
      .map((turn) => {
        return `User: ${turn.question}
Assistant: ${turn.answer}`;
      })
      .join("\n\n");


    // ------------------------------------
    // 5. Build RAG prompt
    // ------------------------------------

    const prompt = `You are a helpful assistant answering questions about the provided documents.

Use ONLY the information in the context below to answer the question.

If the answer is not present in the context, say:

"I don't know based on the provided information."

Use the conversation history only to understand follow-up questions.

Never use conversation history as the source of factual information.

Conversation so far:

${historyText || "(none yet)"}


Context from documents:

${context || "(no relevant context found)"}


New question:

${question}


Answer:`;


    // ------------------------------------
    // 6. Generate answer (Groq -> OpenRouter -> Anthropic fallback chain)
    // ------------------------------------

    const { text, usedProvider } = await generateAnswer(prompt);

    const answer = text; // <-- was `result.text`, `result` no longer exists


    // ------------------------------------
    // 7. Save conversation
    // ------------------------------------

    conversationHistory.push({
      question: question,
      answer: answer
    });


    // ------------------------------------
    // 8. Send response to frontend
    // ------------------------------------

    res.json({
      answer: answer,
      usedProvider: usedProvider, // which fallback model actually answered
      sources: results.matches.map((match) => ({
        source: match.metadata.source,
        score: match.score
      }))
    });

  } catch (error) {

    console.error("❌ Chat error:", error);

    res.status(500).json({
      error: "Something went wrong while processing your question."
    });

  }

});


// ========================================
// Server Configuration
// ========================================

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
});