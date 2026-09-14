require("dotenv").config();

const readline = require("readline");
const { GoogleGenAI } = require("@google/genai");
const { Pinecone } = require("@pinecone-database/pinecone");


// ===============================
// Gemini Configuration
// ===============================

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY
});


// ===============================
// Pinecone Configuration
// ===============================

const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY
});

const index = pc.index(
  process.env.PINECONE_INDEX_NAME
);


// ===============================
// Conversation Memory
// ===============================

// Stores previous question + answer pairs
let conversationHistory = [];


// ===============================
// Create Query Embedding
// ===============================

async function embedQuery(text) {

  const response = await ai.models.embedContent({

    model: "gemini-embedding-001",

    contents: [text],

    config: {
      // MUST MATCH ingest.js
      outputDimensionality: 1536
    }

  });

  return response.embeddings[0].values;
}


// ===============================
// Ask Question
// ===============================

async function askQuestion(question) {

  // --------------------------------
  // 1. Convert question to embedding
  // --------------------------------

  const queryVector = await embedQuery(
    question
  );


  // --------------------------------
  // 2. Search Pinecone
  // --------------------------------

  const results = await index.query({

    vector: queryVector,

    topK: 3,

    includeMetadata: true

  });


  // --------------------------------
  // 3. Extract relevant text
  // --------------------------------

  const context = results.matches

    .map((match) => match.metadata.text)

    .join("\n\n---\n\n");


  // --------------------------------
  // 4. Prepare conversation history
  // --------------------------------

  const historyText = conversationHistory

    .slice(-3)

    .map((turn) => {

      return `User: ${turn.question}
Assistant: ${turn.answer}`;

    })

    .join("\n\n");


  // --------------------------------
  // 5. Create prompt
  // --------------------------------

  const prompt = `You are a helpful assistant answering questions about the provided documents.

Use ONLY the context below to answer the question.

If the answer is not present in the context,
say "I don't know based on the provided information."

Use the conversation history only to understand
what a follow-up question refers to.

Never answer from conversation history alone.

Conversation so far:

${historyText || "(none yet)"}

Context:

${context}

New question:

${question}

Answer:`;


  // --------------------------------
  // 6. Ask Gemini to generate answer
  // --------------------------------

  const result = await ai.models.generateContent({

    model: "gemini-3.6-flash",

    contents: prompt

  });


  // --------------------------------
  // 7. Save conversation
  // --------------------------------

  conversationHistory.push({

    question: question,

    answer: result.text

  });


  return result.text;
}


// ===============================
// Command Line Interface
// ===============================

const rl = readline.createInterface({

  input: process.stdin,

  output: process.stdout

});


console.log(
  'RAG chatbot ready. Type a question, or "exit" to quit.\n'
);


// ===============================
// Chat Loop
// ===============================

function loop() {

  rl.question("You: ", async (input) => {

    if (
      input.trim().toLowerCase() === "exit"
    ) {

      rl.close();

      return;
    }


    try {

      const answer = await askQuestion(
        input
      );

      console.log(
        `\nBot: ${answer}\n`
      );

    } catch (error) {

      console.error(
        "\n❌ Error:",
        error.message
      );

    }


    loop();

  });

}


loop();