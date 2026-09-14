require("dotenv").config();

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

  // 1. Convert question into vector
  const queryVector = await embedQuery(
    question
  );

  console.log(
    "Query vector dimension:",
    queryVector.length
  );


  // 2. Search Pinecone
  const results = await index.query({

    vector: queryVector,

    topK: 3,

    includeMetadata: true

  });


  // 3. Display retrieved chunks

  console.log("\n--- Retrieved chunks ---");

  results.matches.forEach((match, i) => {

    console.log(
      `[${i}] score=${match.score.toFixed(3)} ` +
      `source=${match.metadata.source}`
    );

  });


  // 4. Extract text from metadata

  const context = results.matches

    .map((match) => match.metadata.text)

    .join("\n\n---\n\n");


  // 5. Build prompt

  const prompt = `You are a helpful assistant.

Answer the question using ONLY the information
provided in the context below.

If the answer is not present in the context,
say "I don't know based on the provided information."

Context:
${context}

Question:
${question}

Answer:`;


  // 6. Ask Gemini to generate the answer

  const result = await ai.models.generateContent({

    model: "gemini-3.6-flash",

    contents: prompt

  });


  // 7. Display answer

  console.log("\n--- Answer ---");

  console.log(result.text);

}


// ===============================
// Get Question From Command Line
// ===============================

const question =
  process.argv.slice(2).join(" ") ||
  "What is this about?";


askQuestion(question).catch((error) => {

  console.error("\n❌ Query failed:");

  console.error(error);

});