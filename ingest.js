require('dotenv').config();

const fs = require('fs');
const path = require('path');
const { GoogleGenAI } = require('@google/genai');
const { Pinecone } = require('@pinecone-database/pinecone');


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
// Chunk Text
// ===============================

function chunkText(text, size = 400, overlap = 50) {
  const words = text.split(/\s+/);

  const chunks = [];

  let start = 0;

  while (start < words.length) {

    const end = Math.min(
      start + size,
      words.length
    );

    chunks.push(
      words
        .slice(start, end)
        .join(' ')
    );

    if (end === words.length) {
      break;
    }

    start += size - overlap;
  }

  return chunks;
}


// ===============================
// Generate Embeddings
// ===============================

async function embedBatch(texts) {

  const response = await ai.models.embedContent({
    model: 'gemini-embedding-001',

    contents: texts,

    config: {
      outputDimensionality: 1536
    }
  });

  return response.embeddings.map(
    (embedding) => embedding.values
  );
}


// ===============================
// Main Ingestion Function
// ===============================

async function main() {

  const dataDir = './data';

  // Check if data folder exists
  if (!fs.existsSync(dataDir)) {

    console.error(
      '❌ Data directory does not exist:',
      dataDir
    );

    return;
  }


  // Get all .txt files
  const files = fs
    .readdirSync(dataDir)
    .filter((file) => file.endsWith('.txt'));


  if (files.length === 0) {

    console.log(
      '⚠️ No .txt files found inside ./data'
    );

    return;
  }


  let allRecords = [];


  // ===============================
  // Process Each File
  // ===============================

  for (const file of files) {

    const fullPath = path.join(
      dataDir,
      file
    );


    // Read file
    const text = fs.readFileSync(
      fullPath,
      'utf-8'
    );


    // Split into chunks
    const chunks = chunkText(text);


    console.log(
      `${file}: split into ${chunks.length} chunks`
    );


    // Generate embeddings
    const vectors = await embedBatch(
      chunks
    );


    // Create Pinecone records
    chunks.forEach((chunkStr, i) => {

      allRecords.push({

        id: `${file}-chunk-${i}`,

        values: vectors[i],

        metadata: {
          text: chunkStr,
          source: file
        }

      });

    });

  }


  // ===============================
  // Upload to Pinecone
  // ===============================

  console.log(
    `Uploading ${allRecords.length} vectors to Pinecone...`
  );


 await index.upsert({
  records: allRecords
});


  console.log(
    `✅ Upserted ${allRecords.length} chunks into Pinecone.`
  );


  // ===============================
  // Show Pinecone Stats
  // ===============================

  const stats =
    await index.describeIndexStats();


  console.log(
    '📊 New index stats:',
    stats
  );

}


// ===============================
// Run Application
// ===============================

main().catch((error) => {

  console.error(
    '❌ Ingestion failed:'
  );

  console.error(error);

});