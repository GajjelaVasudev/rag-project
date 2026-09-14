import "dotenv/config";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

console.log("OpenAI client initialized");
console.log("Pinecone client initialized");

async function testConnection() {
  try {
    const indexes = await pinecone.listIndexes();

    console.log("\nPinecone indexes:");
    console.log(indexes);

    const indexName = process.env.PINECONE_INDEX_NAME;

    const index = pinecone.index(indexName);

    const stats = await index.describeIndexStats();

    console.log("\nIndex statistics:");
    console.log(stats);

    console.log("\n✅ Connection successful!");
  } catch (error) {
    console.error("\n❌ Connection failed:");
    console.error(error);
  }
}

testConnection();