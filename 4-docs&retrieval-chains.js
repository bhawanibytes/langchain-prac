import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
// import { Document } from "@langchain/core/documents";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import * as dotenv from 'dotenv';
dotenv.config();

// Create model
const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.7,
    maxTokens: 1000,
    verbose: false
});
// create  prompt
const prompt = ChatPromptTemplate.fromTemplate(`
    Answer the user's question.
    Contect: {context}
    Question: {input}
    `)
// create document
// const documentA = new Document({
//     pageContent: 'LangChain Expression Language LCEL LangChain Expression Language or LCEL is a declarative way to easily compose chains together. Any chain constructed this way will automatically have full sync, async, and streaming support. If youre looking for a good place to get started, check out the Cookbook section - it shows off the various Expression Language pieces in order from simple to more complex'
// })
// const documentB = new Document({
//     pageContent: "passPhrase is LANGCHAIN IS AWESOME"
// })

// create normal chain
// const chain = prompt.pipe(model);

//creating web content loader
const loader = new CheerioWebBaseLoader(
    "https://js.langchain.com/v0.1/docs/expression_language/"
)

// creating doc from loader by calling it
const docs = await loader.load();

// creating text splitter
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 20
});

// splitting in varouis docs
const splitDocs = await splitter.splitDocuments(docs);

// embbeding
const embeddings = new OpenAIEmbeddings();

// putting docs and embbeding in memory Vectore Store
const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
);

// retrieving doc from vectore store
const retriever = vectorStore.asRetriever({
    k: 2
});

//creating document supoorting chain
const chain = await createStuffDocumentsChain({
    llm: model,
    prompt
})

const retrievalChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever
})

// invoke chain and store response
const response = await retrievalChain.invoke({
    input: "What is LCEL? And what is passPhrase?"
})

console.log(response);