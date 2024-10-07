import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import * as dotenv from 'dotenv';
dotenv.config();

// load data and create in memory vector store
const createVectorStore = async () => {
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
    return vectorStore
}

// create retrieval chain
const createChain = async () => {
    // Create model
    const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.7,
    maxTokens: 1000,
    verbose: false
    });
    // create  prompt
    const prompt = ChatPromptTemplate.fromMessages([
        [
            "system",
            "Answer the user's question based on the following context: {context}."
        ],
        new MessagesPlaceholder("chat_history"),
        [
            "user", "{input}"
        ]
    ]);
    // retrieving doc from vectore store
    const retriever = vectorStore.asRetriever({
        k: 2
    });
    const retrieverPrompt = ChatPromptTemplate.fromMessages([
        new MessagesPlaceholder("chat_history"),
        ["user", "{input}"],
        ["user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation"]
    ])
    // history aware retriever 
    const historyAwareRetriever = await createHistoryAwareRetriever({
        llm: model,
        retriever,
        rephrasePrompt: retrieverPrompt
    }) 
    
    //creating document supoorting chain
    const chain = await createStuffDocumentsChain({
        llm: model,
        prompt
    });
    
    const conversationChain = await createRetrievalChain({
        combineDocsChain: chain,
        retriever: historyAwareRetriever
    });

    return conversationChain;
}

const vectorStore = await createVectorStore();
const chain = await createChain(vectorStore);

//chat history
const chatHistory = [
    new HumanMessage("Hello"),
    new AIMessage("Hi, how can I help you?"),
    new HumanMessage("My  name is Leon"),
    new AIMessage("Hi Leon, how can I help you?"),
    new HumanMessage("What is LCEL?"),
    new AIMessage("LCEL stands for Langchain Expression Language")
]

// invoke chain and store response
const response = await chain.invoke({
    input: "What is it?",
    chat_history: chatHistory
})

console.log(response.answer);