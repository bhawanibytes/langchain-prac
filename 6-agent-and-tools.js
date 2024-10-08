import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { AgentExecutor, createOpenAIFunctionsAgent } from "langchain/agents";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import readline, { createInterface } from "readline";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import * as dotenv from "dotenv";
import { createRetrieverTool } from "langchain/tools/retriever";
dotenv.config();

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



// create model
const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo-1106",
    temperature: 0.7
});

// create prompt
const prompt = ChatPromptTemplate.fromMessages([
    ("system", "You are a helpful assitant, called Bhawani."),
    new MessagesPlaceholder("chat_history"),
    ("human","{input}"),
    new MessagesPlaceholder("agent_scratchpad")
]);

// create and assign tools
const searchTool = new TavilySearchResults();
const retrieverTool = new createRetrieverTool(retriever,{
    name: "lcel_search",
    description: "Use this tool when searching for information about Langchain Expression Language (LCEL)"
})
const tools = [ searchTool, retrieverTool ];

// create agent
const agent = await createOpenAIFunctionsAgent({
    llm: model,
    prompt,
    tools
});

// create agent executor
const agentExecutor = new AgentExecutor({
    agent,
    tools
});

// get user input
const rl = createInterface({
    input: process.stdin,
    output: process.stdout
});

// chat history array
const chatHistory = [];

const askQuestions = () => {
    rl.question("User: ", async (input) => {
        if(input.toLowerCase() === "exit"){
            rl.close();
            return;
        }
        //call agent
        const response = await agentExecutor.invoke({
            input,
            chat_history: chatHistory
        });
        console.log("Agent:", response.output);
        chatHistory.push(new HumanMessage(input));
        chatHistory.push(new AIMessage(response.output));
        askQuestions();
    });
}

askQuestions();