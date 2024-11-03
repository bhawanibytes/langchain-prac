import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import * as dotenv from "dotenv";
import { BufferMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";
dotenv.config();

// create model
const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.7
});

// create prompt
const prompt = ChatPromptTemplate.fromTemplate(`
    Your are an AI assitant.
    History: {history}
    {input}
`);

// creating buffer memory
const memory = new BufferMemory({
    memoryKey: "history"
});

// using chain classes
const chain = new ConversationChain({
    llm: model,
    prompt,
    memory
});

// using LCEL
// const chain = prompt.pipe(model);

// get responses
const inputs = {
    input: "Hello There"
}

const response = await chain.invoke(inputs);
console.log(response);