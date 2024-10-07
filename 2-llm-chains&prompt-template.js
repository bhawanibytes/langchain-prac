import { ChatOpenAI } from "@langchain/openai";
import {ChatPromptTemplate} from "@langchain/core/prompts";
import * as dotenv from 'dotenv';
dotenv.config();

// Create model
const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.7,
    maxTokens: 1000,
    verbose: true
});

// Create Prompt Template
// const prompt  = ChatPromptTemplate.fromTemplate(
//     'You are a comedian. Tell a joke based on the following word {input}'
// );

// Create Prompt Template
const prompt  = ChatPromptTemplate.fromMessages([
    ["system", "generate a joke based on a word provided by the user."],
    ["human", "{input}"]
]);


// console.log(await prompt.format({input: "chicken"}));

// Create Chain
const chain = prompt.pipe(model);

//Call Chain
const response = await chain.invoke({
    input: "bird & dog"
});

console.log(response);