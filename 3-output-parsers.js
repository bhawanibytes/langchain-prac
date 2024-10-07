import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser, CommaSeparatedListOutputParser } from '@langchain/core/output_parsers'
import { StructuredOutputParser } from "langchain/output_parsers";
import { z } from "zod";
import * as dotenv from 'dotenv';
dotenv.config();

// Create model
const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.7,
    maxTokens: 1000,
    verbose: false
});

async function callStringOutputParser() {
    // Create Prompt Template
    const prompt  = ChatPromptTemplate.fromMessages([
        ["system", "generate a joke based on a word provided by the user."],
        ["human", "{input}"]
    ]);
    
    // Create String Parser
    const parser = new StringOutputParser();
    
    // Create Chain
    const chain = prompt.pipe(model).pipe(parser);
    
    //Call Chain
    return await chain.invoke({
        input: "dog"
    });
    
}

async function callListOutputParser() {
    //create prompt
    const prompt = ChatPromptTemplate.fromTemplate(`
        Provide 5 synonyms, seperated by commas, for the following word {word}
        `)
    // create commaSeperatedListOutputParser
    const parser = new CommaSeparatedListOutputParser();
    // chain
    const chain = prompt.pipe(model).pipe(parser);
    //returning the chain while calling it
    return await chain.invoke({
        word: "happy"
    })
}

async function callStructureParser() {
    //create prompt
    const prompt = ChatPromptTemplate.fromTemplate(`
        Extract information from the following phrase.
        Formatting Instructions: {format_instructions}
        Phrase: {phrase}
        `);
    // create parser
    const parser = StructuredOutputParser.fromNamesAndDescriptions({
        name: "the name of the person",
        age: "the age of the person"
    });
    //create chain
    const chain = prompt.pipe(model).pipe(parser);
    // returning chain while invoking it
    return await chain.invoke({
        phrase: "max is 30 years old",
        format_instructions: parser.getFormatInstructions()
    })
}

async function callZodParser() {
    //create prompt
    const prompt = ChatPromptTemplate.fromTemplate(`
        Extract information from the following phrase.
        Formatting Instructions: {format_instructions}
        Phrase: {phrase}
        `);
    // create parser
    const parser = StructuredOutputParser.fromZodSchema(
        z.object({
            recipe: z.string().describe("name of recipe"),
            ingrediant: z.array(z.string()).describe("ingrediants")
        })
    );
    //create chain
    const chain = prompt.pipe(model).pipe(parser);
    // returning chain while invoking it
    return await chain.invoke({
        phrase: "The ingrediants for a speghettiu Bolognese recipe are tomatoes, minced beef, garlic, wine and herbs.",
        format_instructions: parser.getFormatInstructions()
    })
}

// const response = await callStringOutputParser();
// const response = await callListOutputParser();
// const response = await callStructureParser();
const response = await callZodParser();
console.log(response);