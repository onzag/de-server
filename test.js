import { generateCompletion, prepareAnalysis, runQuestion } from "./base.js";

// Useful for quickly testing if the generation works
await generateCompletion({
    messages: [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "Hello, how are you?" }
    ],
    stopAt: [],
    stopAfter: [],
    maxParagraphs: 0,
    maxCharacters: 0,
    startCountingFromToken: null,
    trail: null
}, (text) => {
}, () => {
    console.log("\nGeneration complete.");
}, (error) => {
    console.log("Error during generation:", error.message);
});

await prepareAnalysis({
    system: "You are an expert in literature analysis, the user will provide you with stories to analyze and ask questions about them.",
    userTrail: "<story>The story is about a young hero embarking on a quest.\n\nThe hero faces many challenges along the way.\n\nThe hero name is Arin.</story>"
}, () => {
    console.log("Analysis prepared successfully.");
}, (error) => {
    console.log("Error during analysis preparation:", error.message);
});

await runQuestion({
    question: "<question>What is the main theme of the story?</question>",
    stopAt: [
        "."
    ],
    stopAfter: [],
    maxParagraphs: 100,
    maxCharacters: 500,
    trail: "<answer>The main theme is ",
    grammar: `root ::= "well I don't really know " [a-zA-Z0-9 _-]+ "."`,
}, (answer) => {
    console.log(answer);
}, (error) => {
    console.log("Error during question answering:", error.message);
});

await runQuestion({
    question: "<question>Who is the protagonist of the story?</question>",
    stopAt: [
        "</answer>",
        ".",
        ",",
        ";"
    ],
    maxParagraphs: 1,
    maxCharacters: 500,
    stopAfter: [],
    trail: "<answer>The protagonist name is ",
    grammar: null,
}, (answer) => {
    console.log(answer);
}, (error) => {
    console.log("Error during question answering:", error.message);
});

await runQuestion({
    question: "<question>Is the hero named Arin? Answer with YES or NO.</question>",
    stopAt: [
        "</answer>",
        ".",
        ",",
        ";"
    ],
    maxParagraphs: 1,
    maxCharacters: 500,
    stopAfter: [
        "yes",
        "no"
    ],
    trail: "<answer>",
    grammar: null,
}, (answer) => {
    console.log(answer);
}, (error) => {
    console.log("Error during question answering:", error.message);
});