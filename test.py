from base import generate_completion, prepare_analysis, run_question, load_config
import sys


def main():
    if len(sys.argv) < 2:
        print("Please provide a config path as the first argument.", file=sys.stderr)
        sys.exit(1)

    load_config(sys.argv[1])

    # Useful for quickly testing if the generation works
    generate_completion({
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "stopAt": [],
        "stopAfter": [],
        "maxParagraphs": 0,
        "maxCharacters": 0,
        "startCountingFromToken": None,
        "trail": None
    }, lambda rqid: None,
       lambda text: None,
       lambda: print("\nGeneration complete."),
       lambda error: print("Error during generation:", str(error)))

    prepare_analysis({
        "system": "You are an expert in literature analysis, the user will provide you with stories to analyze and ask questions about them.",
        "userTrail": "<story>The story is about a young hero embarking on a quest.\n\nThe hero faces many challenges along the way.\n\nThe hero name is Arin.</story>"
    }, lambda: print("Analysis prepared successfully."),
       lambda error: print("Error during analysis preparation:", str(error)))

    run_question({
        "question": "<question>What is the main theme of the story?</question>",
        "stopAt": ["."],
        "stopAfter": [],
        "maxParagraphs": 100,
        "maxCharacters": 500,
        "trail": "<answer>The main theme is ",
        "grammar": 'root ::= "well I don\'t really know " [a-zA-Z0-9 _-]+ "."',
    }, lambda rqid: None, lambda answer: print(answer),
       lambda error: print("Error during question answering:", str(error)))

    run_question({
        "question": "<question>Who is the protagonist of the story?</question>",
        "stopAt": ["</answer>", ".", ",", ";"],
        "maxParagraphs": 1,
        "maxCharacters": 500,
        "stopAfter": [],
        "trail": "<answer>The protagonist name is ",
        "grammar": None,
    }, lambda rqid: None, lambda answer: print(answer),
       lambda error: print("Error during question answering:", str(error)))

    run_question({
        "question": "<question>Is the hero named Arin? Answer with YES or NO.</question>",
        "stopAt": ["</answer>", ".", ",", ";"],
        "maxParagraphs": 1,
        "maxCharacters": 500,
        "stopAfter": ["yes", "no"],
        "trail": "<answer>",
        "grammar": None,
    }, lambda rqid: None, lambda answer: print(answer),
       lambda error: print("Error during question answering:", str(error)))


if __name__ == "__main__":
    main()
