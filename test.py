from base import generate_completion, prepare_analysis, run_question, load_config
import sys


def main():
    if len(sys.argv) < 2:
        print("Please provide a config path as the first argument.", file=sys.stderr)
        sys.exit(1)

    load_config(sys.argv[1])

    # Useful for quickly testing if the generation works
    events = generate_completion({
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "stopAt": [],
        "stopAfter": [],
        "maxParagraphs": 0,
        "maxCharacters": 0,
        "maxSafetyCharacters": 0,
        "startCountingFromToken": None,
        "trail": None
    })

    for event in events:
        if 'error' in event:
            print("Error during generation:", event['error'])
        elif 'request_id' in event:
            print("Request ID:", event['request_id'])
        elif 'done' in event and event['done']:
            print("Generation done.")

    events2 = prepare_analysis({
        "system": "You are an expert in literature analysis, the user will provide you with stories to analyze and ask questions about them.",
        "userTrail": "<story>The story is about a young hero embarking on a quest.\n\nThe hero faces many challenges along the way.\n\nThe hero name is Arin.</story>"
    })

    for event in events2:
        if 'error' in event:
            print("Error during generation:", event['error'])

    events3 = run_question({
        "question": "<question>What is the main theme of the story?</question>",
        "stopAt": ["."],
        "stopAfter": [],
        "maxParagraphs": 100,
        "maxCharacters": 500,
        "maxSafetyCharacters": 0,
        "trail": "<answer>The main theme is ",
        "grammar": 'root ::= "well I don\'t really know " [a-zA-Z0-9 _-]+ "."',
    })

    for event in events3:
        if 'error' in event:
            print("Error during question answering:", event['error'])
        elif 'answer' in event:
            print("Answer:", event['answer'])

    events4 = run_question({
        "question": "<question>Who is the protagonist of the story?</question>",
        "stopAt": ["</answer>", ".", ",", ";"],
        "maxParagraphs": 1,
        "maxCharacters": 500,
        "maxSafetyCharacters": 0,
        "stopAfter": [],
        "trail": "<answer>The protagonist name is ",
        "grammar": None,
    })

    for event in events4:
        if 'error' in event:
            print("Error during question answering:", event['error'])
        elif 'answer' in event:
            print("Answer:", event['answer'])

    events5 = run_question({
        "question": "<question>Is the hero named Arin? Answer with YES or NO.</question>",
        "stopAt": ["</answer>", ".", ",", ";"],
        "maxParagraphs": 1,
        "maxCharacters": 500,
        "maxSafetyCharacters": 0,
        "stopAfter": ["yes", "no"],
        "trail": "<answer>",
        "grammar": None,
    })

    for event in events5:
        if 'error' in event:
            print("Error during question answering:", event['error'])
        elif 'answer' in event:
            print("Answer:", event['answer'])


if __name__ == "__main__":
    main()
