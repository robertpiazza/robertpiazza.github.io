# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 09:29:38 2022

@author: rober
"""

import re, string, sys

INCLUDED_LETTERS = ''

EXCLUDED_LETTERS = ''

KNOWN_LETTERS ='#####'



# Init ArguementParser
parser = argparse.ArgumentParser()

# A string representing a wordle row, with # for uncertain placeholders
parser.add_argument("input_string", help="world row with # for unknowns")

# Includes are letters known to be in the word
parser.add_argument("-i", "--include", help="characters that are for sure included")

# Excludes are letters known to not be in the words
parser.add_argument("-e", "--exclude", help="characters that are for sure exluded")

# Additional args for verbose output and writing to file
parser.add_argument("-v", "--verbose", help="output all possible information", action="store_true")
parser.add_argument("-w", "--write", help="write to file at the current directory instead of echoing output", action="store_true")

# Parse Arguements 
args = parser.parse_args()

verbose = args.verbose
if verbose:
    print("Arguments")
    print(args, end="\n\n")
    print("[GIVEN]: {}".format(args.input_string))

# Reject invalid lengths
if len(args.input_string) != 5:
    sys.exit("Invalid number of characters: {}. Expected 5".format(len(args.input_string)))

validated = []
# Ensure given string has only expected chars
unvalidated_string = KNOWN_LETTERS
for char in unvalidated_string:

    # If char is valid add to validated string as lowercase
    if char in string.ascii_lowercase or char in string.ascii_uppercase or char == "#":
        if char == "#":
            validated.append(char)
        else:
            validated.append(char.lower())


# Generate valid string from validated
validated_string = "".join([i for i in validated])

unfinished_regex = []
# Turn validated string into a regex string
for c in validated_string:
    if c == "#":
        unfinished_regex.append("[a-z]")
    else:
        unfinished_regex.append("[{}]".format(c))

# Create simple regex
wordle_regex = "".join([j for j in unfinished_regex])

possible_words = []
# Iterate through dictionary of five letter words with regex
with open("five-letter-words.txt") as f:
    for line in f:
        # If the line matches our regular expressions
        if (re.findall(wordle_regex, line)):
            possible_words.append(line)

words_excluded = 0
excludes_parsed = []
# If passed chars to exclude, filter the words 
if args.exclude:
    for word in possible_words:
        # Iterate through all words counting any excludes
        word_has_excludes = 0

        # Iterate through each char in word comparing to excludes
        for char in word:
            for exclude in args.exclude:
                if char == exclude:
                    word_has_excludes += 1 
        
        # Only add words to list without excludes
        if word_has_excludes == 0: 
            excludes_parsed.append(word)
        else:
            words_excluded += 1

    # Update list of possible words
    possible_words = excludes_parsed
    
includes_parsed = []
# If passed chars that are known in the word
if args.include:
    
    # Iterate through all possible words
    for word in possible_words:
        has_includes = 0

        # Iterate through every known letter, per possible word
        for include in args.include:
            if include in word:
                has_includes += 1

        # If the word has the same number of includes, add it to the list
        if has_includes == len(args.include):
            includes_parsed.append(word)

    # Update the list of possible words
    possible_words = includes_parsed

# Sort the list
possible_words.sort()

# If there is only one result, only echo do not print
if len(possible_words) == 1:
    for line in possible_words:
        print("The Solution is:\n\n{}".format(line))
        sys.exit()

# If there is a result echo it out or write to file
if len(possible_words) != 0:
    
    if args.write:
        with open("./possible_words.txt", "w") as wf:
            for word in possible_words:
                wf.write(word)
    else:
        print("Possible Solutions:\n")
        [print(i, end="") for i in possible_words]

        if verbose:
            print("Number of words excluded: {}".format(words_excluded))

    print("\nFound: {}".format(len(possible_words)))
        
# Else there are no words, double check your inputs
else:
    print("No Results")