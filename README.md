# ankify-dictionary

Creates custom Kindle dictionaries that ensure **every word lookup** gets logged to Vocab Builder - even words not in the original dictionary.

## How it works

1. Extracts all words from your epub books
2. Pulls real definitions from your existing MOBI dictionary where available
3. Creates dummy entries for words not in the dictionary
4. Builds a new MOBI dictionary with entries for every word in your books

## Setup

```bash
pip install mobi
```

Requires [Kindle Previewer 3](https://www.amazon.com/Kindle-Previewer/b?node=21381691011) (bundles kindlegen).

## Directory structure

```
data/
  inputs/
    ebooks/       # Place .epub files here
    dictionary/   # Place source .mobi dictionary here
  outputs/
    dictionary/   # Generated dictionaries appear here
```

## Usage

```bash
cd src

# Extract words from all epubs
py ankify_dictionary.py epubs

# Extract indexed forms from dictionary (optional, for stats)
py ankify_dictionary.py dictionary

# Build dictionary for a single book (interactive selection)
py ankify_dictionary.py build-single

# Build dictionary for all books combined
py ankify_dictionary.py build

# Run all steps
py ankify_dictionary.py all
```

## Kindle setup

1. Copy the generated `ANKI_*.mobi` to your Kindle's `documents` folder
2. On Kindle: Settings → Language & Dictionaries → Dictionaries → Select the ANKI dictionary for your language
3. Read and look up words - all lookups now appear in Vocab Builder
