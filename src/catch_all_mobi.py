"""
Extract surface forms from epub files for Kindle vocab builder support.
"""

import argparse
import os
import re
import subprocess
import zipfile
import tempfile
import shutil
import time
from pathlib import Path
from html.parser import HTMLParser

import mobi

# Default language pair (source language -> translation language)
DEFAULT_SRC_LANG = "pl"
DEFAULT_DST_LANG = "en"

# Latin script with common European diacritics (covers most Western/Central/Eastern European languages)
LATIN_WORD_PATTERN = re.compile(r'[a-zA-ZàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿąćęłńóśźżĄĆĘŁŃÓŚŹŻčďěňřšťůžČĎĚŇŘŠŤŮŽőűŐŰßẞ]+', re.UNICODE)

# Paths relative to project root
DATA_DIR = Path(__file__).parent.parent / "data"
INPUTS_DIR = DATA_DIR / "inputs"
EBOOKS_DIR = INPUTS_DIR / "ebooks"
DICTIONARY_DIR = INPUTS_DIR / "dictionary"
OUTPUTS_DIR = DATA_DIR / "outputs"
SURFACE_FORMS_DIR = OUTPUTS_DIR / "surface_forms"
DICTIONARY_FORMS_DIR = OUTPUTS_DIR / "dictionary_forms"
OUTPUT_DICTIONARY_DIR = OUTPUTS_DIR / "dictionary"


class HTMLTextExtractor(HTMLParser):
    """Extract plain text from HTML content."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self._skip_data = False
        self._skip_tags = {'script', 'style', 'head', 'meta', 'link'}

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self._skip_tags:
            self._skip_data = True

    def handle_endtag(self, tag):
        if tag.lower() in self._skip_tags:
            self._skip_data = False

    def handle_data(self, data):
        if not self._skip_data:
            self.text_parts.append(data)

    def get_text(self):
        return ' '.join(self.text_parts)


def extract_text_from_epub(epub_path: Path) -> str:
    """Extract all text content from an epub file."""
    text_parts = []

    with zipfile.ZipFile(epub_path, 'r') as zf:
        for filename in zf.namelist():
            if filename.endswith(('.html', '.xhtml', '.htm')):
                try:
                    raw = zf.read(filename)
                    # Try multiple encodings
                    for encoding in ['utf-8', 'cp1250', 'iso-8859-2', 'latin-1']:
                        try:
                            content = raw.decode(encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        content = raw.decode('utf-8', errors='replace')

                    parser = HTMLTextExtractor()
                    parser.feed(content)
                    text_parts.append(parser.get_text())
                except Exception:
                    continue

    return '\n'.join(text_parts)


def extract_surface_forms(text: str) -> set[str]:
    """
    Extract unique surface forms (words) from text with smart filtering.

    Filtering rules:
    - Remove single characters
    - Remove pure numbers
    - Remove words with digits mixed in
    - Keep only words with actual letters
    - Preserve case (important for proper nouns in some languages)
    """
    words = LATIN_WORD_PATTERN.findall(text)

    # Apply filtering rules
    filtered_words = set()
    for word in words:
        # Skip single characters
        if len(word) <= 1:
            continue
        # Skip if contains digits (shouldn't happen with our regex, but safety check)
        if any(c.isdigit() for c in word):
            continue
        # Store lowercase for deduplication but preserve original for lookup
        filtered_words.add(word.lower())

    return filtered_words


def process_epub(epub_path: Path, output_dir: Path) -> int:
    """Process a single epub file and save surface forms to text file."""
    print(f"Processing: {epub_path.name}")

    # Extract text from epub
    text = extract_text_from_epub(epub_path)

    # Extract surface forms
    surface_forms = extract_surface_forms(text)

    # Create output filename (sanitize the epub name)
    output_name = epub_path.stem + "_surface_forms.txt"
    output_path = output_dir / output_name

    # Sort and save
    sorted_forms = sorted(surface_forms)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted_forms))

    print(f"  Found {len(sorted_forms)} unique surface forms -> {output_path.name}")
    return len(sorted_forms)


def process_all_epubs():
    """Process all epub files in the ebooks directory."""
    # Ensure output directory exists
    SURFACE_FORMS_DIR.mkdir(parents=True, exist_ok=True)

    # Find all epub files
    epub_files = list(EBOOKS_DIR.glob("*.epub"))

    if not epub_files:
        print(f"No epub files found in {EBOOKS_DIR}")
        return

    print(f"Found {len(epub_files)} epub file(s) in {EBOOKS_DIR}")
    print(f"Output directory: {SURFACE_FORMS_DIR}")
    print("-" * 60)

    total_forms = 0
    for epub_path in epub_files:
        try:
            count = process_epub(epub_path, SURFACE_FORMS_DIR)
            total_forms += count
        except Exception as e:
            print(f"  Error processing {epub_path.name}: {e}")

    print("-" * 60)
    print(f"Done! Processed {len(epub_files)} books, {total_forms} total surface forms extracted.")


def extract_dictionary_forms(mobi_path: Path) -> set[str]:
    """
    Extract all surface forms (headwords + inflections) from a MOBI dictionary.

    Parses idx:orth value="..." for headwords and idx:iform value="..." for inflections.
    """
    print(f"Extracting forms from dictionary: {mobi_path.name}")

    # Extract MOBI to temp directory
    tempdir, filepath = mobi.extract(str(mobi_path))

    try:
        # Find the HTML file containing dictionary entries
        html_path = Path(tempdir) / "mobi7" / "book.html"
        if not html_path.exists():
            # Try alternative locations
            for pattern in ["**/*.html", "**/*.htm"]:
                files = list(Path(tempdir).glob(pattern))
                if files:
                    html_path = max(files, key=lambda p: p.stat().st_size)
                    break

        if not html_path.exists():
            raise FileNotFoundError(f"No HTML file found in extracted MOBI: {tempdir}")

        # Read and parse the dictionary HTML
        with open(html_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        forms = set()

        # Extract headwords: <idx:orth value="word">
        orth_pattern = re.compile(r'<idx:orth[^>]*value="([^"]+)"', re.IGNORECASE)
        for match in orth_pattern.finditer(content):
            word = match.group(1).strip().lower()
            if word and len(word) > 1:
                forms.add(word)

        # Extract inflected forms: <idx:iform ... value="word"/>
        iform_pattern = re.compile(r'<idx:iform[^>]*value="([^"]+)"', re.IGNORECASE)
        for match in iform_pattern.finditer(content):
            word = match.group(1).strip().lower()
            if word and len(word) > 1:
                forms.add(word)

        print(f"  Found {len(forms)} unique forms")
        return forms

    finally:
        # Clean up temp directory
        shutil.rmtree(tempdir, ignore_errors=True)


def process_dictionary(mobi_path: Path, output_dir: Path) -> int:
    """Process a MOBI dictionary and save all known forms to a text file."""
    # Extract forms
    forms = extract_dictionary_forms(mobi_path)

    # Create output filename
    output_name = mobi_path.stem + "_forms.txt"
    output_path = output_dir / output_name

    # Sort and save
    sorted_forms = sorted(forms)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted_forms))

    print(f"  Saved to: {output_path.name}")
    return len(sorted_forms)


def process_all_dictionaries():
    """Process all MOBI dictionary files in the dictionary directory."""
    # Ensure output directory exists
    DICTIONARY_FORMS_DIR.mkdir(parents=True, exist_ok=True)

    # Find all mobi files
    mobi_files = list(DICTIONARY_DIR.glob("*.mobi"))

    if not mobi_files:
        print(f"No MOBI files found in {DICTIONARY_DIR}")
        return

    print(f"Found {len(mobi_files)} MOBI file(s) in {DICTIONARY_DIR}")
    print(f"Output directory: {DICTIONARY_FORMS_DIR}")
    print("-" * 60)

    total_forms = 0
    for mobi_path in mobi_files:
        try:
            count = process_dictionary(mobi_path, DICTIONARY_FORMS_DIR)
            total_forms += count
        except Exception as e:
            print(f"  Error processing {mobi_path.name}: {e}")

    print("-" * 60)
    print(f"Done! Processed {len(mobi_files)} dictionaries, {total_forms} total forms extracted.")


def load_all_surface_forms() -> set[str]:
    """Load all surface forms from all book files in the surface_forms directory."""
    all_forms = set()
    for txt_file in SURFACE_FORMS_DIR.glob("*.txt"):
        with open(txt_file, 'r', encoding='utf-8') as f:
            all_forms.update(line.strip().lower() for line in f if line.strip())
    return all_forms


def load_dictionary_forms() -> set[str]:
    """Load all known dictionary forms from the dictionary_forms directory."""
    all_forms = set()
    for txt_file in DICTIONARY_FORMS_DIR.glob("*.txt"):
        with open(txt_file, 'r', encoding='utf-8') as f:
            all_forms.update(line.strip().lower() for line in f if line.strip())
    return all_forms


def generate_dummy_entry(word: str) -> str:
    """Generate a minimal dummy entry for a word."""
    return f'<idx:entry scriptable="yes"><idx:orth value="{word}"></idx:orth><b>{word}</b><br/>.</idx:entry>'


def extract_dictionary_entries(mobi_path: Path) -> dict[str, str]:
    """
    Extract all dictionary entries indexed by their headwords and inflected forms.

    Returns a dict mapping each word (headword or iform) to its full <idx:entry> HTML block.
    """
    print(f"Extracting entries from dictionary: {mobi_path.name}")

    tempdir, _ = mobi.extract(str(mobi_path))

    try:
        html_path = Path(tempdir) / "mobi7" / "book.html"
        if not html_path.exists():
            for pattern in ["**/*.html", "**/*.htm"]:
                files = list(Path(tempdir).glob(pattern))
                if files:
                    html_path = max(files, key=lambda p: p.stat().st_size)
                    break

        if not html_path.exists():
            raise FileNotFoundError(f"No HTML file found in extracted MOBI: {tempdir}")

        with open(html_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        # Extract all <idx:entry>...</idx:entry> blocks
        entry_pattern = re.compile(r'<idx:entry[^>]*>.*?</idx:entry>', re.IGNORECASE | re.DOTALL)
        orth_pattern = re.compile(r'<idx:orth[^>]*value="([^"]+)"', re.IGNORECASE)
        iform_pattern = re.compile(r'<idx:iform[^>]*value="([^"]+)"', re.IGNORECASE)

        word_to_entry = {}
        entry_count = 0

        for match in entry_pattern.finditer(content):
            entry_html = match.group(0)
            entry_count += 1

            # Index by headword
            for orth_match in orth_pattern.finditer(entry_html):
                word = orth_match.group(1).strip().lower()
                if word and len(word) > 1:
                    word_to_entry[word] = entry_html

            # Index by inflected forms
            for iform_match in iform_pattern.finditer(entry_html):
                word = iform_match.group(1).strip().lower()
                if word and len(word) > 1 and word not in word_to_entry:
                    word_to_entry[word] = entry_html

        print(f"  Indexed {len(word_to_entry)} words from {entry_count} entries")
        return word_to_entry

    finally:
        shutil.rmtree(tempdir, ignore_errors=True)


def generate_entries_html(words: set[str], dict_entries: dict[str, str]) -> tuple[str, int, int]:
    """
    Generate HTML entries for words, using real definitions when available.

    Returns (html_string, real_count, dummy_count).
    """
    entries = []
    real_count = 0
    dummy_count = 0

    for word in sorted(words):
        if word in dict_entries:
            entries.append(dict_entries[word])
            real_count += 1
        else:
            entries.append(generate_dummy_entry(word))
            dummy_count += 1

    return '\n'.join(entries), real_count, dummy_count


def create_augmented_dictionary(src_lang: str = DEFAULT_SRC_LANG, dst_lang: str = DEFAULT_DST_LANG):
    """Create a standalone dictionary with entries for all words in ebooks, using real definitions when available."""
    OUTPUT_DICTIONARY_DIR.mkdir(parents=True, exist_ok=True)

    # Load all surface forms from ebooks
    print("Loading surface forms from ebooks...")
    surface_forms = load_all_surface_forms()
    print(f"  Loaded {len(surface_forms)} unique surface forms")

    if not surface_forms:
        print("No surface forms found. Run 'epubs' command first.")
        return

    # Find original dictionary and extract entries
    mobi_files = list(DICTIONARY_DIR.glob("*.mobi"))
    if mobi_files:
        print(f"\nExtracting entries from original dictionary...")
        dict_entries = extract_dictionary_entries(mobi_files[0])
    else:
        print("\nNo original dictionary found, using dummy entries for all words.")
        dict_entries = {}

    output_name = "ANKI_vocab_entries"
    build_dictionary(surface_forms, dict_entries, output_name, src_lang, dst_lang)


def create_single_book_dictionary(src_lang: str = DEFAULT_SRC_LANG, dst_lang: str = DEFAULT_DST_LANG):
    """Create a dictionary for a single selected book."""
    OUTPUT_DICTIONARY_DIR.mkdir(parents=True, exist_ok=True)

    # Find surface form files
    surface_files = sorted(SURFACE_FORMS_DIR.glob("*.txt"))
    if not surface_files:
        print("No surface form files found. Run 'epubs' command first.")
        return

    # List books with numbers
    print("Available books:")
    print("-" * 60)
    for i, f in enumerate(surface_files, 1):
        # Clean up the filename for display
        name = f.stem.replace("_surface_forms", "")
        if len(name) > 55:
            name = name[:52] + "..."
        print(f"  {i}. {name}")
    print("-" * 60)

    # Get user selection
    try:
        choice = input("Select book number (or 'q' to quit): ").strip()
        if choice.lower() == 'q':
            return
        idx = int(choice) - 1
        if idx < 0 or idx >= len(surface_files):
            print("Invalid selection.")
            return
    except ValueError:
        print("Invalid input.")
        return

    selected_file = surface_files[idx]
    book_name = selected_file.stem.replace("_surface_forms", "")
    print(f"\nSelected: {book_name}")

    # Load surface forms for this book
    print("Loading surface forms...")
    with open(selected_file, 'r', encoding='utf-8') as f:
        surface_forms = set(line.strip().lower() for line in f if line.strip())
    print(f"  Loaded {len(surface_forms)} unique words")

    # Find original dictionary and extract entries
    mobi_files = list(DICTIONARY_DIR.glob("*.mobi"))
    if mobi_files:
        print(f"\nExtracting entries from original dictionary...")
        dict_entries = extract_dictionary_entries(mobi_files[0])
    else:
        print("\nNo original dictionary found, using dummy entries for all words.")
        dict_entries = {}

    # Create safe filename
    safe_name = re.sub(r'[^\w\-]', '_', book_name)[:50]
    output_name = f"ANKI_{safe_name}"
    build_dictionary(surface_forms, dict_entries, output_name, src_lang, dst_lang)


def build_dictionary(surface_forms: set[str], dict_entries: dict[str, str], output_name: str,
                     src_lang: str = DEFAULT_SRC_LANG, dst_lang: str = DEFAULT_DST_LANG):
    """Build a MOBI dictionary from surface forms and dictionary entries."""
    tempdir = Path(tempfile.mkdtemp(prefix="ankify_"))
    print(f"Working directory: {tempdir}")

    try:
        # Generate entries for ALL surface forms, using real definitions where available
        print(f"Generating entries for {len(surface_forms)} words...")
        entries_html, real_count, dummy_count = generate_entries_html(surface_forms, dict_entries)
        print(f"  {real_count} with real definitions, {dummy_count} with dummy entries")

        # Create minimal dictionary HTML
        html_content = f'''<?xml version="1.0" encoding="utf-8"?>
<html xmlns:idx="www.mobipocket.com" xmlns:mbp="www.mobipocket.com" xmlns="http://www.w3.org/1999/xhtml">
<head><meta http-equiv="Content-Type" content="text/html; charset=utf-8"/></head>
<body>
<mbp:frameset>
{entries_html}
</mbp:frameset>
</body>
</html>'''

        html_path = tempdir / "dictionary.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Create OPF file
        opf_content = f'''<?xml version="1.0" encoding="utf-8"?>
<package unique-identifier="uid">
  <metadata>
    <dc-metadata xmlns:dc="http://purl.org/metadata/dublin_core">
      <dc:title>{output_name}</dc:title>
      <dc:language>{src_lang}</dc:language>
      <dc:creator>ankify-dictionary</dc:creator>
    </dc-metadata>
    <x-metadata>
      <DictionaryInLanguage>{src_lang}</DictionaryInLanguage>
      <DictionaryOutLanguage>{dst_lang}</DictionaryOutLanguage>
      <DefaultLookupIndex>{src_lang}</DefaultLookupIndex>
    </x-metadata>
  </metadata>
  <manifest>
    <item id="content" href="dictionary.html" media-type="text/x-oeb1-document"/>
  </manifest>
  <spine>
    <itemref idref="content"/>
  </spine>
</package>'''

        opf_path = tempdir / "dictionary.opf"
        with open(opf_path, 'w', encoding='utf-8') as f:
            f.write(opf_content)

        # Find kindlegen executable (Kindle Previewer 3 bundles it)
        kindlegen_paths = [
            Path(os.environ.get('LOCALAPPDATA', '')) / 'Amazon' / 'Kindle Previewer 3' / 'lib' / 'kindlegen.exe',
            Path(os.environ.get('PROGRAMFILES', '')) / 'Amazon' / 'Kindle Previewer 3' / 'lib' / 'kindlegen.exe',
            Path('kindlegen'),  # Fallback to PATH
        ]

        kindlegen_exe = None
        for path in kindlegen_paths:
            if path.exists():
                kindlegen_exe = str(path)
                print(f"Using kindlegen from: {path}")
                break

        if not kindlegen_exe:
            kindlegen_exe = 'kindlegen'  # Try PATH as last resort

        output_mobi = OUTPUT_DICTIONARY_DIR / f"{output_name}.mobi"
        print(f"Creating {output_mobi.name}...")

        try:
            start_time = time.time()
            result = subprocess.run(
                [kindlegen_exe, str(opf_path), '-c0', '-o', f"{output_name}.mobi"],
                capture_output=True,
                text=True,
                cwd=tempdir
            )
            elapsed = time.time() - start_time

            # kindlegen returns 1 for warnings, 2 for errors
            if result.returncode > 1:
                print(f"kindlegen error: {result.stderr}")
                print(f"stdout: {result.stdout}")
                return

            # Move output to final location
            generated_mobi = tempdir / f"{output_name}.mobi"
            if generated_mobi.exists():
                shutil.move(str(generated_mobi), str(output_mobi))
                print(f"\nSuccess! Created: {output_mobi} (kindlegen took {elapsed:.1f}s)")
                print(f"  {real_count} entries with real definitions")
                print(f"  {dummy_count} entries with dummy placeholders")
                print(f"\nUsage: Copy both this AND your original dictionary to Kindle.")
                print(f"  Kindle will search both dictionaries for lookups.")
            else:
                print(f"Error: Expected output file not found: {generated_mobi}")

        except FileNotFoundError:
            print("Error: kindlegen not found.")
            print("Please install Kindle Previewer 3: https://www.amazon.com/Kindle-Previewer/b?node=21381691011")

    finally:
        shutil.rmtree(tempdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Extract surface forms from epubs and dictionaries for Kindle vocab builder support."
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["epubs", "dictionary", "build", "build-single", "all"],
        default="all",
        help="Processing stage: 'epubs', 'dictionary', 'build' (all books), 'build-single' (pick one book), or 'all'"
    )
    parser.add_argument(
        "--src-lang", "-s",
        default=DEFAULT_SRC_LANG,
        help=f"Source language code for dictionary (default: {DEFAULT_SRC_LANG})"
    )
    parser.add_argument(
        "--dst-lang", "-d",
        default=DEFAULT_DST_LANG,
        help=f"Target/translation language code (default: {DEFAULT_DST_LANG})"
    )
    args = parser.parse_args()

    if args.command in ("epubs", "all"):
        process_all_epubs()
        if args.command == "all":
            print()

    if args.command in ("dictionary", "all"):
        process_all_dictionaries()
        if args.command == "all":
            print()

    if args.command == "build":
        create_augmented_dictionary(args.src_lang, args.dst_lang)

    if args.command == "build-single":
        create_single_book_dictionary(args.src_lang, args.dst_lang)

    if args.command == "all":
        create_augmented_dictionary(args.src_lang, args.dst_lang)


if __name__ == "__main__":
    main()
