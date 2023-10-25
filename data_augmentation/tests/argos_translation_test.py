import argparse
import argostranslate.package
import argostranslate.translate

# NOTE: Translation for Chinese still has issues as of writing:
# https://github.com/argosopentech/argos-translate/issues/225
# NOTE: Polish translation does not seem to work as of writing


def get_language_choices():
    # Returns dict of 2 letter code -> full name
    return {
        "en": "English",
        "ar": "Arabic",
        "az": "Azerbaijani",
        "ca": "Catalan",
        "cs": "Czech",
        "da": "Danish",
        "nl": "Dutch",
        "eo": "Esperanto",
        "fi": "Finnish",
        "fr": "French",
        "de": "German",
        "el": "Greek",
        "he": "Hebrew",
        "hi": "Hindi",
        "hu": "Hungarian",
        "id": "Indonesian",
        "ga": "Irish",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "fa": "Persian",
        "pl": "Polish",
        "pt": "Portuguese",
        "ru": "Russian",
        "sk": "Slovak",
        "es": "Spanish",
        "sv": "Swedish",
        "th": "Thai",
        "tr": "Turkish",
        "uk": "Ukrainian",
        "zh": "Mandarin",
    }


def parse_args():
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="Hello world!")
    parser.add_argument("--to-code", choices=get_language_choices(), default="fr")
    parser.add_argument("--from-code", choices=get_language_choices(), default="en")
    parser.add_argument("--all", action="store_true", help="Translate to all languages")
    return parser.parse_args()


def translate_text(text, from_code, to_code):
    # Translate text using Argos Translate

    # Download and install translation package
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(
            lambda x: x.from_code == from_code and x.to_code == to_code,
            available_packages,
        )
    )
    argostranslate.package.install_from_path(package_to_install.download())

    # Translate text
    translated_text = argostranslate.translate.translate(text, from_code, to_code)

    return translated_text


def main(args):
    text = args.text
    from_code = args.from_code
    to_code = args.to_code
    if args.all:
        for to_code in get_language_choices():
            if from_code == to_code:
                print(
                    f"Skipping translation from {from_code} to {to_code} (same as source)"
                )
                print()
                continue

            print(f"Translating '{text}' from {from_code} to {to_code}")
            translated = translate_text(text, from_code, to_code)
            print(translated)
            print()
    else:
        # Translate normally
        translated = translate_text(text, from_code, to_code)
        print(translated)


if __name__ == "__main__":
    args = parse_args()
    main(args)
