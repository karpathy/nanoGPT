"""
Tasteful utilities -- ala CKG
"""

import os
import io
import sys
import json
from typing import List, Dict




#### COLORS ####
# Map Color Names to Terminal Color Codes
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=39,
    #additional colors do string highlighting (i.e, 42 prints strings highlighted green w white text) or are plain white
    red_highlight=41,
    green_highlight=42,
    yellow_highlight=43,
    blue_highlight=44,
)

def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman, the policy optimization guy
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

# Main Util to print output nice (cp = color print)
def cp(*objects, color=None, bold=False, highlight=False, **kwargs):
    """
    item: stdout, this is the thing you want to print (typically any STDOUT should do)
    color: string, of one of the supported colors (will print out the item with this color instead of the default STDOUT)
    bold: bool, flag to make the output Bold or not (default False)
    highlight: bool, flag to higlight the output or not (default False)

    Supported colors are:
    [gray, red, green, yellow, blue, magenta, cyan, white, crimson]
      *the colors above will change from terminal to terminal, as they are related to a User's current color scheme
    """

    if color == None:
        print(*objects)
    else:
        try:
            sep = kwargs["sep"] if "sep" in kwargs else " "
            objects = [str(obj) for obj in objects]
            objects = sep.join(objects)
            objects = colorize(objects, color, bold, highlight)
            print(objects)
        except Exception as err:
            print(f"Object to be printed with {color} cannot be turned into a string (recieved error: '{err}'), defaulting to regular print instead:")
            print(objects)

# Quick Utils for [Errors, Warnings, Logs/Info, Successes]
def printerr(*objects):
    cp(*objects, color="red")

def printwar(*objects):
    cp(*objects, color="yellow")

def printlog(*objects):
    cp(*objects, color="blue")

def printok(*objects):
    cp(*objects, color="green")


# taken from the catppuccin mocha colorscheme
cat_colors_dict = {
    "rosewater":        "#f5e0dc", 
    "flamingo":         "#f2cdcd", 
    "pink":             "#f5c2e7", 
    "mauve":            "#cba6f7", 
    "red":              "#f38ba8", 
    "maroon":           "#eba0ac", 
    "peach":            "#fab387", 
    "yellow":           "#f9e2af", 
    "green":            "#a6e3a1", 
    "teal":             "#94e2d5", 
    "sky":              "#89dceb", 
    "sapphire":         "#74c7ec", 
    "blue":             "#89b4fa", 
    "lavender":         "#b4befe", 
    "text":             "#cdd6f4", 
    "subtext1":         "#bac2de", 
    "subtext0":         "#a6adc8", 
    "overlay2":         "#9399b2", 
    "overlay1":         "#7f849c", 
    "overlay0":         "#6c7086", 
    "surface2":         "#585b70", 
    "surface1":         "#45475a", 
    "surface0":         "#313244", 
    "base":             "#1e1e2e", 
    "mantle":           "#181825", 
    "crust":            "#11111", 
}


#### SYSTEM ####
def mkdir(path: str, verbose: bool=False):
    """ Helper function to make dirs and subdirs for a given path """
    if os.path.exists(path):
        if verbose: print(f"Path: '{path}' already exists, leaving as is")
    else:
        os.makedirs(path)

# Mess with print statements
class HiddenPrints:
    """ context manage to surpress unwanted print statements """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
    # one can use this class like:
    # with HiddenPrints():
    #     print("wow")
    # 'wow' will not print to stdout, it will be sent to the shadowrealm

def clear_screen():
    """ a programmatic equivalent of typing 'clear' in your terminal, TUI anyone? """
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
 
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')

def procinfo(title):
    """ print information about a process, pid et al """
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())


#### DATA ####
# Extract Dicts from Local File Paths
def read_json_data(filepath):
    if os.path.exists(filepath):
        file = io.open(filepath, mode="r", encoding="utf-8")
        data = json.load(file)
        file.close()
        return data #dictionary for parsing/uploading
    else:
        print(f"File: '{filepath}' doesn't seem to exist, returning empty dictionary")
        return {} #base for creating any new data dicts (writer util can work with one entry)

# Write out JSONs w Human-Readable indents
def write_json_data(filepath, data):
    with io.open(filepath, mode="w", encoding="utf-8") as f:
        file = json.dumps(data, indent=4, ensure_ascii=False)
        f.write(file)
        f.close()

def write_txt_data(filepath, data):
    """ Dump a String to a text file, should refactor if ever using """
    with io.open(filepath, mode="w", encoding="utf-8") as f:
        f.write(data)
        f.close()

def read_txt_data(filepath):
    """ Read in text from filepath as an array of strings """
    with open(filepath, "r") as f:
        lines = f.readlines()
    return lines

def read_jsonl_data(filepath: str) -> List[Dict]:
    if os.path.exists(filepath):
        data = []
        file = io.open(filepath, mode="r", encoding="utf-8")
        for line in file:
            data.append(json.loads(line))
        file.close()
        return data #list of dicts per line in JSON
    else:
        print(f"File: '{filepath}' doesn't seem to exist")
        return {}

def write_jsonl_data(filepath: str, data: List[Dict]):
    with open(filepath, mode="w", encoding="utf-8") as f:
        # Write out all provided lines
        for jeh_son in data:
            try:
                f.write(json.dumps(dict(jeh_son), separators=(',', ':')))
            except Exception as err:
                printerr(f"error in writing jsonl file, recieved: {err} for data:\n")
                print(jeh_son)
            f.write("\n") #newlines must be verbosely written in JSONL
        f.close()

