import json
from argparse import ArgumentParser
from pathlib import Path
from collections import Counter

def main():
    parser = ArgumentParser()
    parser.add_argument("diarized_path", type=str)

if __name__=="__main__":
    main()