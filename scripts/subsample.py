import argparse
import json
import os
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', help='Directory containing `dialogues_XXX.json` files', required=True)
    parser.add_argument('-o', '--out', help='Output directory location and name', required=True)
    args = parser.parse_args()

    pattern = re.compile(r"dialogues_[0-9]+\.json")
    for file in os.listdir(args.dir):
        if pattern.match(file):
            with open(os.path.join(args.dir, file), "r") as f:
                data = json.load(f)
            with open(os.path.join(args.out, file), "w") as f:
                json.dump(data[::5], f)


if __name__ == '__main__':
    main()
