"""
Created by Omar Cuenca (JOmarCuenca)
Date : 03/09/2021
"""

from datetime import date
import argparse

MULTILINE_QUOTES = "\"\"\"\n"


def getToday() -> str:
    today = date.today()
    return today.strftime("%d/%m/%Y")

class ProgramArgs:
    def __init__(self,pyFile : str, created : bool) -> None:
        self.pyFile     = pyFile
        self.created    = created
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adds credits to all our .py files')
    parser.add_argument('pyFile', help='path to the .py File to edit')
    parser.add_argument("--created", dest="disclaimer", nargs="?", const=True, default=False, help="Marks the files as being created instead of modified")
    rawArgs = parser.parse_args()
    args = ProgramArgs(rawArgs.pyFile, rawArgs.disclaimer)

    linesToAdd = [
        MULTILINE_QUOTES
    ]

    if(args.created):
        linesToAdd.append("Created/Modified by:\n")
    else:
        linesToAdd.append("Modified by:\n")
    
    with open("utils/credits.txt","r") as f:
        creditLines = f.readlines()
        linesToAdd.extend([f"- {creditLine}" for creditLine in creditLines])
        f.close()

    linesToAdd.extend([
        f"\nDate: {getToday()}\n",
        MULTILINE_QUOTES,
        '\n'
    ])

    with open(args.pyFile, "r") as f:
        linesToAdd.extend(f.readlines())
        f.close()

    with open(args.pyFile, "w") as f:
        f.writelines(linesToAdd)
        f.close()

    print(f"Done modifying file {args.pyFile}")




