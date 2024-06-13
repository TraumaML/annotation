# author: Bruce Atwood
# -*- coding: utf-8 -*-

import re
import os
from pathlib import Path
from multiprocessing import Pool
import click
from collections import Counter

def star_replace(word: str, first: bool=False, last: bool=False)-> str:
    """
    Helper to expression_to_regex

    Parameters
    ----------
    word : str
        word in expression.
    first : Boolean, optional
        First word in sentence? The default is False.
    last : Boolean, optional
        Last word in sentence? The default is False.

    Returns
    -------
    str
        python regex version of word, with white spaces where necessary
    """
    outer_star = "[a-zA-Z]*"  # exclusive
    inner_star = "[\w]*"  # inclusive
        
    # handle OR
    if "|" in word:     
        tmp = [star_replace(expr, first=first, last=last) for expr in word.split("|")]
        return "("+ "|".join(tmp) + ")"
    
    if first and last:
        if word[-1] == "*":
            word = word[:-1] + outer_star
        else:
            word = word + "\\b"
        if word[0] == "*":
            word = outer_star + word[1:]
        else:
            word = "\\b" + word
        return word


    if first:
        if word[-1] == "*":
            word = word[:-1] + inner_star
        if word[0] == "*":
            word = outer_star + word[1:]
        else:
            word = "\\b" + word
        return word + " "

    if last:
         if word[-1] == "*":
            word = word[:-1] + outer_star
         else:
            word = word + "\\b"
         if word[0] == "*":
            word = inner_star + word[1:]
         return word

    return word.replace("*", inner_star) + " "


def expression_to_regex(expression: str) -> str:
    """
    Converts a regular expression as defined by the CRIS NLP library
    to a python regular expression

    ex inputs:
        blunt* [0-2_words] *affect*
        sleep* or slep* [0-2_words] disorder*

    To denote optional words, subexpression must end with 'words]'
    and define range of words, with no spaces (see example above)

    *** only handles one OR per statement ***

    """

    expr_list= expression.split()

    if "or" in expr_list:
        index = expr_list.index("or")
        post, prior = expr_list.pop(index+1), expr_list.pop(index-1)
        expr_list[index-1] = prior + "|" + post

    for i in range(len(expr_list)):

        # handle multiple optional words
        if expr_list[i][-6:] == "words]":
            nums = sorted(re.findall("\d", expr_list[i]))
            max_num, min_num = int(nums[-1]), int(nums[0])
            expr_list[i] = "\w* " * min_num + "(\w* )?" * (max_num-min_num)

        elif len(expr_list)==1:
            return star_replace(expr_list[i], first=True, last=True)

        elif i == 0:
            expr_list[i] = star_replace(expr_list[i], first=True)

        elif i == len(expr_list)-1:
            expr_list[i] = star_replace(expr_list[i], last=True)

        else:
            expr_list[i] = star_replace(expr_list[i])

    return "".join(expr_list)


def annotate(file_path: Path, regexes: list[tuple[re.Pattern,str]], write=True) -> Counter:
    """
    Opens the .txt file, and creates and writes to a new .ANN file by running list
    of regular expressions on each line in .txt file

    returns a counter so you can see stats about what's being annotated, for those with eyes to see
    """
    anno_file_path =  str(Path(file_path).parent) + os.sep + str(Path(file_path).stem) + ".ann"
    match_count = Counter()
    _sum = 0

    with Path(file_path).open(encoding="utf8") as ehr_file:

        f = ehr_file.read()
        for regex,kind in regexes:
            iterator = regex.finditer(f)
            for match in iterator:
                start = match.span()[0]
                end = match.span()[1]
                if write:
                    with open(anno_file_path, "a", encoding = "utf8") as anno_file:
                        anno_file.write(f'T{_sum}\t{kind} {start} {end}\t{match.group()}\n')
                match_count[kind] += 1
                _sum += 1

        return match_count


def compile_regex_path(regex_path: Path) -> list[re.Pattern]:
    """
    Opens the regex_path, translates the regexes into pythonic regexes, and returns
    a list of compiled regex patterns.

    NOTE: 
        Ignores case
        Multiline
    """
    try:
        with Path(regex_path).open(encoding="utf8") as regex_file:
            regexes = [re.compile(expression_to_regex(expr), re.IGNORECASE|re.MULTILINE) for expr in regex_file]
    except:
        raise FileNotFoundError
    return regexes


@click.command()
@click.argument('dir_path', type=click.Path(exists=True), required=False)
@click.argument('symptom_regex_path', type=click.Path(exists=True), required=False)
@click.argument('event_regex_path', type=click.Path(exists=True), required=False)
@click.argument('substance_regex_path', type=click.Path(exists=True), required=False)
def main(dir_path: Path, symptom_regex_path: Path, event_regex_path: Path, substance_regex_path: Path) -> None:
    """
    dir_path: path to directory full of .txt files to annotate
    regex_file_path: path to .txt file, with 1 regular expression per line
        ex: preanno_symptoms_list.txt
    """

    dir_path = Path(dir_path)
    symptom_regex_path = Path(symptom_regex_path)
    event_regex_path = Path(event_regex_path)
    substance_regex_path = Path(substance_regex_path)

    # ive never heard of itertools.ziplongest so no need to make me feel bad
    symptom_regexes = [(regex,"Symptom") for regex in compile_regex_path(symptom_regex_path)]
    event_regexes = [(regex,"Event") for regex in compile_regex_path(event_regex_path)]
    substance_regexes = [(regex,"Substance") for regex in compile_regex_path(substance_regex_path)]
    regexes = symptom_regexes + event_regexes + substance_regexes

    def work_generator():
        for file_path in Path(dir_path).iterdir():
            if not file_path.name.endswith(".txt"):
                continue

            yield file_path, regexes

    Pool().starmap(annotate, work_generator(), chunksize=100)
    

if __name__ == '__main__':
    main()


