import os
import sys
from os import path

from lexer import lex
from parser import parse_module, TokenSource, print_parser_error
from validator import validate_module
from codegen import codegen_module


def compile(filepath : str) -> str | None:
    with open(filepath, 'r') as f:
        text = f.read()

    (tokens, error) = lex(text)
    if error:
        print(error)
        return None

    token_source = TokenSource(tokens, 0, text)
    (parsed_module, error) = parse_module(token_source)

    if error:
        print_parser_error(error, text)
        return None

    validated_module, error = validate_module(parsed_module)
    if error:
        print(error)
        return None

    return codegen_module(validated_module)


def make_absolute_file_path(working_dir : str, relative_or_absolute_file_path : str) -> str:
    if path.isabs(relative_or_absolute_file_path):
        return relative_or_absolute_file_path
    return  path.join(working_dir, relative_or_absolute_file_path)


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('usage: compile <input_file_path> <output_file_path>')
        exit(-1)

    wd = os.getcwd()
    in_filepath = make_absolute_file_path(wd, sys.argv[1])
    out_filepath = make_absolute_file_path(wd, sys.argv[2])

    generated = compile(in_filepath)

    if generated:
        with open(out_filepath, 'w') as file:
            file.write(generated)




