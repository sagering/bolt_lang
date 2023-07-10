from typing import Optional, Union
from dataclasses import dataclass
from enum import Enum

def is_digit(c) -> bool:
    return c >= '0' and c <= '9'


def is_ignored_whitespace(c) -> bool:
    return c in ' \f\r\t\v '


def is_alphabetic(c) -> bool:
    return c in 'abcdefghikjlmopqrstuvwxyzABCDEFGHIKJLMNOPQRSTUVWXYZ'


def is_name_character(c) -> bool:
    return is_name_start_character(c) or c in '0123456789'


def is_name_start_character(c) -> bool:
    return c in 'abcdefghikjlmnopqrstuvwxyzABCDEFGHIKJLMNOPQRSTUVWXYZ_'


class TokenKind(Enum):
    # (
    LeftParen = 1
    # )
    RightParen = 2
    Plus = 3
    Minus = 4
    Asterisk = 5
    ForwardSlash = 6
    Percent = 7
    Number = 8
    Name = 9
    Colon = 10
    Comma = 11
    GreaterThan = 12
    Newline = 13
    Indent = 14
    Dedent = 15
    Equals = 16
    EOF = 17
    # [
    LeftBracket = 18
    # ]
    RightBracket = 19
    LeftCurly = 20
    RightCurly = 21
    Ampersand = 22
    Dot = 23
    String = 23
    LessThan = 24


@dataclass
class Token:
    kind : TokenKind
    data : Optional[Union[float, str]] = None


@dataclass
class TextIndex:
    text : str
    index : int


@dataclass
class Span:
    start : int
    end : int


@dataclass
class TokenSpan:
    token : Token
    span : Span

    def __iter__(self):
        return iter((self.token, self.span))


def lex(input : str) -> tuple[list[TokenSpan] | None, str | None]:
    tokens : list[TokenSpan] = []
    tindex = TextIndex(input, 0)

    line_blank = True
    line_start = 0
    indent_stack = []

    # Append EOF
    input += chr(26)

    while tindex.index < len(tindex.text):
        char = tindex.text[tindex.index]

        # EOF
        if char == chr(26):
            tokens.append(TokenSpan(Token(TokenKind.EOF), Span(tindex.index, tindex.index + 1)))
            break

        if is_ignored_whitespace(char):
            tindex.index += 1
        elif char == '\n':
            if not line_blank:
                tokens.append(TokenSpan(Token(TokenKind.Newline), Span(tindex.index, tindex.index + 1)))
            tindex.index += 1
            line_start = tindex.index
            line_blank = True
        else:
            if line_blank:
                col = tindex.index - line_start

                if col % 4 != 0:
                    return None, 'indentation error'

                if (len(indent_stack) == 0 and col > 0) or (len(indent_stack) > 0 and indent_stack[-1] < col):
                    tokens.append(TokenSpan(Token(TokenKind.Indent), Span(tindex.index, tindex.index + 1)))
                    indent_stack.append(col)
                else:
                    while len(indent_stack) > 0 and indent_stack[-1] > col:
                        tokens.append(TokenSpan(Token(TokenKind.Dedent), Span(tindex.index, tindex.index + 1)))
                        indent_stack.pop()

                line_blank = False

            if char == '(':
                tokens.append(TokenSpan(Token(TokenKind.LeftParen), Span(tindex.index, tindex.index + 1)))
                tindex.index += 1
            elif char == ')':
                tokens.append(TokenSpan(Token(TokenKind.RightParen), Span(tindex.index, tindex.index + 1)))
                tindex.index += 1
            elif char == '*':
                tokens.append(TokenSpan(Token(TokenKind.Asterisk), Span(tindex.index, tindex.index + 1)))
                tindex.index += 1
            elif char == '%':
                tokens.append(TokenSpan(Token(TokenKind.Percent), Span(tindex.index, tindex.index + 1)))
                tindex.index += 1
            elif char == '/':
                tokens.append(TokenSpan(Token(TokenKind.ForwardSlash), Span(tindex.index, tindex.index + 1)))
                tindex.index += 1
            elif char == '+':
                tokens.append(TokenSpan(Token(TokenKind.Plus), Span(tindex.index, tindex.index + 1)))
                tindex.index += 1
            elif char == '-':
                tokens.append(TokenSpan(Token(TokenKind.Minus), Span(tindex.index, tindex.index + 1)))
                tindex.index += 1
            elif char == ':':
                tokens.append(TokenSpan(Token(TokenKind.Colon), Span(tindex.index, tindex.index + 1)))
                tindex.index += 1
            elif char == ',':
                tokens.append(TokenSpan(Token(TokenKind.Comma), Span(tindex.index, tindex.index + 1)))
                tindex.index += 1
            elif char == '>':
                tokens.append(TokenSpan(Token(TokenKind.GreaterThan), Span(tindex.index, tindex.index + 1)))
                tindex.index += 1
            elif char == '<':
                tokens.append(TokenSpan(Token(TokenKind.LessThan), Span(tindex.index, tindex.index + 1)))
                tindex.index += 1
            elif char == '=':
                tokens.append(TokenSpan(Token(TokenKind.Equals), Span(tindex.index, tindex.index + 1)))
                tindex.index += 1
            elif char == '[':
                tokens.append(TokenSpan(Token(TokenKind.LeftBracket), Span(tindex.index, tindex.index + 1)))
                tindex.index += 1
            elif char == ']':
                tokens.append(TokenSpan(Token(TokenKind.RightBracket), Span(tindex.index, tindex.index + 1)))
                tindex.index += 1
            elif char == '{':
                tokens.append(TokenSpan(Token(TokenKind.LeftCurly), Span(tindex.index, tindex.index + 1)))
                tindex.index += 1
            elif char == '}':
                tokens.append(TokenSpan(Token(TokenKind.RightCurly), Span(tindex.index, tindex.index + 1)))
                tindex.index += 1
            elif char == '&':
                tokens.append(TokenSpan(Token(TokenKind.Ampersand), Span(tindex.index, tindex.index + 1)))
                tindex.index += 1
            elif char == '"':
                token_span, error = lex_string(tindex)
                if error: return None, error
                tokens.append(token_span)
            elif char in '.':
                if tindex.text[tindex.index + 1].isdigit():
                    token_span, error = lex_number(tindex)
                    if error: return None, error
                    tokens.append(token_span)
                else:
                    tokens.append(TokenSpan(Token(TokenKind.Dot), Span(tindex.index, tindex.index + 1)))
                    tindex.index += 1
            elif char.isdigit():
                token_span, error = lex_number(tindex)
                if error: return None, error
                tokens.append(token_span)
            elif is_name_start_character(char):
                token_span, error = lex_name(tindex)
                if error: return None, error
                tokens.append(token_span)
            else:
                raise NotImplementedError(f"Unhandled character {char}")

    while len(indent_stack) > 0:
        tokens.append(TokenSpan(Token(TokenKind.Dedent), Span(tindex.index, tindex.index + 1)))
        indent_stack.pop()

    tokens.append(TokenSpan(Token(TokenKind.EOF), Span(tindex.index, tindex.index + 1)))

    return tokens, None


def lex_number(tindex : TextIndex) -> (TokenSpan | None, str | None):
    start : int = tindex.index
    point_cnt : int = 0
    digit_cnt : int = 0

    while tindex.index < len(tindex.text) and (is_digit(tindex.text[tindex.index]) or tindex.text[tindex.index] == '.'):
        if tindex.text[tindex.index] == '.':
            point_cnt += 1
        else:
            digit_cnt += 1

        tindex.index += 1

    if point_cnt > 1:
        return None, "unexpected number of points in number"

    if digit_cnt == 0:
        return None, "no digits in number"

    return TokenSpan(Token(TokenKind.Number, tindex.text[start:tindex.index]), Span(start, tindex.index)), None


def lex_name(tindex : TextIndex) -> (TokenSpan | None, str | None):
    start : int = tindex.index

    if tindex.index < len(tindex.text) and not is_name_start_character(tindex.text[tindex.index]):
        return None, f'unexpected start character {tindex.text[tindex.index]} of name token'

    while tindex.index < len(tindex.text) and is_name_character(tindex.text[tindex.index]):
        tindex.index += 1

    if tindex.index == start:
        return None, 'failed to lex name'

    return TokenSpan(Token(TokenKind.Name, tindex.text[start:tindex.index]), Span(start, tindex.index)), None


def lex_string(tindex : TextIndex) -> (TokenSpan | None, str | None):
    start : int = tindex.index

    if tindex.index < len(tindex.text) and not tindex.text[tindex.index] == '"':
        return None, f'unexpected start character {tindex.text[tindex.index]} of string token'

    tindex.index += 1

    while tindex.index < len(tindex.text) and tindex.text[tindex.index] != '"':
        tindex.index += 1

    if tindex.index == len(tindex.text) or tindex.text[tindex.index] != '"':
        return None, 'missing closing " of string literal'

    tindex.index += 1

    return TokenSpan(Token(TokenKind.String, tindex.text[start+1:tindex.index-1]), Span(start, tindex.index)), None

if __name__ == "__main__":
    print(lex("( 1231.1231 ) * ( 0213.1 )\n    \n"))