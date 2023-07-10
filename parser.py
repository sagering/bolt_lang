import sys
from dataclasses import dataclass
from enum import Enum
from typing import Union, Optional

from lexer import Token, TokenSpan, TokenKind, Span, lex

OptTokenSpan = Optional[TokenSpan]


@dataclass
class ParserError:
    msg: str
    span: Span


@dataclass
class TokenSource:
    tokens: list[TokenSpan]
    index: int
    text: str

    def peek(self, offset = 0) -> TokenSpan:
        return self.tokens[self.index + offset]

    def advance(self, by=1) -> None:
        self.index += by

    def span(self, offset=0) -> Span:
        return self.tokens[self.index + offset].span

    def idx(self):
        return self.tokens[self.index - 1].span.end

    def match_token(self, next_token: Token, offset=0) -> bool:
        if (self.index + offset) >= len(self.tokens):
            return False
        token_span = self.tokens[self.index + offset]
        return token_span.token == next_token

    def match_name(self, name: Optional[str] = None, offset=0) -> bool:
        if (self.index + offset) >= len(self.tokens):
            return False
        token_span = self.tokens[self.index + offset]
        if token_span.token.kind != TokenKind.Name:
            return False
        return name == None or token_span.token.data == name

    def try_consume_token(self, kind: TokenKind) -> tuple[Optional[TokenSpan], Optional[ParserError]]:
        token_span = self.peek()
        if token_span.token.kind != kind:
            return None, ParserError(f'expected token {kind}, got token {token_span.token.kind}', token_span.span)

        self.advance()

        return token_span, None

    def try_consume_name(self, name: str) -> tuple[Optional[TokenSpan], Optional[ParserError]]:
        token_span = self.peek()

        if token_span.token.kind != TokenKind.Name:
            return None, ParserError(f'expected name "{name}", got token {token_span.token.kind}', token_span.span)
        if token_span.token.data != name:
            return None, ParserError(f'expected name "{name}", got name {token_span.token.data}', token_span.span)

        self.advance()

        return token_span, None

    def eof(self) -> bool:
        return self.peek().token.kind == TokenKind.EOF

    def remaining(self):
        return len(self.tokens) - self.index


class Operator(Enum):
    Plus = 1
    Minus = 2
    Multiply = 3
    Divide = 4
    Equals = 5
    And = 6
    LessThan = 7

    def precedence(self) -> int:
        match self:
            case Operator.LessThan:
                return 4
            case Operator.Equals:
                return 5
            case Operator.Plus:
                return 10
            case Operator.Minus:
                return 20
            case Operator.Multiply:
                return 30
            case Operator.Divide:
                return 40
            case _:
                raise NotImplementedError()

    def literal(self) -> str:
        match self:
            case Operator.Plus:
                return '+'
            case Operator.Minus:
                return '-'
            case Operator.Multiply:
                return '*'
            case Operator.Divide:
                return '/'
            case Operator.LessThan:
                return '<'
            case Operator.Equals:
                return '=='
            case Operator.And:
                return '&'
            case _:
                raise NotImplementedError()


@dataclass
class ParsedOperator:
    op: Operator
    span: Span


@dataclass
class ParsedNumber:
    value: str
    span: Span


@dataclass
class ParsedName:
    # TODO: consider renaming value to name?
    value: str
    span: Span


@dataclass
class ParsedUnaryOperation:
    rhs: 'ParsedExpression'
    op: ParsedOperator
    span: Span


@dataclass
class ParsedBinaryOperation:
    lhs: 'ParsedExpression'
    rhs: 'ParsedExpression'
    op: ParsedOperator
    span: Span


@dataclass
class ParsedCall:
    expr: 'ParsedPrimaryExpression'
    args: list['ParsedExpression']
    span: Span


@dataclass
class ParsedIndexExpression:
    expr: 'ParsedPrimaryExpression'
    index: 'ParsedExpression'
    span: Span


@dataclass
class ParsedSliceExpression:
    expr: 'ParsedPrimaryExpression'
    start: 'ParsedExpression'
    end: 'ParsedExpression'
    span: Span


@dataclass
class ParsedDotExpression:
    expr: 'ParsedPrimaryExpression'
    name: ParsedName
    span: Span


@dataclass
class ParsedStructExpression:
    expr : 'ParsedExpression'
    span : Span


@dataclass
class ParsedArray:
    exprs : list['ParsedExpression']
    span : Span


@dataclass
class ParsedString:
    value : str
    span : Span


ParsedAtom = Union[ParsedName, ParsedNumber, ParsedArray, ParsedString]

ParsedPrimaryExpression = Union[ParsedCall, ParsedIndexExpression, ParsedSliceExpression, ParsedDotExpression, ParsedStructExpression, ParsedAtom]

ParsedExpression = Union[ParsedUnaryOperation, ParsedBinaryOperation, ParsedPrimaryExpression]


@dataclass
class ParsedParameter:
    name: ParsedName
    type: 'ParsedType'
    span: Span


@dataclass
class ParsedBlock:
    statements: list['ParsedStatement']
    span: Span


@dataclass
class ParsedVariableDeclaration:
    name: str
    type: 'ParsedType'
    initializer: ParsedExpression
    span: Span


@dataclass
class ParsedAssignment:
    name: ParsedName
    value: ParsedExpression
    span: Span


@dataclass
class ParsedWhile:
    condition: ParsedExpression
    block: ParsedBlock
    span: Span


@dataclass
class ParsedFunctionDefinition:
    name: ParsedName
    pars: list[ParsedParameter]
    return_type: 'ParsedType'
    body: ParsedBlock
    span: Span


@dataclass
class ParsedExternFunctionDeclaration:
    name: ParsedName
    pars: list[ParsedParameter]
    return_type: 'ParsedType'
    span: Span


@dataclass
class ParsedField:
    name: ParsedName
    type: 'ParsedType'
    span : Span


@dataclass
class ParsedStruct:
    name: ParsedName
    fields: list[ParsedField]
    structs: list['ParsedStruct']
    span: Span


@dataclass
class ParsedReturn:
    expression: ParsedExpression
    span: Span


@dataclass
class ParsedBreakStatement:
    span: Span


@dataclass
class ParsedIfStatement:
    condition: ParsedExpression
    body: ParsedBlock
    span: Span


ParsedStatement = Union[
    ParsedReturn, ParsedFunctionDefinition, ParsedVariableDeclaration, ParsedAssignment, ParsedWhile, ParsedExpression, ParsedBreakStatement, ParsedIfStatement, ParsedStruct]

ParsedDeclDef = Union[ParsedFunctionDefinition, ParsedStruct, ParsedVariableDeclaration, ParsedExternFunctionDeclaration]

@dataclass
class ParsedModule:
    body: list[ParsedDeclDef]
    span: Span


def parse_operator(token_source: TokenSource) -> (ParsedOperator | None, ParserError | None):
    (token, span) = token_source.peek()
    kind = token.kind
    start = span.start

    match kind:
        case TokenKind.Plus:
            token_source.advance()
            return ParsedOperator(Operator.Plus, Span(start, token_source.idx())), None
        case TokenKind.Minus:
            token_source.advance()
            return ParsedOperator(Operator.Minus, Span(start, token_source.idx())), None
        case TokenKind.Asterisk:
            token_source.advance()
            return ParsedOperator(Operator.Multiply, Span(start, token_source.idx())), None
        case TokenKind.ForwardSlash:
            token_source.advance()
            return ParsedOperator(Operator.Divide, Span(start, token_source.idx())), None
        case TokenKind.LessThan:
            token_source.advance(1)
            return ParsedOperator(Operator.LessThan, Span(start, token_source.idx())), None
        case TokenKind.Equals if token_source.match_token(Token(TokenKind.Equals), 1):
            token_source.advance(2)
            return ParsedOperator(Operator.Equals, Span(start, token_source.idx())), None
        case TokenKind.Ampersand:
            token_source.advance()
            return ParsedOperator(Operator.And, Span(start, token_source.idx())), None
        case _:
            return None, ParserError(f"Failed to parse operator, unmatched token {kind}", span)


def parse_primary_expr(left : ParsedPrimaryExpression | None, token_source : TokenSource) -> (ParsedPrimaryExpression | None, ParserError | None):
    if left:
        if token_source.match_token(Token(TokenKind.Dot)):
            parsed_dot_expr, error = parse_dot_expr(left, token_source)
            if error: return None, error
            return parse_primary_expr(parsed_dot_expr, token_source)
        elif token_source.match_token(Token(TokenKind.LeftParen)):
            parsed_call, error = parse_call(left, token_source)
            if error: return None, error
            return parse_primary_expr(parsed_call, token_source)
        elif token_source.match_token(Token(TokenKind.LeftCurly)):
            parsed_struct_expression, error = parse_struct_expression(left, token_source)
            if error: return None, error
            return parse_primary_expr(parsed_struct_expression, token_source)
        elif token_source.match_token(Token(TokenKind.LeftBracket)):
            start_index = token_source.index
            expr, error = parse_index_expr(left, token_source)
            if error:
                # Reset start index, because parse_index_expr might have already consumed some tokens which are needed
                # in parse_slice_expr.
                token_source.index = start_index
                expr, error = parse_slice_expr(left, token_source)
            if error:
                return None, error
            return parse_primary_expr(expr, token_source)
        else:
            return left, None

    token, span = token_source.peek()
    start = span.start

    match token.kind:
        case TokenKind.Number:
            token_source.advance()
            return ParsedNumber(token.data, Span(start, token_source.idx())), None
        case TokenKind.String:
            token_source.advance()
            return ParsedString(token.data, Span(start, token_source.idx())), None
        case TokenKind.Name:
            parsed_name, error = parse_name(token_source)
            if error: return None, error
            return parse_primary_expr(parsed_name, token_source)
        case TokenKind.LeftBracket:
            return parse_array(token_source)

        case _: return None, ParserError(f"Failed to parse primary expression, got token {token.kind}", span)


def parse_operand(token_source: TokenSource) -> (ParsedExpression | None, ParserError | None):
    (token, span) = token_source.peek()
    start = span.start
    kind = token.kind

    match kind:
        case TokenKind.LeftParen:
            token_source.advance()

            expr_span, error = parse_expression(token_source)
            if error: return None, error

            token_span, error = token_source.try_consume_token(TokenKind.RightParen)
            if error: return None, error

            return expr_span, None

        case TokenKind.Plus:
            op, error = parse_operator(token_source)
            if error: return None, error

            operand, error = parse_operand(token_source)
            if error: return None, error

            return ParsedUnaryOperation(operand, op, Span(start, token_source.idx())), None

        case TokenKind.Minus:
            op, error = parse_operator(token_source)
            if error: return None, error

            operand, error = parse_operand(token_source)
            if error: return None, error

            return ParsedUnaryOperation(operand, op, Span(start, token_source.idx())), None

        case TokenKind.Ampersand:
            op, error = parse_operator(token_source)
            if error: return None, error

            operand, error = parse_operand(token_source)
            if error: return None, error

            return ParsedUnaryOperation(operand, op, Span(start, token_source.idx())), None

        case TokenKind.Asterisk:
            op, error = parse_operator(token_source)
            if error: return None, error

            operand, error = parse_operand(token_source)
            if error: return None, error

            return ParsedUnaryOperation(operand, op, Span(start, token_source.idx())), None

        case _:
            return parse_primary_expr(None, token_source)


def parse_call(left : ParsedPrimaryExpression, token_source: TokenSource) -> (ParsedCall | None, ParserError | None):
    _, error = token_source.try_consume_token(TokenKind.LeftParen)
    if error: return None, error

    _, error = token_source.try_consume_token(TokenKind.RightParen)

    args = []
    if error:
        while not token_source.eof():
            expr, error = parse_expression(token_source)
            if error: return None, error

            args.append(expr)

            _, error = token_source.try_consume_token(TokenKind.Comma)
            if error: break

        _, error = token_source.try_consume_token(TokenKind.RightParen)
        if error: return None, error

    return ParsedCall(left, args, Span(left.span.start, token_source.idx())), None


def parse_index_expr(left : ParsedPrimaryExpression, token_source: TokenSource) -> (ParsedIndexExpression | None, ParserError | None):
    _, error = token_source.try_consume_token(TokenKind.LeftBracket)
    if error: return None, error

    expr, error = parse_expression(token_source)
    if error: return None, error

    _, error = token_source.try_consume_token(TokenKind.RightBracket)
    if error: return None, error

    return ParsedIndexExpression(left, expr, Span(left.span.start, token_source.idx())), None


def parse_slice_expr(left : ParsedPrimaryExpression, token_source: TokenSource) -> (ParsedSliceExpression | None, ParserError | None):
    _, error = token_source.try_consume_token(TokenKind.LeftBracket)
    if error: return None, error

    start_expr, _ = parse_expression(token_source)

    _, error = token_source.try_consume_token(TokenKind.Colon)
    if error: return None, error

    end_expr, _ = parse_expression(token_source)

    _, error = token_source.try_consume_token(TokenKind.RightBracket)
    if error: return None, error

    return ParsedSliceExpression(left, start_expr, end_expr, Span(left.span.start, token_source.idx())), None


def parse_dot_expr(left : ParsedPrimaryExpression, token_source: TokenSource) -> (ParsedDotExpression | None, ParserError | None):
    _, error = token_source.try_consume_token(TokenKind.Dot)
    if error: return None, error

    parsed_name, error = parse_name(token_source)
    if error: return None, error

    return ParsedDotExpression(left, parsed_name, Span(left.span.start, token_source.idx())), None


def parse_name(token_source : TokenSource) -> (ParsedName | None, ParserError | None):
    token_span, error = token_source.try_consume_token(TokenKind.Name)
    if error: return None, error
    return ParsedName(token_span.token.data, token_span.span), None


def parse_array(token_source: TokenSource) -> (ParsedArray | None, ParserError | None):
    _, error = token_source.try_consume_token(TokenKind.LeftBracket)
    if error: return None, error

    exprs = []
    start = token_source.idx()

    while True:
        expr, error = parse_expression(token_source)
        if error: return None, error
        exprs.append(expr)

        token_span, error = token_source.try_consume_token(TokenKind.Comma)
        if error: break

    _, error = token_source.try_consume_token(TokenKind.RightBracket)
    if error: return None, error

    return ParsedArray(exprs, Span(start, token_source.idx())), None


def parse_struct_expression(parsed_expr: ParsedPrimaryExpression, token_source: TokenSource) -> (ParsedStructExpression | None, ParserError | None):
    _, error = token_source.try_consume_token(TokenKind.LeftCurly)
    if error: return None, error

    _, error = token_source.try_consume_token(TokenKind.RightCurly)
    if error: return None, error

    return ParsedStructExpression(parsed_expr, Span(parsed_expr.span.start, token_source.idx())), None


def parse_expression(token_source: TokenSource) -> (ParsedExpression | None, ParserError | None):
    stack: list[Union[ParsedExpression, ParsedOperator]] = []
    last_precedence: int = sys.maxsize

    def consolidate_stack(precedence: int) -> None:
        nonlocal stack
        nonlocal last_precedence

        while precedence <= last_precedence and len(stack) > 2:
            rhs: ParsedExpression = stack.pop()
            op: ParsedOperator = stack.pop()
            lhs: ParsedExpression = stack.pop()

            binary_operation = ParsedBinaryOperation(lhs, rhs, op, Span(rhs.span.start, lhs.span.end))

            if len(stack) > 0:
                parsed_op: ParsedOperator = stack[-1]
                last_precedence = parsed_op.op.precedence()
            else:
                last_precedence = sys.maxsize

            stack.append(binary_operation)

    while not token_source.eof():

        operand, error = parse_operand(token_source)
        if error:
            if len(stack) == 0:
                return None, error
            else:
                break

        stack.append(operand)

        parsed_op, error = parse_operator(token_source)
        if error: break

        precedence = parsed_op.op.precedence()

        consolidate_stack(precedence)

        stack.append(parsed_op)
        last_precedence = precedence

    if len(stack) % 2 == 1:
        consolidate_stack(-1)

    if len(stack) != 1:
        return None, ParserError(f"Failed to parse expression: stack {stack}", token_source.span())

    return stack.pop(), None


def print_expression(expr: ParsedExpression, depth: int) -> None:
    indent: str = " " * 4 * depth

    match expr:
        case ParsedNumber(value, _):
            print(indent + str(value))
        case ParsedUnaryOperation(rhs, op, _):
            print(indent + "UnaryOperation")
            print(indent + " " * 4 + op.literal())
            print_expression(rhs.expression, depth + 1)
        case ParsedBinaryOperation(lhs, rhs, op, _):
            print(indent + "BinaryOperation")
            print_expression(lhs.expression, depth + 1)
            print(indent + " " * 4 + op.literal())
            print_expression(rhs.expression, depth + 1)
        case _:
            raise NotImplementedError(f"Unmachted expression {expr}")


def parse_return_stmt(token_source: TokenSource) -> (ParsedReturn | None, ParserError | None):
    token_span, error = token_source.try_consume_name("return")
    if error: return None, error

    start: int = token_span.span.start

    parsed_expr, error = parse_expression(token_source)
    if error: return None, error

    _, error = token_source.try_consume_token(TokenKind.Newline)
    if error: return None, error

    return ParsedReturn(parsed_expr, Span(start, token_source.idx())), None


@dataclass
class ParsedArrayType:
    parsed_type : 'ParsedType'
    length : ParsedExpression
    span : Span


@dataclass
class ParsedPointerType:
    parsed_type : 'ParsedType'
    span : Span


@dataclass
class ParsedSliceType:
    parsed_type : 'ParsedType'
    span : Span


ParsedTypeLiteral = Union[ParsedArrayType, ParsedPointerType, ParsedSliceType]
ParsedType = Union[ParsedTypeLiteral, ParsedExpression]


def parse_pointer_type(token_source : TokenSource) -> (ParsedPointerType | None, ParserError | None):
    token_span, error = token_source.try_consume_token(TokenKind.Asterisk)
    if error: return None, error

    start = token_span.span.start

    parsed_type, error = parse_type(token_source)
    if error: return None, error

    return ParsedPointerType(parsed_type, Span(start, token_source.idx())), None


# array : [3]i32
def parse_array_type(token_source: TokenSource) -> (ParsedArrayType | None, ParserError | None):
    token_span, error = token_source.try_consume_token(TokenKind.LeftBracket)
    if error: return None, error

    start = token_span.span.start

    length_expr, error = parse_expression(token_source)
    if error: return None, error

    token_span, error = token_source.try_consume_token(TokenKind.RightBracket)
    if error: return None, error

    parsed_type, error = parse_type(token_source)
    if error: return None, error

    return ParsedArrayType(parsed_type, length_expr, Span(start, token_source.idx())), None


# slice : []i32
def parse_slice_type(token_source : TokenSource) -> (ParsedSliceType | None, ParserError | None):
    token_span, error = token_source.try_consume_token(TokenKind.LeftBracket)
    if error: return None, error

    start = token_span.span.start

    token_span, error = token_source.try_consume_token(TokenKind.RightBracket)
    if error: return None, error

    parsed_type, error = parse_type(token_source)
    if error: return None, error

    return ParsedSliceType(parsed_type, Span(start, token_source.idx())), None


def parse_type_literal(token_source : TokenSource) -> (ParsedTypeLiteral | None, ParserError | None):
    kind = token_source.peek().token.kind

    match kind:
        case TokenKind.Asterisk:
            return parse_pointer_type(token_source)
        case TokenKind.LeftBracket:
            if token_source.peek(1).token.kind == TokenKind.RightBracket:
                return parse_slice_type(token_source)
            else:
                return parse_array_type(token_source)
        case _:
            return None, ParserError('expect "*" or "[" in type literal', token_source.span())


def parse_type(token_source : TokenSource) -> (ParsedType | None, ParserError | None):
    match token_source.peek().token.kind:
        case TokenKind.Name:
            return parse_expression(token_source)
        case _:
            return parse_type_literal(token_source)


def parse_variable_declaration(token_source: TokenSource) -> (ParsedVariableDeclaration | None, ParserError | None):
    token_span, error = token_source.try_consume_name("let")
    if error: return None, error

    start: int = token_span.span.start

    token_span, error = token_source.try_consume_token(TokenKind.Name)
    if error: return None, error

    name = token_span.token.data

    _, error = token_source.try_consume_token(TokenKind.Colon)
    if error: return None, error

    parsed_type, error = parse_type(token_source)
    if error: return None, error

    _, error = token_source.try_consume_token(TokenKind.Equals)
    if error: return None, error

    expression, error = parse_expression(token_source)
    if error: return None, error

    _, error = token_source.try_consume_token(TokenKind.Newline)
    if error: return None, error

    return ParsedVariableDeclaration(name, parsed_type, expression, Span(start, token_source.idx())), None


def parse_while_stmt(token_source: TokenSource) -> (ParsedWhile | None, ParserError | None):
    token_span, error = token_source.try_consume_name("while")
    if error: return None, error

    start = token_span.span.start

    expression, error = parse_expression(token_source)
    if error: return None, error

    block, error = parse_block(token_source)
    if error: return None, error

    return ParsedWhile(expression, block, Span(start, token_source.idx())), None


def parse_assignment_stmt(token_source: TokenSource) -> (ParsedAssignment | None, ParserError | None):
    start = token_source.span().start

    name, error = parse_name(token_source)
    if error: return None, error

    _, error = token_source.try_consume_token(TokenKind.Equals)
    if error: return None, error

    expression, error = parse_expression(token_source)
    if error: return None, error

    token_span, error = token_source.try_consume_token(TokenKind.Newline)
    if error: return None, error

    return ParsedAssignment(name, expression, Span(start, token_source.idx())), None


def parse_break_stmt(token_source: TokenSource) -> tuple[Optional[ParsedBreakStatement], Optional[ParserError]]:
    token_span, error = token_source.try_consume_name("break")
    if error: return None, error

    start = token_span.span.start

    _, error = token_source.try_consume_token(TokenKind.Newline)
    if error: return None, error

    return ParsedBreakStatement(Span(start, token_source.idx())), None


def parse_if_stmt(token_source: TokenSource) -> tuple[Optional[ParsedIfStatement], Optional[ParserError]]:
    token_span, error = token_source.try_consume_name("if")
    if error: return None, error

    start = token_span.span.start

    expression, error = parse_expression(token_source)
    if error: return None, error

    block, error = parse_block(token_source)
    if error: return None, error

    return ParsedIfStatement(expression, block, Span(start, token_source.idx())), None


def parse_struct_field(token_source : TokenSource) -> (ParsedField | None, ParserError | None):

    token_span, error = token_source.try_consume_token(TokenKind.Name)
    if error: return None, error

    start = token_span.span.start
    name = ParsedName(token_span.token.data, token_span.span)

    token_span, error = token_source.try_consume_token(TokenKind.Colon)
    if error: return None, error

    parsed_type, error = parse_type(token_source)
    if error: return None, error

    token_span, error = token_source.try_consume_token(TokenKind.Newline)
    if error: return None, error

    return ParsedField(name, parsed_type, Span(start, token_source.idx())), None


def parse_struct(token_source: TokenSource) -> (ParsedStruct | None, ParserError | None):
    token_span, error = token_source.try_consume_name("struct")
    if error: return None, error

    start = token_span.span.start

    token_span, error = token_source.try_consume_token(TokenKind.Name)
    if error: return None, error

    name = ParsedName(token_span.token.data, token_span.span)

    token_span, error = token_source.try_consume_token(TokenKind.Newline)
    if error: return None, error

    token_span, error = token_source.try_consume_token(TokenKind.Indent)
    if error: return None, error

    fields: list[ParsedField] = []
    structs: list[ParsedStruct] = []

    while not token_source.eof():

        if token_source.match_name('struct'):
            parsed_struct, error = parse_struct(token_source)
            if error: return None, error
            structs.append(parsed_struct)
        # any other name
        elif token_source.match_name():
            parsed_field, error = parse_struct_field(token_source)
            if error: return None, error
            fields.append(parsed_field)
        else:
            break

    if len(fields) == 0:
        return None, ParserError(f'missing struct fields', Span(token_source.idx(), token_source.idx() + 1))

    token_span, error = token_source.try_consume_token(TokenKind.Dedent)
    if error: return None, error

    return ParsedStruct(name, fields, structs, Span(start, token_source.idx())), None


def parse_block(token_source: TokenSource) -> (ParsedBlock | None, ParserError | None):
    token_span, error = token_source.try_consume_token(TokenKind.Newline)
    if error: return None, error

    start = token_span.span.start

    token_span, error = token_source.try_consume_token(TokenKind.Indent)
    if error: return None, error

    stmts: list[ParsedStatement] = []

    while not token_source.eof():
        if token_source.match_token(Token(TokenKind.Dedent)):
            token_source.advance()
            break
        elif token_source.match_token(Token(TokenKind.Name, 'let')):
            stmt, error = parse_variable_declaration(token_source)
            if error: return None, error
            stmts.append(stmt)
        elif token_source.match_token(Token(TokenKind.Name, 'return')):
            stmt, error = parse_return_stmt(token_source)
            if error: return None, error
            stmts.append(stmt)
        elif token_source.match_token(Token(TokenKind.Name, 'while')):
            stmt, error = parse_while_stmt(token_source)
            if error: return None, error
            stmts.append(stmt)
        elif token_source.match_token(Token(TokenKind.Name, 'break')):
            stmt, error = parse_break_stmt(token_source)
            if error: return None, error
            stmts.append(stmt)
        elif token_source.remaining() >= 2 and token_source.peek().token.kind == TokenKind.Name and token_source.peek(1).token.kind == TokenKind.Equals:
            stmt, error = parse_assignment_stmt(token_source)
            if error: return None, error
            stmts.append(stmt)
        else:
            stmt, error = parse_expression(token_source)
            if error: return None, error
            _, error = token_source.try_consume_token(TokenKind.Newline)
            if error: return None, error
            stmt.span = Span(stmt.span.start, token_source.idx())
            stmts.append(stmt)

    if len(stmts) == 0:
        return None, ParserError('Failed to parse block, expect at least one statement', token_source.span())

    return ParsedBlock(stmts, Span(start, token_source.idx())), None


def parse_parameter(token_source: TokenSource) -> [ParsedParameter, ParserError]:
    (token_span, error) = token_source.try_consume_token(TokenKind.Name)
    if error: return None, error

    start = token_span.span.start
    par_name = ParsedName(token_span.token.data, token_span.span)

    _, error = token_source.try_consume_token(TokenKind.Colon)
    if error: return None, error

    parsed_type, error = parse_type(token_source)
    if error: return None, error

    return ParsedParameter(par_name, parsed_type, Span(start, token_source.idx())), None


def parse_extern_function_declaration(token_source: TokenSource) -> (ParsedExternFunctionDeclaration | None, ParserError | None):
    # extern
    token_span, error = token_source.try_consume_token(TokenKind.Name)
    if error: return None, error

    if token_span.token.data != 'extern':
        return None, ParserError('expected extern keyword', token_source.span())

    start = token_span.span.start

    # name
    token_span, error = token_source.try_consume_token(TokenKind.Name)
    if error: return None, error
    name = ParsedName(token_span.token.data, token_span.span)

    # (
    token_span, error = token_source.try_consume_token(TokenKind.LeftParen)
    if error: return None, error

    # parameter list
    pars = []
    while token_source.peek().token.kind == TokenKind.Name:
        par, error = parse_parameter(token_source)
        if error: return None, error

        pars.append(par)

        _, error = token_source.try_consume_token(TokenKind.Comma)
        if error: break

    # )
    _, error = token_source.try_consume_token(TokenKind.RightParen)
    if error: return None, error

    # :
    _, error = token_source.try_consume_token(TokenKind.Colon)
    if error: return None, error

    # type
    parsed_type, error = parse_type(token_source)
    if error: return None, error

    token_span, error = token_source.try_consume_token(TokenKind.Newline)
    if error: return None, error

    return ParsedExternFunctionDeclaration(name, pars, parsed_type, Span(start, token_source.idx())), None


def parse_function_definition(token_source: TokenSource) -> (ParsedFunctionDefinition | None, ParserError | None):
    # name = def
    # token_span, error = token_source.try_consume_name("def")
    # if error: return None, error

    # name
    token_span, error = token_source.try_consume_token(TokenKind.Name)
    if error: return None, error
    name = ParsedName(token_span.token.data, token_span.span)
    start = token_span.span.start
    is_extern = False

    if name.value == 'extern':
        is_extern = True
        token_span, error = token_source.try_consume_token(TokenKind.Name)
        if error: return None, error
        name = ParsedName(token_span.token.data, token_span.span)

    # (
    token_span, error = token_source.try_consume_token(TokenKind.LeftParen)
    if error: return None, error

    # parameter list
    pars = []
    while token_source.peek().token.kind == TokenKind.Name:
        par, error = parse_parameter(token_source)
        if error: return None, error

        pars.append(par)

        _, error = token_source.try_consume_token(TokenKind.Comma)
        if error: break

    # )
    _, error = token_source.try_consume_token(TokenKind.RightParen)
    if error: return None, error

    # :
    _, error = token_source.try_consume_token(TokenKind.Colon)
    if error: return None, error

    # type
    parsed_type, error = parse_type(token_source)
    if error: return None, error

    # body
    block, error = parse_block(token_source)
    if error: return None, error

    return ParsedFunctionDefinition(name, pars, parsed_type, block, Span(start, token_source.idx())), None


def parse_module(token_source: TokenSource) -> (ParsedModule | None, ParserError | None):
    body: list[ParsedDeclDef] = []

    while not token_source.eof():
        (token, span) = token_source.peek()
        if token.kind == TokenKind.Name and token.data == 'struct':
            struct, error = parse_struct(token_source)
            if error: return None, error
            body.append(struct)
        elif token.kind == TokenKind.Name and token.data == 'let':
            variable_decl, error = parse_variable_declaration(token_source)
            if error: return None, error
            body.append(variable_decl)
        elif token.kind == TokenKind.Name and token.data == 'extern':
            extern_function_declaration, error = parse_extern_function_declaration(token_source)
            if error: return None, error
            body.append(extern_function_declaration)
        elif token.kind == TokenKind.Name:
            function_definition, error = parse_function_definition(token_source)
            if error: return None, error
            body.append(function_definition)
        else:
            return None, ParserError(f'failed to parse module, unexpected token {token.kind}', span)

    return ParsedModule(body, Span(0, token_source.idx())), None


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_parser_error(error: ParserError, text: str):
    def overlaps(a: Span, b: Span) -> bool:
        return a.start < b.end and a.end > b.start

    lines = text.split('\n')

    idx = 0
    line_idx = 0
    found = False

    for line in lines:
        line_span = Span(idx,
                         idx + len(line) + 2)  # +2 because +1 for the '\n' char and +1 because span end is exclusive

        if overlaps(error.span, line_span):
            found = True
            break

        idx = line_span.end - 1
        line_idx += 1

    print(Colors.FAIL + f'Parser error at {line_idx + 1}:{error.span.start - idx + 1}:\n' + Colors.ENDC)

    if found:
        print("...")
        print(lines[line_idx - 1])
        print(lines[line_idx][:error.span.start - idx] + Colors.FAIL + lines[line_idx][
                                                                       error.span.start - idx:error.span.end - idx] + Colors.ENDC +
              lines[line_idx][error.span.end - idx:])
        print(Colors.FAIL + ' ' * (error.span.start - idx) + '^- ' + error.msg + Colors.ENDC)
        print(lines[line_idx + 1])
        print("...")
    else:
        print('\n'.join(lines[-3:]))


def main():
    file_name = 'tests/example.bolt'

    with open(file_name, 'r') as f:
        text = f.read()

    (tokens, error) = lex(text)
    if error:
        print(error)
        return

    for token in tokens:
        print(token)

    token_source = TokenSource(tokens, 0, text)
    module, error = parse_module(token_source)

    if error:
        print_parser_error(error, text)
        return

    print(module)


if __name__ == "__main__":
    main()
