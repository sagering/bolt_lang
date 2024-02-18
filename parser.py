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

    def skip(self, tokenkind: TokenKind):
        while self.remaining() > 0:
            token_span = self.peek()
            if token_span.token.kind != tokenkind:
                break
            self.advance()

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
    Address = 8
    SetUnion = 8

    def precedence(self) -> int:
        match self:
            case Operator.And:
                return 20
            case Operator.Equals:
                return 25
            case Operator.LessThan:
                return 30
            case Operator.SetUnion:
                return 31
            case Operator.Plus:
                return 35
            case Operator.Minus:
                return 40
            case Operator.Multiply:
                return 45
            case Operator.Divide:
                return 50
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
                return '&&'
            case Operator.Address:
                return '&'
            case _:
                raise NotImplementedError()


@dataclass
class ParsedOperator:
    op: Operator
    span: Span


class ComplexOperator(Enum):
    Array = 1
    Slice = 2


@dataclass
class ParsedComplexOperator:
    op: ComplexOperator
    par: Optional['ParsedExpression']
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
    op: ParsedOperator | ParsedComplexOperator
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
class ParsedInitializerExpression:
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


@dataclass
class ParsedField:
    name: ParsedName
    type: 'ParsedExpression'
    span : Span


@dataclass
class ParsedStructExpression:
    name: Optional[ParsedName]
    fields: list[ParsedField]
    span: Span


ParsedAtom = Union[ParsedName, ParsedNumber, ParsedArray, ParsedString]

ParsedPrimaryExpression = Union[ParsedCall, ParsedIndexExpression, ParsedSliceExpression, ParsedDotExpression, ParsedInitializerExpression, ParsedAtom, ParsedStructExpression]

ParsedExpression = Union[ParsedUnaryOperation, ParsedBinaryOperation, ParsedPrimaryExpression]


@dataclass
class ParsedParameter:
    is_comptime : bool
    name: ParsedName
    type: 'ParsedExpression'
    span: Span


@dataclass
class ParsedBlock:
    statements: list['ParsedStatement']
    span: Span


@dataclass
class ParsedVariableDeclaration:
    span: Span
    type: Optional['ParsedExpression']
    is_comptime : bool
    name: str
    initializer: ParsedExpression


@dataclass
class ParsedAssignment:
    to: ParsedName
    value: ParsedExpression
    span: Span


@dataclass
class ParsedWhile:
    condition: ParsedExpression
    block: ParsedBlock
    span: Span
    is_comptime : bool


@dataclass
class ParsedFunctionDefinition:
    name: ParsedName
    is_comptime : bool
    pars: list[ParsedParameter]
    return_type: 'ParsedExpression'
    body: ParsedBlock
    span: Span

    def runtime_par_count(self):
        return len(list(filter(lambda par: not par.is_comptime, self.pars)))

    def comptime_par_count(self):
        return len(list(filter(lambda par: par.is_comptime, self.pars)))

    def any_par_count(self):
        return len(list(filter(lambda par: isinstance(par.type, ParsedName) and par.type.value == 'any', self.pars)))


@dataclass
class ParsedComptimeFunctionDefinition:
    name: ParsedName
    pars: list[ParsedParameter]
    return_type: 'ParsedExpression'
    body: ParsedBlock
    span: Span


@dataclass
class ParsedExternFunctionDeclaration:
    name: ParsedName
    pars: list[ParsedParameter]
    return_type: 'ParsedExpression'
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
    is_comptime : bool


ParsedStatement = Union[
    ParsedReturn, ParsedFunctionDefinition, ParsedVariableDeclaration, ParsedAssignment, ParsedWhile, ParsedExpression, ParsedBreakStatement, ParsedIfStatement]

ParsedDeclDef = Union[ParsedFunctionDefinition, ParsedStructExpression, ParsedVariableDeclaration, ParsedExternFunctionDeclaration]

@dataclass
class ParsedModule:
    body: list[ParsedDeclDef]
    span: Span


def parse_operator(token_source: TokenSource) -> (ParsedOperator | None, ParserError | None):
    (token, span) = token_source.peek()
    kind = token.kind
    start = span.start

    match kind:
        case TokenKind.And:
            token_source.advance()
            return ParsedOperator(Operator.And, Span(start, token_source.idx())), None
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
        case TokenKind.Pipe:
            token_source.advance(1)
            return ParsedOperator(Operator.SetUnion, Span(start, token_source.idx())), None
        case TokenKind.Equals if token_source.match_token(Token(TokenKind.Equals), 1):
            token_source.advance(2)
            return ParsedOperator(Operator.Equals, Span(start, token_source.idx())), None
        case TokenKind.Ampersand:
            token_source.advance()
            return ParsedOperator(Operator.Address, Span(start, token_source.idx())), None
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
            source_start = token_source.index
            parsed_struct_expression, error = parse_initializer_expression(left, token_source)
            if error:
                token_source.index = source_start
                return left, None
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

    if token_source.match_name('struct'):
        parsed_struct_expr, error = parse_struct_expression(token_source)
        if error:
            return None, error
        return parse_primary_expr(parsed_struct_expr, token_source)

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

    def parse_array_or_slice_type_expr(token_source: TokenSource) -> (ParsedExpression | None, ParserError | None):
        token_span, error = token_source.try_consume_token(TokenKind.LeftBracket)
        start = token_span.span.start

        token_span_right_bracket, error = token_source.try_consume_token(TokenKind.RightBracket)

        expr = None
        if error:
            expr, error = parse_expression(token_source)
            if error: return None, error

            token_span_right_bracket, error = token_source.try_consume_token(TokenKind.RightBracket)
            if error: return None, error

        operand, error = parse_operand(token_source)
        if error: return None, error

        operator_span = Span(start, token_span_right_bracket.span.end)
        if expr:
            return ParsedUnaryOperation(operand, ParsedComplexOperator(ComplexOperator.Array, expr, operator_span),
                                        Span(start, token_source.idx())), None

        return ParsedUnaryOperation(operand, ParsedComplexOperator(ComplexOperator.Slice, None, operator_span),
                                    Span(start, token_source.idx())), None

    match kind:
        case TokenKind.LeftParen:
            token_source.advance()

            expr_span, error = parse_expression(token_source)
            if error: return None, error

            token_span, error = token_source.try_consume_token(TokenKind.RightParen)
            if error: return None, error

            return parse_primary_expr(expr_span, token_source)

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

        case TokenKind.LeftBracket:
            start_index = token_source.index
            expr, first_error = parse_array_or_slice_type_expr(token_source)

            if first_error:
                token_source.index = start_index
                expr, second_error = parse_primary_expr(None, token_source)
                if second_error:
                    return None, first_error
                return expr, None

            return expr, None

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


def parse_initializer_expression(parsed_expr: ParsedPrimaryExpression, token_source: TokenSource) -> (ParsedInitializerExpression | None, ParserError | None):
    _, error = token_source.try_consume_token(TokenKind.LeftCurly)
    if error: return None, error

    _, error = token_source.try_consume_token(TokenKind.RightCurly)
    if error: return None, error

    return ParsedInitializerExpression(parsed_expr, Span(parsed_expr.span.start, token_source.idx())), None


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


def parse_variable_declaration(token_source: TokenSource) -> (ParsedVariableDeclaration | None, ParserError | None):
    # @
    token_span, error = token_source.try_consume_token(TokenKind.At)
    is_comptime = not error

    token_span, error = token_source.try_consume_token(TokenKind.Name)
    if error: return None, error

    start = token_span.span.start

    name = token_span.token.data

    _, error = token_source.try_consume_token(TokenKind.Colon)
    if error: return None, error

    parsed_type_expr, _ = parse_expression(token_source)

    _, error = token_source.try_consume_token(TokenKind.Equals)
    if error: return None, error

    expression, error = parse_expression(token_source)
    if error: return None, error

    _, error = token_source.try_consume_token(TokenKind.Newline)
    if error: return None, error

    return ParsedVariableDeclaration(Span(start, token_source.idx()), parsed_type_expr, is_comptime, name, expression), None


def parse_while_stmt(token_source: TokenSource) -> (ParsedWhile | None, ParserError | None):
    # @
    token_span, error = token_source.try_consume_token(TokenKind.At)
    start = token_span.span.start if token_span else None

    is_comptime = error is None

    # while
    token_span, error = token_source.try_consume_name("while")
    if error: return None, error

    start = token_span.span.start if start is None else start

    # condition
    expression, error = parse_expression(token_source)
    if error: return None, error

    # block
    block, error = parse_block(token_source)
    if error: return None, error

    return ParsedWhile(expression, block, Span(start, token_source.idx()), is_comptime), None


def parse_assignment_stmt(token_source: TokenSource, parsed_expr : ParsedExpression) -> (ParsedAssignment | None, ParserError | None):
    start = parsed_expr.span.start

    _, error = token_source.try_consume_token(TokenKind.Equals)
    if error: return None, error

    value, error = parse_expression(token_source)
    if error: return None, error

    token_span, error = token_source.try_consume_token(TokenKind.Newline)
    if error: return None, error

    return ParsedAssignment(parsed_expr, value, Span(start, token_source.idx())), None


def parse_break_stmt(token_source: TokenSource) -> tuple[Optional[ParsedBreakStatement], Optional[ParserError]]:
    token_span, error = token_source.try_consume_name("break")
    if error: return None, error

    start = token_span.span.start

    _, error = token_source.try_consume_token(TokenKind.Newline)
    if error: return None, error

    return ParsedBreakStatement(Span(start, token_source.idx())), None


def parse_if_stmt(token_source: TokenSource) -> tuple[Optional[ParsedIfStatement], Optional[ParserError]]:

    token_span, error = token_source.try_consume_token(TokenKind.At)
    start = token_span.span.start if token_span else None

    is_comptime = error is None

    token_span, error = token_source.try_consume_name("if")
    if error: return None, error

    start = token_span.span.start if start is None else start

    expression, error = parse_expression(token_source)
    if error: return None, error

    block, error = parse_block(token_source)
    if error: return None, error

    return ParsedIfStatement(expression, block, Span(start, token_source.idx()), is_comptime), None


def parse_struct_field(token_source : TokenSource) -> (ParsedField | None, ParserError | None):

    token_span, error = token_source.try_consume_token(TokenKind.Name)
    if error: return None, error

    start = token_span.span.start
    name = ParsedName(token_span.token.data, token_span.span)

    token_span, error = token_source.try_consume_token(TokenKind.Colon)
    if error: return None, error

    parsed_type_expr, error = parse_expression(token_source)
    if error: return None, error

    token_span, error = token_source.try_consume_token(TokenKind.Newline)
    if error: return None, error

    return ParsedField(name, parsed_type_expr, Span(start, token_source.idx())), None


def parse_struct_expression(token_source: TokenSource) -> (ParsedStructExpression | None, ParserError | None):
    token_span, error = token_source.try_consume_name("struct")
    if error: return None, error

    start = token_span.span.start

    token_span, error = token_source.try_consume_token(TokenKind.Name)

    if token_span:
        name = ParsedName(token_span.token.data, token_span.span)
    else:
        name = None

    token_span, error = token_source.try_consume_token(TokenKind.LeftCurly)
    if error: return None, error

    token_source.skip(TokenKind.Newline)

    fields: list[ParsedField] = []

    while not token_source.eof():
        if token_source.match_name():
            parsed_field, error = parse_struct_field(token_source)
            if error: return None, error
            fields.append(parsed_field)
        else:
            break

    if len(fields) == 0:
        return None, ParserError(f'missing struct fields', Span(token_source.idx(), token_source.idx() + 1))

    token_span, error = token_source.try_consume_token(TokenKind.RightCurly)
    if error: return None, error

    return ParsedStructExpression(name, fields, Span(start, token_source.idx())), None


def parse_block(token_source: TokenSource) -> (ParsedBlock | None, ParserError | None):
    token_span, error = token_source.try_consume_token(TokenKind.LeftCurly)
    if error: return None, error

    start = token_span.span.start

    token_source.skip(TokenKind.Newline)

    stmts: list[ParsedStatement] = []

    while not token_source.eof():
        if token_source.match_token(Token(TokenKind.RightCurly)):
            token_source.advance()
            token_source.skip(TokenKind.Newline)
            break
        elif token_source.remaining() >= 2 and token_source.peek().token.kind == TokenKind.Name and token_source.peek(1).token.kind == TokenKind.Colon:
            stmt, error = parse_variable_declaration(token_source)
            if error: return None, error
            stmts.append(stmt)
        elif token_source.remaining() >= 3 and token_source.peek().token.kind == TokenKind.At and token_source.peek(1).token.kind == TokenKind.Name and token_source.peek(
                2).token.kind == TokenKind.Colon:
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
        elif token_source.remaining() >= 2 and token_source.peek().token.kind == TokenKind.At and token_source.match_name('while', 1):
            stmt, error = parse_while_stmt(token_source)
            if error: return None, error
            stmts.append(stmt)
        elif token_source.match_token(Token(TokenKind.Name, 'break')):
            stmt, error = parse_break_stmt(token_source)
            if error: return None, error
            stmts.append(stmt)
        elif token_source.match_token(Token(TokenKind.Name, 'if')):
            stmt, error = parse_if_stmt(token_source)
            if error: return None, error
            stmts.append(stmt)
        elif token_source.remaining() >= 2 and token_source.peek().token.kind == TokenKind.At and token_source.match_name('if', 1):
            stmt, error = parse_if_stmt(token_source)
            if error: return None, error
            stmts.append(stmt)
        else:
            expr, error = parse_expression(token_source)
            if error: return None, error

            if token_source.peek().token.kind == TokenKind.Equals:
                stmt, error = parse_assignment_stmt(token_source, expr)
                if error: return None, error
                stmts.append(stmt)
            else:
                _, error = token_source.try_consume_token(TokenKind.Newline)
                if error: return None, error
                expr.span = Span(expr.span.start, token_source.idx())
                stmts.append(expr)

    if len(stmts) == 0:
        return None, ParserError('Failed to parse block, expect at least one statement', token_source.span())

    return ParsedBlock(stmts, Span(start, token_source.idx())), None


def parse_parameter(token_source: TokenSource) -> (ParsedParameter | None, ParserError | None):
    # @
    token_span, error = token_source.try_consume_token(TokenKind.At)
    is_comptime = not error
    start = token_span.span.start if (token_span is not None) else None

    token_span, error = token_source.try_consume_token(TokenKind.Name)
    if error: return None, error

    start = start if start is not None else token_span.span.start
    par_name = ParsedName(token_span.token.data, token_span.span)

    _, error = token_source.try_consume_token(TokenKind.Colon)
    if error: return None, error

    parsed_type_expr, error = parse_expression(token_source)
    if error: return None, error

    return ParsedParameter(is_comptime, par_name, parsed_type_expr, Span(start, token_source.idx())), None


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
    while token_source.peek().token.kind in [TokenKind.Name, TokenKind.At]:
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
    parsed_type_expr, error = parse_expression(token_source)
    if error: return None, error

    token_span, error = token_source.try_consume_token(TokenKind.Newline)
    if error: return None, error

    return ParsedExternFunctionDeclaration(name, pars, parsed_type_expr, Span(start, token_source.idx())), None


def parse_function_definition(token_source: TokenSource) -> (ParsedFunctionDefinition | None, ParserError | None):
    # @
    function_is_comptime = False

    token_span, error = token_source.try_consume_token(TokenKind.At)
    if not error:
        function_is_comptime = True

    # name
    token_span, error = token_source.try_consume_token(TokenKind.Name)
    if error: return None, error

    name = ParsedName(token_span.token.data, token_span.span)
    start = token_span.span.start

    # (
    token_span, error = token_source.try_consume_token(TokenKind.LeftParen)
    if error: return None, error

    is_comptime = function_is_comptime

    # parameter list
    pars = []
    while token_source.peek().token.kind in [TokenKind.Name, TokenKind.At]:
        par, error = parse_parameter(token_source)
        if error: return None, error

        par.is_comptime |= function_is_comptime
        is_comptime |= par.is_comptime
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
    parsed_type_expr, error = parse_expression(token_source)
    if error: return None, error

    # body
    block, error = parse_block(token_source)
    if error: return None, error

    return ParsedFunctionDefinition(name, function_is_comptime, pars, parsed_type_expr, block, Span(start, token_source.idx())), None


def parse_module(token_source: TokenSource) -> (ParsedModule | None, ParserError | None):
    body: list[ParsedDeclDef] = []

    while not token_source.eof():
        (token, span) = token_source.peek()
        if token.kind == TokenKind.Name and token.data == 'struct':
            struct, error = parse_struct_expression(token_source)
            if error: return None, error
            body.append(struct)
            token_source.skip(TokenKind.Newline)
        elif token_source.remaining() >= 2 and token_source.peek().token.kind == TokenKind.Name and token_source.peek(1).token.kind == TokenKind.Colon:
            variable_decl, error = parse_variable_declaration(token_source)
            if error: return None, error
            body.append(variable_decl)
        elif token_source.remaining() >= 3 and token_source.peek().token.kind == TokenKind.At and token_source.peek(
                1).token.kind == TokenKind.Name and token_source.peek(
                2).token.kind == TokenKind.Colon:
            stmt, error = parse_variable_declaration(token_source)
            if error: return None, error
            body.append(stmt)
        elif token.kind == TokenKind.Name and token.data == 'extern':
            extern_function_declaration, error = parse_extern_function_declaration(token_source)
            if error: return None, error
            body.append(extern_function_declaration)
        elif token.kind == TokenKind.At or token.kind == TokenKind.Name:
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
