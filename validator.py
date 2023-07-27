import sys
from collections import defaultdict
from typing import Union, Optional, ClassVar, Callable
from lexer import Span
from dataclasses import dataclass, field
from parser import ParsedModule, ParsedFunctionDefinition, ParsedExpression, ParsedName, ParsedBinaryOperation, \
    ParsedUnaryOperation, ParsedBlock, ParsedReturn, Operator, ParsedNumber, ParsedCall, ParsedOperator, \
    ParsedVariableDeclaration, ParsedWhile, ParsedBreakStatement, ParsedIfStatement, ParsedStructExpression, \
    ParsedField, ParsedComplexOperator, ComplexOperator, \
    ParsedInitializerExpression, ParsedDotExpression, ParsedPrimaryExpression, ParsedIndexExpression, \
    ParsedArray, ParsedExternFunctionDeclaration, ParsedString, ParsedSliceExpression, print_parser_error, \
    TokenSource, ParsedAssignment, parse_module

from lexer import lex
import random

builtin_types = ['i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64', 'bool', 'f32', 'f64', 'void', 'type']


@dataclass
class FunctionReferenceValue:
    function_definition : 'ValidatedFunctionDefinition'


Value = Union[int, float, bool, str, 'Struct', 'CompleteType', FunctionReferenceValue]


@dataclass
class Variable:
    name: str
    type: 'CompleteType'
    value: Value = None


@dataclass
class Function:
    name: str
    type: 'CompleteType'


@dataclass
class Pointer:
    pass


@dataclass
class Array:
    length: int


@dataclass
class Slice:
    pass


@dataclass
class FunctionPointer:
    pars: list['CompleteType']
    ret: 'CompleteType'
    is_comptime : bool


@dataclass
class Struct:
    @dataclass
    class StructField:
        name: str
        type: 'CompleteType'

    name: Optional[str]
    fields: list[StructField]
    location: str

    def try_get_field(self, name: str) -> StructField | None:
        for field in self.fields:
            if field.name == name:
                return field
        return None

    def fully_qualified_name(self):
        return self.location + '.' + self.name


@dataclass
class NamedType:
    name: str


@dataclass
class CompleteType:
    """ Complete types still require a scope context to resolve the fundamental type names.
        The complete type holds the Type that can come out of an expression.
        It's currently undecided what a Namespace should become, for now these are equivalent
        to builtin or potentially nested declared types (= fundamental types).
    """

    HoldingTypes = Union[Pointer, Array, Slice, FunctionPointer, NamedType]

    val: HoldingTypes
    next: Optional['CompleteType'] = None

    def is_pointer(self) -> bool:
        return isinstance(self.val, Pointer)

    def is_array(self) -> bool:
        return isinstance(self.val, Array)

    def is_slice(self) -> bool:
        return isinstance(self.val, Slice)

    def is_builtin(self) -> bool:
        return self.is_named_type() and self.named_type().name in builtin_types

    def is_named_type(self) -> bool:
        return isinstance(self.val, NamedType)

    def is_type(self) -> bool:
        return isinstance(self.val, NamedType) and self.named_type().name == 'type'

    def is_type_value(self) -> bool:
        self_is_type_value = isinstance(self.val, NamedType) and self.named_type().name != 'type'
        self_is_array_or_slice_or_pointer = self.is_array() or self.is_slice() or self.is_pointer()
        return self_is_type_value or self_is_array_or_slice_or_pointer and self.next.is_type_value()

    def is_integer(self) -> bool:
        return self.is_named_type() and self.named_type().name in ['i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64']

    def is_floating_point(self) -> bool:
        return self.is_named_type() and self.named_type().name in ['f32', 'f64']

    def is_number(self) -> bool:
        return self.is_integer() or self.is_floating_point()

    def is_bool(self) -> bool:
        return self.is_named_type() and self.named_type().name == 'bool'

    def is_function_ptr(self) -> bool:
        return isinstance(self.val, FunctionPointer)

    def get(self) -> HoldingTypes:
        return self.val

    def array(self) -> Array:
        assert (self.is_array())
        return self.get()

    def function_ptr(self) -> FunctionPointer:
        assert (self.is_function_ptr())
        return self.get()

    def named_type(self) -> NamedType:
        assert (self.is_named_type())
        return self.get()

    def to_string(self) -> str:
        if self.is_pointer():
            return '*' + self.next.to_string()
        elif self.is_array():
            return f'[{self.array().length}]' + self.next.to_string()
        elif self.is_slice():
            return '[]' + self.next.to_string()
        elif self.is_named_type():
            return self.named_type().name
        elif self.is_function_ptr():
            function_ptr = self.function_ptr()
            pars = ','.join([par.to_string() for par in function_ptr.pars])
            return f'@function:({pars})->({function_ptr.ret.to_string()})'
        else:
            raise NotImplementedError(self.get())

    def collect_downstream_types(self, bag: dict[str, 'CompleteType']):
        if self.to_string() in bag:
            return
        elif self.is_pointer():
            bag[self.to_string()] = self
        elif self.is_array():
            bag[self.to_string()] = self
        elif self.is_slice():
            bag[self.to_string()] = self
        elif self.is_named_type():
            bag[self.to_string()] = self
        elif self.is_function_ptr():
            bag[self.to_string()] = self
            for par in self.function_ptr().pars:
                par.collect_downstream_types(bag)
            self.function_ptr().ret.collect_downstream_types(bag)
        else:
            raise NotImplementedError(self.get())

    def eq_or_other_safely_convertible(self, other: 'CompleteType') -> bool:
        if self.is_integer() and other.is_integer():
            return check_integer_type_compatibility(self.named_type().name, other.named_type().name)

        return self.eq(other)

    def eq(self, other: 'CompleteType') -> bool:

        if self.is_named_type() and other.is_named_type():
            return self.named_type().name == other.named_type().name

        if self.is_pointer() and other.is_pointer():
            return self.next.eq(other.next)

        if self.is_array() and other.is_array() and (self.array().length == other.array().length):
            return self.next.eq(other.next)

        if self.is_slice() and other.is_slice():
            return self.next.eq(other.next)

        if self.is_function_ptr() and other.is_function_ptr():
            a = self.function_ptr()
            b = other.function_ptr()

            if len(a.pars) == len(b.pars):
                for (ta, tb) in zip(a.pars, b.pars):
                    if not ta.eq(tb):
                        return False

                return True

            return False

        return False


def is_unary_operator_defined(complete_type: CompleteType, op: Operator) -> (CompleteType | None):
    if not complete_type.is_builtin() or complete_type.named_type() == 'bool':
        return None
    else:
        match op:
            case Operator.Plus:
                return complete_type
            case Operator.Minus:
                if complete_type.named_type().name == 'u64':
                    return None
                if complete_type.named_type().name in ['u8', 'u16', 'u32']:
                    conversion_table = {}
                    conversion_table['u8'] = 'i16'
                    conversion_table['u16'] = 'i32'
                    conversion_table['u32'] = 'i64'
                    return CompleteType(NamedType(conversion_table[complete_type.named_type().name]))

                return complete_type
            case _:
                return None


def is_binary_operator_defined(lhs: CompleteType, rhs: CompleteType, op: Operator) -> (CompleteType | None):
    if not lhs.is_builtin() or lhs.named_type() == 'bool':
        return None
    else:
        match op:
            case Operator.Minus | Operator.Plus | Operator.Minus | Operator.Multiply | Operator.Divide:
                return lhs
            case Operator.Equals | Operator.LessThan:
                return CompleteType(NamedType('bool'))
            case _:
                return None


scope_number: int = 0


@dataclass
class Scope:
    name: str = ''
    children: list['Scope'] = field(default_factory=list)
    parent: 'Scope' = None
    inside_while_block: bool = False
    scope_number: int = 0
    is_comptime = False

    functions: list[Function] = field(default_factory=list)
    comptime_functions: list[FunctionReferenceValue] = field(default_factory=list)
    variables: list[Variable] = field(default_factory=list)
    type_infos: dict[str, Struct] = field(default_factory=dict)
    type_variables: list[Variable] = field(default_factory=list)
    full_types: list[CompleteType] = field(default_factory=list)

    scope_cnt: ClassVar[int] = 0

    def __init__(self, name: str = '', parent: 'Scope' = None):
        self.functions = []
        self.variables = []
        self.children = []
        self.type_infos = {}
        self.type_variables = []
        self.full_types = []
        self.comptime_functions = []
        self.parent = parent
        self.name = name
        self.scope_number = self.scope_cnt
        self.scope_cnt += 1

    def get_type_info(self, name : str):
        if name in self.type_infos:
            return self.type_infos[name]

        if self.parent:
            return self.parent.get_type_info(name)

        return None

    def add_type_info(self, type_info : Struct):
        self.type_infos[type_info.name] = type_info

    def add_full_type(self, full_type: CompleteType):
        self.full_types.append(full_type)

    def get_scope_id(self) -> str:
        scope_id = self.name if self.name != '' else f'anonymous{self.scope_number}'
        if self.parent:
            return self.parent.get_scope_id() + '.' + scope_id
        else:
            return scope_id

    def add_var(self, what: Union['ValidatedVariableDeclarationStmt', 'ValidatedParameter']):
        var = Variable(what.name, what.type_expr().value)
        self.variables.append(var)

    def add_child_scope(self, name: str) -> 'Scope':
        scope = Scope(name=name)
        scope.parent = self
        self.children.append(scope)
        return scope

    def add_comptime_function(self, validated_function_def : 'ValidatedFunctionDefinition'):
        self.comptime_functions.append(FunctionReferenceValue(validated_function_def))

    def find_comptime_function(self, name):
        for function in reversed(self.comptime_functions):
            if function.function_definition.name().name == name:
                return function
        if self.parent: return self.parent.find_comptime_function(name)
        return None

    def get_child_scope(self, name: str) -> 'Scope':
        for scope in self.children:
            if scope.name == name:
                return scope
        raise ValueError(f"Child scope '{name}' not found")

    def find_function(self, name: str) -> Function | None:
        for function in reversed(self.functions):
            if function.name == name:
                return function
        if self.parent: return self.parent.find_function(name)
        return None

    def find_var(self, name) -> Variable | None:
        for var in reversed(self.variables):
            if var.name == name:
                return var
        if self.parent: return self.parent.find_var(name)
        return None

    def find_type_var(self, name) -> Variable | None:
        for type_var in self.type_variables:
            if type_var.name == name:
                return type_var
        if self.parent: self.parent.find_type_var(name)
        return None


def create_root_scope():
    root_scope = Scope('root', None)
    return root_scope


@dataclass
class ValidationError:
    msg: str
    span: Span


@dataclass
class ValidatedName:
    children: list['ValidatedNode']  # empty
    span: Span
    name: str


@dataclass
class ValidatedNameExpr:
    children: list['ValidatedNode']  # empty
    span: Span
    name: str
    type: CompleteType


@dataclass
class ValidatedValueExpr:
    children: list['ValidatedNode']  # empty
    span: Span
    value: any
    type: CompleteType


@dataclass
class ValidatedUnaryOperationExpr:
    children: list['ValidatedExpression']
    span: Span
    op: Operator | ComplexOperator
    type: CompleteType

    def rhs(self) -> 'ValidatedExpression': return self.children[0]


@dataclass
class ValidatedBinaryOperationExpr:
    children: list['ValidatedExpression']
    span: Span
    op: Operator
    type: CompleteType

    def lhs(self) -> 'ValidatedExpression': return self.children[0]

    def rhs(self) -> 'ValidatedExpression': return self.children[1]


@dataclass
class ValidatedCallExpr:
    children: list['ValidatedExpression']
    span: Span
    type: CompleteType
    comptime : bool

    def expr(self) -> 'ValidatedExpression': return self.children[0]

    def args(self) -> list['ValidatedExpression']: return self.children[1:]


@dataclass
class ValidatedInitializerExpr:
    children: list['ValidatedNode']
    span: Span
    type: CompleteType

    def expr(self) -> 'ValidatedExpression': return self.children[0]


@dataclass
class ValidatedDotExpr:
    children: list['ValidatedNode']
    span: Span
    type: CompleteType
    auto_deref: bool

    def expr(self) -> 'ValidatedExpression': return self.children[0]

    def name(self) -> 'ValidatedName': return self.children[1]


@dataclass
class ValidatedIndexExpr:
    children: list['ValidatedExpression']
    span: Span
    type: CompleteType

    def expr(self) -> 'ValidatedExpression': return self.children[0]

    def index(self) -> 'ValidatedExpression': return self.children[1]


@dataclass
class ValidatedSliceExpr:
    children: list['ValidatedExpression']
    span: Span
    type: CompleteType

    def expr(self) -> 'ValidatedExpression': return self.children[0]

    def start(self) -> 'ValidatedExpression': return self.children[1]

    def end(self) -> 'ValidatedExpression': return self.children[2]


@dataclass
class ValidatedArrayExpr:
    children: list['ValidatedExpression']
    span: Span
    type: CompleteType

    def expressions(self): return self.children


@dataclass
class ValidatedField:
    children: list['ValidatedNode']
    span: Span

    def name(self) -> ValidatedName: return self.children[0]

    def type_expr(self) -> 'ValidatedExpression': return self.children[1]


@dataclass
class ValidatedStructExpr:
    children: list[ValidatedField]
    span: Span
    name : str
    type: CompleteType

    def fields(self) -> list[ValidatedField]: return self.children[:]


ValidatedExpression = Union[
    ValidatedValueExpr, ValidatedNameExpr, ValidatedUnaryOperationExpr, ValidatedBinaryOperationExpr,
    ValidatedCallExpr, ValidatedDotExpr, ValidatedInitializerExpr, ValidatedIndexExpr, ValidatedArrayExpr,
    ValidatedStructExpr]


@dataclass
class ValidatedParameter:
    children: list['ValidatedNode']  # empty
    span: Span
    name: str

    def type_expr(self) -> ValidatedExpression:
        return self.children[0]


@dataclass
class ValidatedBlock:
    children: list['ValidatedStatement']
    span: Span

    def statements(self): return self.children


@dataclass
class ValidatedFunctionDefinitionPre:
    children: list['ValidatedNode']
    span: Span
    is_comptime : bool

    def name(self) -> ValidatedName: return self.children[0]

    def return_type(self) -> ValidatedExpression: return self.children[1]

    def pars(self) -> list['ValidatedParameter']: return self.children[2:]


@dataclass
class ValidatedFunctionDefinition:
    children: list['ValidatedNode']
    span: Span
    is_comptime : bool

    def name(self) -> ValidatedName: return self.children[0]

    def return_type(self) -> ValidatedExpression: return self.children[1]

    def body(self) -> ValidatedBlock: return self.children[2]

    def pars(self) -> list['ValidatedParameter']: return self.children[3:]


@dataclass
class ValidatedExternFunctionDeclaration:
    children: list['ValidatedNode']
    span: Span

    def name(self) -> ValidatedName: return self.children[0]

    def return_type(self) -> ValidatedExpression: return self.children[1]

    def pars(self) -> list['ValidatedParameter']: return self.children[2:]


OtherValidatedNodes = Union[
    'ValidatedModule', ValidatedName, ValidatedBlock, ValidatedParameter, 'ValidatedField']


@dataclass
class ValidatedReturnStmt:
    children: list['ValidatedNode']
    span: Span

    def expr(self): return self.children[0]


@dataclass
class ValidatedVariableDeclarationStmt:
    children: list['ValidatedNode']
    span: Span
    name: str

    def type_expr(self) -> ValidatedExpression: return self.children[0]
    def initializer(self) -> ValidatedExpression: return self.children[1]


@dataclass
class ValidatedWhileStmt:
    children: list['ValidatedNode']
    span: Span

    def condition(self) -> ValidatedExpression: return self.children[0]
    def block(self) -> ValidatedBlock: return self.children[1]


@dataclass
class ValidatedBreakStmt:
    children: list['ValidatedNode']
    span: Span


@dataclass
class ValidatedIfStmt:
    children: list['ValidatedNode']
    span: Span

    def condition(self) -> ValidatedExpression: return self.children[0]
    def block(self) -> ValidatedBlock: return self.children[1]


@dataclass
class ValidatedAssignmentStmt:
    children: list['ValidatedNode']
    span: Span

    def name(self) -> ValidatedExpression: return self.children[0]
    def expr(self) -> ValidatedExpression: return self.children[1]


@dataclass
class ValidatedStructPre:
    children: list['ValidatedNode']
    span: Span
    name : str
    type: CompleteType  # incomplete type, does not contain the fields


@dataclass
class SliceBoundaryPlaceholder:
    span: Span
    children: list['ValidatedNode'] = field(default_factory=list)


ValidatedStatement = Union[
    ValidatedFunctionDefinition, ValidatedReturnStmt, ValidatedVariableDeclarationStmt, ValidatedWhileStmt,
    ValidatedBreakStmt, ValidatedIfStmt, ValidatedAssignmentStmt, ValidatedExpression]

ValidatedNode = Union[ValidatedStatement, SliceBoundaryPlaceholder, OtherValidatedNodes]


@dataclass
class ValidatedModule:
    children: list['ValidatedNode']
    span: Span

    structs_in_topological_order: list[Struct]
    scope: Scope

    def body(self) -> list[Union[ValidatedFunctionDefinition]]: return self.children

    def evaluate(self):
        raise NotImplementedError(self)


def check_integer_type_compatibility(nominal: str, other: str) -> bool:
    nominal_is_signed = nominal[0] == 'i'
    nominal_size = int(nominal[1:])
    other_is_signed = other[0] == 'i'
    other_size = int(other[1:])

    if nominal_is_signed == other_is_signed:
        return nominal_size >= other_size

    if not nominal_is_signed: return False

    # nominal is signed, and other is unsigned
    # ok cases: i16 and u8, i32 and u8 or u16, i64 and u8 or u16 or u32
    return nominal_is_signed >= 2 * other_is_signed


def integer_literal_too_type(literal: str) -> str | None:
    """Returns None if the integer is too large to fit in any of the builtin integer types."""
    int_value = int(literal)

    if int_value >= 0:
        if int_value < 2 ** 8:
            return 'u8'
        elif int_value < 2 ** 16:
            return 'u16'
        elif int_value < 2 ** 32:
            return 'u32'
        elif int_value < 2 ** 64:
            return 'u64'
        else:
            return None
    else:
        if int_value >= -(2 ** 7):
            return 'i8'
        elif int_value >= -(2 ** 15):
            return 'i16'
        elif int_value >= -(2 ** 31):
            return 'i32'
        elif int_value >= -(2 ** 63):
            return 'i64'
        else:
            return None


def validate_number(type_hint: CompleteType | None, parsed_number: ParsedNumber) -> (
ValidatedValueExpr | None, ValidationError | None):
    if '.' in parsed_number.value:
        return ValidatedValueExpr([], parsed_number.span, float(parsed_number.value), CompleteType(NamedType('f64'))), None
    else:
        if builtin_name := integer_literal_too_type(parsed_number.value):
            return ValidatedValueExpr([], parsed_number.span, int(parsed_number.value),
                                       CompleteType(NamedType(builtin_name))), None
        else:
            return None, ValidationError(f'integer number {parsed_number.value} too large', parsed_number.span)


def validate_string(type_hint: CompleteType | None, parsed_str: ParsedString) -> (
ValidatedValueExpr | None, ValidationError | None):
    return ValidatedValueExpr([], parsed_str.span, parsed_str.value,
                               CompleteType(Slice(), CompleteType(NamedType('u8')))), None


def validate_unop(scope: Scope, _: CompleteType | None, parsed_unop: ParsedUnaryOperation) -> (
ValidatedUnaryOperationExpr | None, ValidationError | None):
    val_expr, error = validate_expression(scope, None, parsed_unop.rhs)
    if error: return None, error

    if val_expr.type.is_type():
        if isinstance(parsed_unop.op, ParsedOperator) and parsed_unop.op.op != Operator.Multiply:
            return None, ValidationError(f'operator not allowed for types', parsed_unop.span)

        if isinstance(parsed_unop.op, ParsedOperator) and parsed_unop.op.op == Operator.Multiply:
            return ValidatedUnaryOperationExpr([val_expr], parsed_unop.span, parsed_unop.op.op, val_expr.type), None

        if isinstance(parsed_unop.op, ParsedComplexOperator) and parsed_unop.op.op == ComplexOperator.Array:
            array_length_expr, error = validate_expression(scope, None, parsed_unop.op.par)
            if error:
                return None, error
            expr = ValidatedUnaryOperationExpr([val_expr, array_length_expr], parsed_unop.span, parsed_unop.op.op, val_expr.type)
            return expr, None


        if isinstance(parsed_unop.op, ParsedComplexOperator) and parsed_unop.op.op == ComplexOperator.Slice:
            return ValidatedUnaryOperationExpr([val_expr], parsed_unop.span, parsed_unop.op.op, val_expr.type), None

        raise NotImplementedError()

    else:
        op = parsed_unop.op.op

        if op == Operator.And:
            return ValidatedUnaryOperationExpr([val_expr], parsed_unop.span, parsed_unop.op.op,
                                               CompleteType(Pointer(), val_expr.type)), None

        if op == Operator.Multiply:
            if not val_expr.type.is_pointer():
                return None, ValidationError(f'cannot dereference type {val_expr.type}', parsed_unop.span)
            return ValidatedUnaryOperationExpr([val_expr], parsed_unop.span, parsed_unop.op.op, val_expr.type.next), None

        if isinstance(val_expr, ValidatedValueExpr) and val_expr.type.is_number() and op == Operator.Minus:
            # The parser currently can only produce positive numbers. Negative numbers will be parsed as unary operation.
            # This case is handled separately to be able to apply the knowledge of the size of the number at compile time
            # to produce the best type, for example:
            # A '-3' is parsed as -( u8 ) and becomes of type i16. To avoid "oversizing" (i16 instead of i8) we can apply
            # the knowledge that the u8 is 3, and hence -3 also fits into i8.
            integer_type_name_after_unop = integer_literal_too_type(f'-{val_expr.value}')

            if not integer_type_name_after_unop:
                return None, ValidationError(f'type {val_expr.type} does not support unary operation with operator {op}',
                                             parsed_unop.span)
            else:
                type_after_unop = CompleteType(NamedType(integer_type_name_after_unop))

        elif not (type_after_unop := is_unary_operator_defined(val_expr.type, op)):
            return None, ValidationError(f'type {val_expr.type} does not support unary operation with operator {op}',
                                         parsed_unop.span)

        return ValidatedUnaryOperationExpr([val_expr], parsed_unop.span, parsed_unop.op.op, type_after_unop), None


def validate_binop(scope: Scope, type_hint: CompleteType | None, parsed_binop: ParsedBinaryOperation) -> (
ValidatedBinaryOperationExpr | None, ValidationError | None):
    lhs, error = validate_expression(scope, None, parsed_binop.lhs)
    if error: return None, error

    rhs, error = validate_expression(scope, None, parsed_binop.rhs)
    if error: return None, error

    op = parsed_binop.op.op

    if not (type_after_binop := is_binary_operator_defined(lhs.type, rhs.type, op)):
        return None, ValidationError(
            f'type {lhs.type.get()} does no support binary operation with operator {op} and other type {rhs.type.get()}',
            parsed_binop.span)

    return ValidatedBinaryOperationExpr([lhs, rhs], parsed_binop.span, parsed_binop.op.op, type_after_binop), None


def validate_call(scope: Scope, type_hint: CompleteType | None, parsed_call: ParsedCall) -> (
ValidatedCallExpr | None, ValidationError | None):
    validated_expr, error = validate_expression(scope, None, parsed_call.expr)
    if error: return None, error

    if not validated_expr.type.is_function_ptr():
        return None, ValidationError(f'expression type {validated_expr.type.get()} not a function pointer',
                                     validated_expr.span)

    validated_args: list[ValidatedExpression] = []

    for arg in parsed_call.args:
        expr, error = validate_expression(scope, None, arg)
        if error: return None, error
        validated_args.append(expr)

    function_ptr: FunctionPointer = validated_expr.type.get()

    if len(function_ptr.pars) != len(validated_args):
        return None, ValidationError(f'Wrong number of arguments in call to function', parsed_call.span)

    for idx, (a, b) in enumerate(zip(function_ptr.pars, validated_args)):
        if not a.eq_or_other_safely_convertible(b.type):
            return None, ValidationError(
                f'Type mismatch in {idx + 1}th argument in call to function, expected={a}, got={b.type}',
                parsed_call.span)

    return ValidatedCallExpr([validated_expr, *validated_args], parsed_call.span, function_ptr.ret, function_ptr.is_comptime), None


def validate_initializer_expression(scope: Scope, type_hint: CompleteType | None,
                                    parsed_initializer_expr: ParsedInitializerExpression) -> (
ValidatedInitializerExpr | None, ValidationError | None):
    validated_type_expr, error = validate_expression(scope, None, parsed_initializer_expr.expr)
    if error: return None, error

    if not validated_type_expr.type.is_type():
        return None, ValidationError(f'expression {validated_type_expr} does not evaluate to type', validated_type_expr.span)

    if not isinstance(validated_type_expr, ValidatedValueExpr):
        return ValidatedInitializerExpr([validated_type_expr], parsed_initializer_expr.span, CompleteType(NamedType('unknown'))), None

    return ValidatedInitializerExpr([validated_type_expr], parsed_initializer_expr.span, validated_type_expr.value), None


def validate_dot_expr(scope: Scope, type_hint: CompleteType | None, parsed_dot_expr: ParsedDotExpression):
    validated_expr, error = validate_expression(scope, None, parsed_dot_expr.expr)
    if error: return None, error

    if validated_expr.type.is_type():
        return None, ValidationError(f'not implemented {validated_expr}', validated_expr.span)

    validated_name, error = validate_name(parsed_dot_expr.name)
    if error: return None, error

    dot_into = None
    auto_deref = False

    # pointers should always have a next type
    if validated_expr.type.is_pointer():
        auto_deref = True
        dot_into = validated_expr.type.next

    if validated_expr.type.is_named_type():
        dot_into = validated_expr.type

    if not dot_into:
        return None, ValidationError(f'cannot dot into type {validated_expr.type}', parsed_dot_expr.span)

    if type_info := scope.get_type_info(dot_into.named_type().name):
        if field := type_info.try_get_field(validated_name.name):
            return ValidatedDotExpr([validated_expr, validated_name], parsed_dot_expr.span, field.type, auto_deref), None

    return None, ValidationError(f'field {validated_name.name} not found', parsed_dot_expr.span)


def validate_index_expr(scope: Scope, type_hint: CompleteType | None, parsed_index_expr: ParsedIndexExpression) -> (
ValidatedIndexExpr | None, ValidationError | None):
    validated_expr, error = validate_expression(scope, None, parsed_index_expr.expr)
    if error: return None, error

    index, error = validate_expression(scope, None, parsed_index_expr.index)
    if error: return None, error

    if not index.type.is_integer():
        return None, ValidationError(f'expected integer as index, got {index.type}', parsed_index_expr.index.span)

    if validated_expr.type.is_array() or validated_expr.type.is_slice():
        return ValidatedIndexExpr([validated_expr, index], parsed_index_expr.span, validated_expr.type.next), None

    return None, ValidationError(f'cannot index {validated_expr.type}', validated_expr.span)


def validate_slice_expr(scope: Scope, type_hint: CompleteType | None, parsed_slice_expr: ParsedSliceExpression) -> (
ValidatedSliceExpr | None, ValidationError | None):
    validated_expr, error = validate_expression(scope, None, parsed_slice_expr.expr)
    if error: return None, error

    if not (validated_expr.type.is_slice() or validated_expr.type.is_array()):
        return None, ValidationError(f'Expression not sliceable', validated_expr.span)

    if parsed_slice_expr.start:
        start, error = validate_expression(scope, None, parsed_slice_expr.start)
        if error: return None, error

        if not start.type.is_integer():
            return None, ValidationError(f'expected integer as index, got {start.type}', parsed_slice_expr.start.span)
    else:
        start = SliceBoundaryPlaceholder(span=parsed_slice_expr.span)

    if parsed_slice_expr.end:
        end, error = validate_expression(scope, None, parsed_slice_expr.end)
        if error: return None, error

        if not end.type.is_integer():
            return None, ValidationError(f'expected integer as index, got {end.type}', parsed_slice_expr.end.span)
    else:
        end = SliceBoundaryPlaceholder(span=parsed_slice_expr.span)

    return ValidatedSliceExpr([validated_expr, start, end], parsed_slice_expr.span,
                              CompleteType(Slice(), validated_expr.type.next)), None


def validate_name_expr(scope: Scope, type_hint: CompleteType | None, parsed_name: ParsedName) -> (
ValidatedNameExpr | None, ValidationError | None):

    if var := scope.find_var(parsed_name.value):
        return ValidatedNameExpr([], parsed_name.span, parsed_name.value, var.type), None

    if type_var := scope.find_type_var(parsed_name.value):
        return ValidatedNameExpr([], parsed_name.span, parsed_name.value, type_var.type), None

    if function := scope.find_function(parsed_name.value):
        return ValidatedNameExpr([], parsed_name.span, parsed_name.value, function.type), None

    if parsed_name.value in ['true', 'false']:
        return ValidatedNameExpr([], parsed_name.span, parsed_name.value, CompleteType(NamedType('bool'))), None

    # Assume that name refers to a type otherwise.
    return ValidatedNameExpr([], parsed_name.span, parsed_name.value, CompleteType(NamedType('type'))), None


def validate_primary_expr(scope, type_hint: CompleteType | None, expr: ParsedPrimaryExpression):
    if isinstance(expr, ParsedName):
        return validate_name_expr(scope, type_hint, expr)
    if isinstance(expr, ParsedNumber):
        return validate_number(type_hint, expr)
    if isinstance(expr, ParsedString):
        return validate_string(type_hint, expr)
    if isinstance(expr, ParsedInitializerExpression):
        return validate_initializer_expression(scope, type_hint, expr)
    if isinstance(expr, ParsedDotExpression):
        return validate_dot_expr(scope, type_hint, expr)

    if isinstance(expr, ParsedCall):
        return validate_call(scope, type_hint, expr)
    if isinstance(expr, ParsedIndexExpression):
        return validate_index_expr(scope, type_hint, expr)
    if isinstance(expr, ParsedSliceExpression):
        return validate_slice_expr(scope, type_hint, expr)
    if isinstance(expr, ParsedArray):
        return validate_array(scope, type_hint, expr)
    if isinstance(expr, ParsedStructExpression):
        return validate_struct_expr(scope, expr)

    raise NotImplementedError(expr)


def validate_array(scope, type_hint: CompleteType | None, array: ParsedArray) -> (
ValidatedArrayExpr | None, ValidationError | None):
    validated_exprs: list[ValidatedExpression] = []
    for expr in array.exprs:
        validated_expr, error = validate_expression(scope, None, expr)
        if error: return None, error
        validated_exprs.append(validated_expr)

    # Only want to allow safe implicit conversions between integer type. This means that we do not allow implicit
    # conversions between floats and integers, or between any other types.

    # So this means:
    if not validated_exprs[0].type.is_integer():
        # All expressions in the list have to have the same type, or
        for expr in validated_exprs[1:]:
            if not validated_exprs[0].type.eq(expr.type):
                return None, ValidationError('unable to infer type of array', array.span)

        element_type = validated_exprs[0].type
    else:
        # All expressions in the list have to be compatible integer types.

        all_integer_types = ['u8', 'u16', 'u32', 'u64', 'i8', 'i16', 'i32', 'i64']
        integer_candidates = set(all_integer_types)

        integer_compatibility = {}

        # TODO: A u8 could also be compatible wit a i8, if known at compile time and smaller than 127.
        integer_compatibility['u8'] = {'u8', 'u16', 'u32', 'u64', 'i16', 'i32', 'i64'}
        integer_compatibility['u16'] = {'u16', 'u32', 'u64', 'i32', 'i64'}
        integer_compatibility['u32'] = {'u32', 'u64', 'i64'}
        integer_compatibility['u64'] = {'u64'}

        integer_compatibility['i8'] = {'i8', 'i16', 'i32', 'i64'}
        integer_compatibility['i16'] = {'i16', 'i32', 'i64'}
        integer_compatibility['i32'] = {'i32', 'i64'}
        integer_compatibility['i64'] = {'i64'}

        for expr in validated_exprs:
            if not expr.type.is_integer():
                return None, ValidationError(f'unable to infer type of array', array.span)

            integer_name = expr.type.named_type().name
            integer_candidates = integer_candidates.intersection(integer_compatibility[integer_name])

        if len(integer_candidates) == 0:
            return None, ValidationError('unable to find a common integer type for expressions array', array.span)

        # Choose the hinted type if provided and possible.
        if type_hint and type_hint.is_array() and type_hint.next.is_integer() and type_hint.next.named_type().name in integer_candidates:
            chosen_name = type_hint.next.named_type().name
        else:
            # We might have multiple candidates left, pick in order of 'all_integer_types'.
            chosen_name = 'i64'

            for name in all_integer_types:
                if name in integer_candidates:
                    chosen_name = name
                    break

        element_type = CompleteType(NamedType(chosen_name))

    return ValidatedArrayExpr(validated_exprs, array.span,
                              CompleteType(Array(len(validated_exprs)), next=element_type)), None


def validate_expression(scope: Scope, type_hint: CompleteType | None, expr: ParsedExpression) -> (
ValidatedExpression | None, ValidationError | None):

    validated_expr = None
    error = None

    if isinstance(expr, ParsedPrimaryExpression):
        validated_expr, error = validate_primary_expr(scope, type_hint, expr)
    elif isinstance(expr, ParsedUnaryOperation):
        validated_expr, error = validate_unop(scope, type_hint, expr)
    elif isinstance(expr, ParsedBinaryOperation):
        validated_expr, error = validate_binop(scope, type_hint, expr)

    if error: return None, error

    assert(validated_expr is not None)

    if not scope.is_comptime and (isinstance(validated_expr, ValidatedCallExpr) and validated_expr.comptime or validated_expr.type.is_type()):
        value = evaluate_expr(validated_expr, scope, Context())
        validated_expr = ValidatedValueExpr([], validated_expr.span, value, validated_expr.type)

    return validated_expr, None


def validate_return_stmt(scope: Scope, parsed_return_stmt: ParsedReturn) -> (
        ValidatedReturnStmt | None, ValidationError | None):
    validated_expr, error = validate_expression(scope, None, parsed_return_stmt.expression)
    if error: return None, error
    return ValidatedReturnStmt([validated_expr], parsed_return_stmt.span), None


def validate_variable_declaration(scope: Scope, parsed_variable_decl: ParsedVariableDeclaration) -> tuple[
    Optional[ValidatedVariableDeclarationStmt], Optional[ValidationError]]:
    if parsed_variable_decl.type:
        validated_type_expr, error = validate_expression(scope, None, parsed_variable_decl.type)
        if error: return None, error

        assert(isinstance(validated_type_expr, ValidatedValueExpr))

        init_expr, error = validate_expression(scope, validated_type_expr.value, parsed_variable_decl.initializer)
        if error: return None, error

        if not validated_type_expr.value.eq_or_other_safely_convertible(init_expr.type):
            return None, ValidationError(
                f'Type mismatch in variable declaration: declaration type = {validated_type_expr.value}, initialization type = {init_expr.type}',
                parsed_variable_decl.span)

        return ValidatedVariableDeclarationStmt([validated_type_expr, init_expr], parsed_variable_decl.span, parsed_variable_decl.name), None

    else:
        init_expr, error = validate_expression(scope, None, parsed_variable_decl.initializer)
        if error: return None, error
        return ValidatedVariableDeclarationStmt([ValidatedValueExpr([], init_expr.span, init_expr.type, CompleteType(NamedType('type'))), init_expr], parsed_variable_decl.span, parsed_variable_decl.name), None


def validate_while_stmt(scope: Scope, parsed_while: ParsedWhile) -> tuple[
    Optional[ValidatedWhileStmt], Optional[ValidationError]]:
    condition, error = validate_expression(scope, CompleteType(NamedType('bool')), parsed_while.condition)
    if error: return None, error

    if not condition.type.is_bool():
        return None, ValidationError(f'expected boolean expression in while condition', parsed_while.condition.span)

    block, error = validate_block(scope, parsed_while.block, while_block=True)
    if error: return None, error

    return ValidatedWhileStmt([condition, block], parsed_while.span), None


def validate_break_stmt(scope: Scope, parsed_break: ParsedBreakStatement) -> tuple[
    Optional[ValidatedBreakStmt], Optional[ValidationError]]:
    if scope.inside_while_block:
        return ValidatedBreakStmt([], parsed_break.span), None

    return None, ValidationError('break statement not in while block', parsed_break.span)


def validate_if_stmt(scope: Scope, parsed_if: ParsedIfStatement) -> tuple[
    Optional[ValidatedIfStmt], Optional[ValidationError]]:
    condition, error = validate_expression(scope, CompleteType(NamedType('bool')), parsed_if.condition)
    if error: return None, error

    if not condition.type.is_bool():
        return None, ValidationError(f'expected boolean expression in while condition', parsed_if.condition.span)

    block, error = validate_block(scope, parsed_if.body)
    if error: return None, error

    return ValidatedIfStmt([condition, block], parsed_if.span), None


def validate_assignment_stmt(scope: Scope, parsed_assignment: ParsedAssignment) -> tuple[Optional[ValidatedAssignmentStmt], Optional[ValidationError]]:
    var = scope.find_var(parsed_assignment.name.value)
    if not var:
        return None, ValidationError(f'expected valid variable', parsed_assignment.name.span)

    value_expr, error = validate_expression(scope, None, parsed_assignment.value)
    if error: return None, error

    if not var.type.eq_or_other_safely_convertible(value_expr.type):
        return None, ValidationError(f'incompatible types in assignment', parsed_assignment.span)

    return ValidatedAssignmentStmt(
        [ValidatedNameExpr([], parsed_assignment.name.span, parsed_assignment.name.value, var.type), value_expr],
        parsed_assignment.span), None


def validate_block(parent_scope: Scope, block: ParsedBlock, while_block=False) -> (
ValidatedBlock | None, ValidationError | None):
    scope = parent_scope.add_child_scope('')
    scope.inside_while_block = while_block
    scope.is_comptime = parent_scope.is_comptime
    stmts: list[ValidatedStatement] = []

    for stmt in block.statements:
        if isinstance(stmt, ParsedReturn):
            validated_return_stmt, error = validate_return_stmt(scope, stmt)
            if error: return None, error
            stmts.append(validated_return_stmt)
        elif isinstance(stmt, ParsedVariableDeclaration):
            validated_variable_decl, error = validate_variable_declaration(scope, stmt)
            if error: return None, error
            stmts.append(validated_variable_decl)
            scope.add_var(validated_variable_decl)
        elif isinstance(stmt, ParsedWhile):
            validated_while_stmt, error = validate_while_stmt(scope, stmt)
            if error: return None, error
            stmts.append(validated_while_stmt)
        elif isinstance(stmt, ParsedBreakStatement):
            validated_break_stmt, error = validate_break_stmt(scope, stmt)
            if error: return None, error
            stmts.append(validated_break_stmt)
        elif isinstance(stmt, ParsedIfStatement):
            validated_if_stmt, error = validate_if_stmt(scope, stmt)
            if error: return None, error
            stmts.append(validated_if_stmt)
        elif isinstance(stmt, ParsedAssignment):
            validated_assignment_stmt, error = validate_assignment_stmt(scope, stmt)
            if error: return None, error
            stmts.append(validated_assignment_stmt)
        elif isinstance(stmt, ParsedExpression):
            validated_expr, error = validate_expression(scope, None, stmt)
            if error: return None, error
            stmts.append(validated_expr)
        else:
            raise NotImplementedError(f'validation for statement {stmt} not implemented!')

    return ValidatedBlock(stmts, block.span), None


def validate_function_definition_pre(scope: Scope,
                                     function_definition: ParsedFunctionDefinition | ParsedExternFunctionDeclaration) -> (
ValidatedFunctionDefinitionPre | None, ValidationError | None):
    validated_name, error = validate_name(function_definition.name)
    if error: return None, error

    validated_pars: list[ValidatedParameter] = []

    # check parameter types
    for par in function_definition.pars:
        validated_type_expr, error = validate_expression(scope, None, par.type)
        if error: return None, error
        if not validated_type_expr.type.is_type():
            return None, ValidationError(f"Expected type, got '{validated_type_expr}'", validated_type_expr.span)
        validated_pars.append(ValidatedParameter([validated_type_expr], par.span, par.name.value))

    # check return type
    validated_return_type, error = validate_expression(scope, None, function_definition.return_type)
    if error: return None, error

    is_comptime = isinstance(function_definition, ParsedFunctionDefinition) and function_definition.is_comptime

    if not validated_return_type.type.is_type():
        return None, ValidationError("Expected type", validated_return_type.span)

    return ValidatedFunctionDefinitionPre(
        [validated_name, validated_return_type, *validated_pars], function_definition.span, is_comptime), None


def validate_function_definition(scope: Scope, function_definition: ParsedFunctionDefinition) -> (
ValidatedFunctionDefinition | None, ValidationError | None):
    validated_name, error = validate_name(function_definition.name)
    if error: return None, error

    validated_pars: list[ValidatedParameter] = []

    # check parameter types
    for par in function_definition.pars:
        validated_type_expr, error = validate_expression(scope, None, par.type)
        if error: return None, error
        if not validated_type_expr.type.is_type():
            return None, ValidationError("Expected type", validated_type_expr.span)
        validated_pars.append(ValidatedParameter([validated_type_expr], par.span, par.name.value))

    # check return type
    validated_return_type_expr, error = validate_expression(scope, None, function_definition.return_type)
    if error: return None, error

    if not validated_return_type_expr.type.is_type():
        return None, ValidationError("Expected type", validated_return_type_expr.span)

    # add parameters and
    child_scope = scope.add_child_scope(validated_name.name)
    child_scope.is_comptime = function_definition.is_comptime

    for validated_par in validated_pars:
        child_scope.add_var(validated_par)

    # TODO: add function to scope for recursive functions

    validated_block, error = validate_block(child_scope, function_definition.body)
    if error: return None, error

    validated_return_stmt: Optional[ValidatedReturnStmt] = None

    for validated_stmt in validated_block.statements():
        if isinstance(validated_stmt, ValidatedReturnStmt):
            validated_return_stmt = validated_stmt
            break

    if not validated_return_stmt:
        return None, ValidationError(f'missing return in function {function_definition.name}', function_definition.span)

    assert(isinstance(validated_return_type_expr, ValidatedValueExpr))
    return_type = validated_return_type_expr.value

    if not return_type.eq_or_other_safely_convertible(validated_return_stmt.expr().type):
        return None, ValidationError(
            f'Return type mismatch in function "{function_definition.name.value}": declared return type is {return_type}, but returning expression of type {validated_return_stmt.expr().type}',
            function_definition.span)

    ret = ValidatedFunctionDefinition(
        [validated_name, validated_return_type_expr,
         validated_block, *validated_pars], function_definition.span, function_definition.is_comptime)

    scope.add_comptime_function(ret)
    return ret, None


def validate_extern_function_declaration(scope: Scope,
                                         extern_function_declaration: ParsedExternFunctionDeclaration) -> (
ValidatedExternFunctionDeclaration | None, ValidationError | None):
    validated_name, error = validate_name(extern_function_declaration.name)
    if error: return None, error

    validated_pars: list[ValidatedParameter] = []

    # check parameter types
    for par in extern_function_declaration.pars:
        validated_type_expr, error = validate_expression(scope, None, par.type)
        if error: return None, error
        if not validated_type_expr.type.is_type():
            return None, ValidationError("Expected type", validated_type_expr.span)
        validated_pars.append(ValidatedParameter([validated_type_expr], par.span, par.name.value))

    # check return type
    validated_return_type, error = validate_expression(scope, None, extern_function_declaration.return_type)
    if error: return None, error

    if not validated_return_type.type.is_type():
        return None, ValidationError("Expected type", validated_return_type.span)

    return ValidatedExternFunctionDeclaration(
        [validated_name, validated_return_type,
         *validated_pars], extern_function_declaration.span), None


def validate_name(parsed_name: ParsedName) -> (ValidatedName | None, ValidationError | None):
    return ValidatedName([], parsed_name.span, parsed_name.value), None


def validate_struct_field(scope: Scope, parsed_field: ParsedField) -> (ValidatedField | None, ValidationError | None):
    name, error = validate_name(parsed_field.name)
    if error: return None, error

    validated_type_expr, error = validate_expression(scope, None, parsed_field.type)
    if error: return None, error
    if not validated_type_expr.type.is_type():
        return None, ValidationError("Expected type", validated_type_expr.span)

    return ValidatedField([name, validated_type_expr], parsed_field.span), None


def validate_struct_pre(scope: Scope, parsed_struct: ParsedStructExpression) -> (ValidatedStructPre | None, ValidationError | None):
    name, error = validate_name(parsed_struct.name)
    if error: return None, error
    return ValidatedStructPre([name], parsed_struct.span, name.name, CompleteType(NamedType('type'))), None


def validate_struct_expr(scope: Scope, parsed_struct: ParsedStructExpression) -> (ValidatedStructExpr | None, ValidationError | None):
    name = None

    if parsed_struct.name:
        validated_name, error = validate_name(parsed_struct.name)
        if error: return None, error
        name = validated_name.name

    fields: list[ValidatedField] = []
    for field in parsed_struct.fields:
        validated_field, error = validate_struct_field(scope, field)
        if error: return None, error
        fields.append(validated_field)

    return ValidatedStructExpr([*fields], parsed_struct.span, name, CompleteType(NamedType('type'))), None


def validate_module(module: ParsedModule) -> (ValidatedModule | None, ValidationError | None):
    root_scope = create_root_scope()
    body: list[Union[ValidatedFunctionDefinition, ValidatedStructExpr, ValidatedVariableDeclarationStmt]] = []

    # pre pass 1
    for stmt in module.body:
        if isinstance(stmt, ParsedStructExpression):
            validated_struct_pre, error = validate_struct_pre(root_scope, stmt)
            if error: return None, error
            root_scope.add_type_info(Struct(validated_struct_pre.name, [], ''))

    # pre pass 2
    for stmt in module.body:
        if isinstance(stmt, ParsedFunctionDefinition) or isinstance(stmt, ParsedExternFunctionDeclaration):
            validated_function_def_pre, error = validate_function_definition_pre(root_scope, stmt)
            if error: return None, error
            function_par_types = [par.type_expr().value for par in validated_function_def_pre.pars()]
            return_type = validated_function_def_pre.return_type().value
            function_type = CompleteType(
            FunctionPointer(function_par_types, return_type, validated_function_def_pre.is_comptime))
            root_scope.functions.append(Function(validated_function_def_pre.name().name, function_type))

    # main pass 1
    for stmt in module.body:
        if isinstance(stmt, ParsedStructExpression):
            validated_struct_expr, error = validate_expression(root_scope, None, stmt)
            if error: return None, error
            body.append(validated_struct_expr)

    # main pass 2
    for stmt in module.body:
        if isinstance(stmt, ParsedFunctionDefinition):
            validated_function_def, error = validate_function_definition(root_scope, stmt)
            if error: return None, error
            body.append(validated_function_def)
        elif isinstance(stmt, ParsedExternFunctionDeclaration):
            validated_extern_function_decl, error = validate_extern_function_declaration(root_scope, stmt)
            if error: return None, error
            body.append(validated_extern_function_decl)
        elif isinstance(stmt, ParsedVariableDeclaration):
            validated_variable_declaration, error = validate_variable_declaration(root_scope, stmt)
            if error: return None, error
            body.append(validated_variable_declaration)
            root_scope.add_var(validated_variable_declaration)

    # Determine order of structs and detect cycles.
    scopes = [root_scope]

    struct_dict: dict[str, Struct] = {}

    # Collect all structs in all scopes.
    while len(scopes) > 0:
        scope = scopes.pop()
        for name, struct in scope.type_infos.items():
            struct_dict[name] = struct

        for child in scope.children:
            scopes.append(child)

    # Build a dependency graph between the structs.
    out_edges = defaultdict(list)
    in_edges = defaultdict(list)

    no_dependency = set()

    for struct_id, struct in struct_dict.items():
        for field in struct.fields:
            current = field.type

            # TODO: Handle optionals properly, once introduced. Optionals to a unsized derivative of a struct, e.g.
            #       pointers (and probably slices?), are not a soft dependency.
            while current:
                # Pointers and slices to struct do not depend on the size of the struct and hence are only a soft dependency.
                if current.is_builtin() or current.is_pointer() or current.is_slice():
                    break

                # Decay Arrays down to the fundamental types.
                if current.is_array():
                    current = current.next
                    continue

                # Other NamedTypes
                assert(current.is_named_type())

                # TODO: lookup full name (for the lack of a better word)

                dependency = current.named_type().name
                if dependency in struct_dict:
                    out_edges[struct_id].append(dependency)
                    in_edges[dependency].append(struct_id)

                break

        if len(out_edges[struct_id]) == 0:
            no_dependency.add(struct_id)

    # Kahn's algorithm for topological sorting.
    ordered = []
    while len(no_dependency) > 0:
        struct_id = no_dependency.pop()

        for other in in_edges[struct_id]:
            out_edges[other].remove(struct_id)
            if len(out_edges[other]) == 0:
                no_dependency.add(other)

        ordered.append(struct_id)

    error_msg = ''

    # Dependency graph has cycles there are any edges left, i.e. if any of node's out edge list is not empty.
    for struct_id, edges in out_edges.items():
        if len(edges) > 0:
            error_msg += f"Cyclic dependency: {struct_id} -> {edges}\n"

    if len(error_msg) > 0:
        return None, ValidationError(error_msg, module.span)

    return ValidatedModule(body, module.span, [struct_dict[struct_id] for struct_id in ordered], root_scope), None


def visit_nodes(node: ValidatedNode, visitor: Callable[[ValidatedNode], None]):
    visitor(node)

    for child in node.children:
        visit_nodes(child, visitor)


class Context:
    stack : list[Variable]
    markers : list[int]

    def __init__(self):
        self.stack = []
        self.markers = []

    def stack_push(self):
        self.markers.append(len(self.stack))

    def stack_pop(self):
        mark = self.markers.pop()

        while len(self.stack) > mark:
            self.stack.pop()

    def resolve_stack_value(self, name : str) -> Variable | None:
        for var in reversed(self.stack):
            if var.name == name:
                return var
        return None


def evaluate_block(block: ValidatedBlock, scope, context: Context) -> (int, Optional[Value]):
    BLOCK_EXIT_STATUS_NONE = 0
    BLOCK_EXIT_STATUS_RETURN = 1
    BLOCK_EXIT_STATUS_BREAK = 2

    status = BLOCK_EXIT_STATUS_NONE
    val = None

    context.stack_push()

    for stmt in block.children:
        if isinstance(stmt, ValidatedVariableDeclarationStmt):
            type_value = evaluate_expr(stmt.type_expr(), scope, context)
            init_value = evaluate_expr(stmt.initializer(), scope, context)
            context.stack.append(Variable(stmt.name, type_value, init_value))
        elif isinstance(stmt, ValidatedIfStmt):
            cond = evaluate_expr(stmt.condition(), scope, context)
            if cond:
                status, val = evaluate_block(stmt.block(), scope, context)
                if status != BLOCK_EXIT_STATUS_NONE:
                    break
        elif isinstance(stmt, ValidatedReturnStmt):
            status = BLOCK_EXIT_STATUS_RETURN
            val = evaluate_expr(stmt.expr(), scope, context)
            break
        elif isinstance(stmt, ValidatedBreakStmt):
            status = BLOCK_EXIT_STATUS_BREAK
            break
        elif isinstance(stmt, ValidatedAssignmentStmt):
            var = context.resolve_stack_value(stmt.name().name)
            var.value = evaluate_expr(stmt.expr(), scope, context)
        elif isinstance(stmt, ValidatedWhileStmt):
            while evaluate_expr(stmt.condition(), scope, context):
                status, val = evaluate_block(stmt.block(), scope, context)
                if status == BLOCK_EXIT_STATUS_BREAK:
                    break
                if status == BLOCK_EXIT_STATUS_RETURN:
                    break
            if status == BLOCK_EXIT_STATUS_RETURN:
                break
        else:
            raise NotImplementedError(stmt)

    context.stack_pop()

    return status, val


def evaluate_expr(expr: ValidatedExpression, scope: Scope, context: Context) -> Value:
    if isinstance(expr, ValidatedValueExpr):
        return expr.value

    if isinstance(expr, ValidatedNameExpr):
        var = context.resolve_stack_value(expr.name)
        if var:
            return var.value

        if expr.type.is_type():
            return CompleteType(NamedType(expr.name))

        return scope.find_comptime_function(expr.name)

    if isinstance(expr, ValidatedUnaryOperationExpr):
        if expr.type.is_type():

            if isinstance(expr.op, Operator) and expr.op != Operator.Multiply:
                raise NotImplementedError()

            if isinstance(expr.op, Operator) and expr.op == Operator.Multiply:
                nested_type = evaluate_expr(expr.rhs(), scope, context)
                return CompleteType(Pointer(), nested_type)

            if isinstance(expr.op, ComplexOperator) and expr.op == ComplexOperator.Array:
                array_length = evaluate_expr(expr.children[1], scope, context)
                nested_type = evaluate_expr(expr.rhs(), scope, context)
                return CompleteType(Array(array_length), nested_type)

            if isinstance(expr.op, ComplexOperator) and expr.op == ComplexOperator.Slice:
                nested_type = evaluate_expr(expr.rhs(), scope, context)
                return CompleteType(Slice(), nested_type)

            raise NotImplementedError()

        elif isinstance(expr.op, Operator):
            if expr.op == Operator.Plus:
                return evaluate_expr(expr, scope, context)
            if expr.op == Operator.Minus:
                return -evaluate_expr(expr, scope, context)

        raise NotImplementedError()

    if isinstance(expr, ValidatedBinaryOperationExpr):
        if expr.type.is_integer() or expr.type.is_floating_point() or expr.type.is_bool():
            if expr.op == Operator.Plus:
                return evaluate_expr(expr.lhs(), scope, context) + evaluate_expr(expr.rhs(), scope, context)
            if expr.op == Operator.Minus:
                return evaluate_expr(expr.lhs(), scope, context) - evaluate_expr(expr.rhs(), scope, context)
            if expr.op == Operator.Multiply:
                return evaluate_expr(expr.lhs(), scope, context) * evaluate_expr(expr.rhs(), scope, context)
            if expr.op == Operator.Divide:
                return evaluate_expr(expr.lhs(), scope, context) / evaluate_expr(expr.rhs(), scope, context)
            if expr.op == Operator.LessThan:
                return evaluate_expr(expr.lhs(), scope, context) <= evaluate_expr(expr.rhs(), scope, context)
            if expr.op == Operator.Equals:
                return evaluate_expr(expr.lhs(), scope, context) == evaluate_expr(expr.rhs(), scope, context)
            if expr.op == Operator.And:
                return evaluate_expr(expr.lhs(), scope, context) and evaluate_expr(expr.rhs(), scope, context)

        raise NotImplementedError()

    if isinstance(expr, ValidatedCallExpr):
        function_reference : FunctionReferenceValue = evaluate_expr(expr.expr(), scope, context)
        context.stack_push()
        for par, arg in zip(function_reference.function_definition.pars(), expr.args()):
            par_type_value = evaluate_expr(par.type_expr(), scope, context)
            var = Variable(par.name, par_type_value, evaluate_expr(arg, scope, context))
            context.stack.append(var)
        _, return_value = evaluate_block(function_reference.function_definition.body(), scope, context)
        context.stack_pop()
        return return_value

    if isinstance(expr, ValidatedStructExpr):
        fields = []

        signature = ''

        for field in expr.fields():
            field_type = evaluate_expr(field.type_expr(), scope, context)
            fields.append(Struct.StructField(field.name().name, field_type))
            signature += f"{field.name().name}_{field_type.to_string()}__"

        signature = hash(signature) + sys.maxsize + 1 # we want positive numbers

        name = expr.name
        if name is None:
            name = '___anonymous_struct__' + str(signature)

        struct = Struct(name, fields, '')
        scope.add_type_info(struct)
        return CompleteType(NamedType(struct.name))

    if isinstance(expr, ValidatedArrayExpr):
        raise NotImplementedError(expr)

    if isinstance(expr, ValidatedDotExpr):
        raise NotImplementedError(expr)

    if isinstance(expr, ValidatedInitializerExpr):
        raise NotImplementedError(expr)

    if isinstance(expr, ValidatedIndexExpr):
        raise NotImplementedError(expr)


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

#
# def print_validated_module(module: ValidatedModule) -> None:
#     pprint.pprint(module, width=1, compact=True)


def print_validator_error(error: ValidationError, text: str):
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

    print(Colors.FAIL + f'Validation error at {line_idx + 1}:{error.span.start - idx + 1}:\n' + Colors.ENDC)

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
    parsed_module, error = parse_module(token_source)

    if error: return print_parser_error(error, text)

    validated_module, error = validate_module(parsed_module)
    if error: return print(error)

if __name__ == '__main__':
    main()
