import sys
from enum import Enum
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

builtin_types = ['i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64', 'bool', 'f32', 'f64', 'void', 'type']

# TODO: Make sure to copy list when assigning array values
ArrayValue = list['Value']
StructValue = dict[str, 'Value']


@dataclass
class SliceValue:
    start: int
    end: int
    ptr: ArrayValue
    byte_offset = 0


Value = Union[int, float, bool, 'CompleteType', ArrayValue, StructValue, 'SliceValue']


@dataclass
class Variable:
    name: str
    type: 'CompleteType'
    value: Optional[Value]
    is_comptime: bool


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

    def is_struct(self) -> bool:
        return self.is_named_type() and not self.named_type().name in builtin_types

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

    def is_u8(self) -> bool:
        return self.is_named_type() and self.named_type().name == 'u8'

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

    def __str__(self):
        return self.to_string()

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


Field = Struct("Field", [
    Struct.StructField("name", CompleteType(Slice(), CompleteType(NamedType("u8"))))
], '')

TypeInfo = Struct("TypeInfo", [
    Struct.StructField("fields", CompleteType(Slice(), CompleteType(NamedType("Field"))))
], '')


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
    if lhs.is_bool() and rhs.is_bool() and op == Operator.And:
        return CompleteType(NamedType('bool'))

    if lhs.is_type() and rhs.is_type() and op == Operator.Equals:
        return CompleteType(NamedType('bool'))

    if lhs.is_number():
        match op:
            case Operator.Minus | Operator.Plus | Operator.Minus | Operator.Multiply | Operator.Divide:
                return lhs
            case Operator.Equals | Operator.LessThan:
                return CompleteType(NamedType('bool'))
            case _:
                return None

    return None


@dataclass
class Scope:
    name: str = ''
    children: list['Scope'] = field(default_factory=list)
    parent: 'Scope' = None
    inside_while_block: bool = False
    scope_number: int = 0

    functions: list['ValidatedFunctionDefinition'] = field(default_factory=list)
    comptime_function_definitions: list[ParsedFunctionDefinition] = field(default_factory=list)

    type_infos: dict[str, Struct] = field(default_factory=dict)

    scope_cnt: ClassVar[int] = 0

    def __init__(self, name: str = '', parent: 'Scope' = None):
        self.functions = []
        self.comptime_function_definitions = []
        self.vars = []
        self.children = []
        self.type_infos = {}
        self.parent = parent
        self.name = name
        self.scope_number = self.scope_cnt
        self.scope_cnt += 1

    def get_root_scope(self) -> 'Scope':
        scope = self
        while scope.parent:
            scope = scope.parent
        return scope

    def get_type_info(self, name: str) -> Struct | None:
        if name in self.type_infos:
            return self.type_infos[name]

        if self.parent:
            return self.parent.get_type_info(name)

        return None

    def check_type_exists(self, complete_type: CompleteType) -> bool:
        current = complete_type
        while current.is_array() or current.is_slice() or current.is_pointer():
            current = current.next

        assert (current.is_named_type())

        named_type = current.named_type()
        if named_type.name in builtin_types:
            return True

        node = self
        while node:
            # TODO: What about the order in which we should check either type_info or type_variable?
            if named_type.name in node.type_infos:
                return True
            for var in reversed(node.vars):
                if var.name == named_type.name:
                    return var.type.is_type()
            node = node.parent

        return False

    def add_type_info(self, type_info: Struct):
        self.type_infos[type_info.name] = type_info

    def get_scope_id(self) -> str:
        scope_id = self.name if self.name != '' else f'anonymous{self.scope_number}'
        if self.parent:
            return self.parent.get_scope_id() + '.' + scope_id
        else:
            return scope_id

    def add_var(self, var: Variable):
        self.vars.append(var)

    def add_child_scope(self, name: str) -> 'Scope':
        scope = Scope(name=name)
        scope.parent = self
        self.children.append(scope)
        return scope

    def add_function(self, function):
        self.functions.append(function)

    def add_comptime_function_definition(self, parsed_function_definition: ParsedFunctionDefinition):
        self.comptime_function_definitions.append(parsed_function_definition)

    def find_comptime_function_definition(self, name) -> ParsedFunctionDefinition | None:
        for parsed_comptime_function_definition in reversed(self.comptime_function_definitions):
            if parsed_comptime_function_definition.name.value == name:
                return parsed_comptime_function_definition
        if self.parent:
            return self.parent.find_comptime_function_definition(name)
        return None

    def find_function(self, name) -> Optional['ValidatedFunctionDefinition']:
        for function in reversed(self.functions):
            if function.name().name == name:
                return function
        if self.parent:
            return self.parent.find_function(name)
        return None

    def get_child_scope(self, name: str) -> 'Scope':
        for scope in self.children:
            if scope.name == name:
                return scope
        raise ValueError(f"Child scope '{name}' not found")

    def find_var(self, name) -> Variable | None:
        for var in reversed(self.vars):
            if var.name == name:
                return var
        if self.parent:
            return self.parent.find_var(name)
        return None


def create_root_scope():
    root_scope = Scope('root', None)
    return root_scope


@dataclass
class ValidationError:
    msg: str
    span: Span

    def __init__(self, msg, span):
        self.msg = msg
        self.span = span


@dataclass
class ValidatedName:
    children: list['ValidatedNode']  # empty
    span: Span
    name: str


class ExpressionMode(Enum):
    Variable = 1
    Value = 2


@dataclass
class ValidatedNameExpr:
    children: list['ValidatedNode']  # empty
    span: Span
    type: CompleteType
    is_comptime: bool
    mode: ExpressionMode
    name: str


@dataclass
class ValidatedComptimeValueExpr:
    children: list['ValidatedNode']  # empty
    span: Span
    type: CompleteType
    is_comptime: bool
    mode: ExpressionMode
    value: any


@dataclass
class ValidatedUnaryOperationExpr:
    children: list['ValidatedExpression']
    span: Span
    type: CompleteType
    is_comptime: bool
    mode: ExpressionMode
    op: Operator | ComplexOperator

    def rhs(self) -> 'ValidatedExpression': return self.children[0]


@dataclass
class ValidatedBinaryOperationExpr:
    children: list['ValidatedExpression']
    span: Span
    type: CompleteType
    is_comptime: bool
    mode: ExpressionMode
    op: Operator

    def lhs(self) -> 'ValidatedExpression': return self.children[0]

    def rhs(self) -> 'ValidatedExpression': return self.children[1]


@dataclass
class ValidatedCallExpr:
    children: list['ValidatedExpression']
    span: Span
    type: CompleteType
    is_comptime: bool
    mode: ExpressionMode
    name: str

    comptime_arg_count: int

    def expr(self) -> 'ValidatedExpression': return self.children[0]

    def args(self) -> list['ValidatedExpression']: return self.children[1:]


@dataclass
class ValidatedTypeInfoCallExpr:
    children: list['ValidatedExpression']
    span: Span
    type: CompleteType
    is_comptime: bool
    mode: ExpressionMode

    def args(self) -> list['ValidatedExpression']: return self.children


@dataclass
class ValidatedInitializerExpr:
    children: list['ValidatedNode']
    span: Span
    type: CompleteType
    is_comptime: bool
    mode: ExpressionMode

    def expr(self) -> 'ValidatedExpression': return self.children[0]


@dataclass
class ValidatedDotExpr:
    children: list['ValidatedNode']
    span: Span
    type: CompleteType
    is_comptime: bool
    mode: ExpressionMode
    auto_deref: bool

    def expr(self) -> 'ValidatedExpression': return self.children[0]

    def name(self) -> 'ValidatedName': return self.children[1]


@dataclass
class ValidatedIndexExpr:
    children: list['ValidatedExpression']
    span: Span
    type: CompleteType
    is_comptime: bool
    mode: ExpressionMode

    def expr(self) -> 'ValidatedExpression': return self.children[0]

    def index(self) -> 'ValidatedExpression': return self.children[1]


@dataclass
class ValidatedSliceExpr:
    children: list['ValidatedExpression']
    span: Span
    type: CompleteType
    is_comptime: bool
    mode: ExpressionMode

    def src(self) -> 'ValidatedExpression': return self.children[0]

    def start(self) -> 'ValidatedExpression': return self.children[1]

    def end(self) -> 'ValidatedExpression': return self.children[2]


@dataclass
class ValidatedArrayExpr:
    children: list['ValidatedExpression']
    span: Span
    type: CompleteType
    is_comptime: bool
    mode: ExpressionMode

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
    type: CompleteType
    is_comptime: bool
    mode: ExpressionMode
    name: str

    def fields(self) -> list[ValidatedField]: return self.children[:]


ValidatedExpression = Union[
    ValidatedComptimeValueExpr, ValidatedNameExpr, ValidatedUnaryOperationExpr, ValidatedBinaryOperationExpr,
    ValidatedCallExpr, ValidatedDotExpr, ValidatedInitializerExpr, ValidatedIndexExpr, ValidatedArrayExpr,
    ValidatedStructExpr]


@dataclass
class ValidatedParameter:
    children: list['ValidatedNode']
    span: Span
    name: str
    is_comptime: bool

    bound_value: Optional[Value] = None

    def type_expr(self) -> ValidatedExpression:
        return self.children[0]


@dataclass
class ValidatedBlock:
    children: list['ValidatedStatement']
    span: Span

    def statements(self): return self.children


@dataclass
class ValidatedFunctionDefinition:
    children: list['ValidatedNode']
    span: Span

    is_incomplete: bool
    is_extern: bool
    is_comptime: bool

    def name(self) -> ValidatedName:
        return self.children[0]

    def return_type(self) -> ValidatedExpression:
        return self.children[1]

    def body(self) -> ValidatedBlock:
        if self.is_incomplete:
            raise LookupError('Function has now body yet')
        return self.children[2]

    def pars(self) -> list['ValidatedParameter']:
        if self.is_incomplete: return self.children[2:]
        return self.children[3:]

    def type(self) -> CompleteType:
        par_types = [p.type_expr().value for p in self.pars()]
        ret_type = self.return_type().value
        return CompleteType(FunctionPointer(par_types, ret_type))


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
    is_comptime: bool

    def expr(self): return self.children[0]


@dataclass
class ValidatedVariableDeclarationStmt:
    children: list['ValidatedNode']
    span: Span
    is_comptime: bool
    name: str

    def type_expr(self) -> ValidatedExpression: return self.children[0]

    def initializer(self) -> ValidatedExpression: return self.children[1]


@dataclass
class ValidatedWhileStmt:
    children: list['ValidatedNode']
    span: Span
    is_comptime: bool

    def condition(self) -> ValidatedExpression: return self.children[0]

    def block(self) -> ValidatedBlock: return self.children[1]


@dataclass
class ValidatedBreakStmt:
    children: list['ValidatedNode']
    span: Span
    is_comptime: bool


@dataclass
class ValidatedIfStmt:
    children: list['ValidatedNode']
    span: Span
    is_comptime: bool

    def condition(self) -> ValidatedExpression: return self.children[0]

    def block(self) -> ValidatedBlock: return self.children[1]


@dataclass
class ValidatedAssignmentStmt:
    children: list['ValidatedNode']
    span: Span
    is_comptime: bool

    def to(self) -> ValidatedExpression: return self.children[0]

    def expr(self) -> ValidatedExpression: return self.children[1]


@dataclass
class ValidatedStructPre:
    children: list['ValidatedNode']
    span: Span
    name: str
    type: CompleteType  # incomplete type, does not contain the fields


@dataclass
class SliceBoundaryPlaceholder:
    span: Span
    children: list['ValidatedNode'] = field(default_factory=list)


@dataclass
class ValidatedExpressionStmt:
    children: list['ValidatedExpression']
    span: Span
    is_comptime: bool

    def expr(self) -> ValidatedExpression:
        return self.children[0]


ValidatedStatement = Union[
    ValidatedFunctionDefinition, ValidatedReturnStmt, ValidatedVariableDeclarationStmt, ValidatedWhileStmt,
    ValidatedBreakStmt, ValidatedIfStmt, ValidatedAssignmentStmt, ValidatedExpressionStmt]

ValidatedNode = Union[ValidatedStatement, ValidatedExpression, SliceBoundaryPlaceholder, OtherValidatedNodes]


@dataclass
class ValidatedModule:
    children: list['ValidatedNode']
    span: Span

    structs_in_topological_order: list[Struct]
    scope: Scope

    def body(self) -> list[Union[ValidatedFunctionDefinition]]: return self.children


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


def integer_literal_to_type(literal: str) -> str | None:
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
        ValidatedComptimeValueExpr | None, ValidationError | None):
    if '.' in parsed_number.value:
        return ValidatedComptimeValueExpr([], parsed_number.span, CompleteType(NamedType('f64')), True, ExpressionMode.Value,
                                          float(parsed_number.value)), None
    else:
        if builtin_name := integer_literal_to_type(parsed_number.value):
            return ValidatedComptimeValueExpr([], parsed_number.span,
                                              CompleteType(NamedType(builtin_name)), True, ExpressionMode.Value,
                                              int(parsed_number.value)), None
        else:
            return None, ValidationError(f'integer number {parsed_number.value} too large', parsed_number.span)


def str_to_slicevalue(s : str) -> SliceValue:
    assert(isinstance(s, str))
    bytes = s.encode('utf-8')
    return SliceValue(0, len(bytes), list(bytes))


def validate_string(type_hint: CompleteType | None, parsed_str: ParsedString) -> (
        ValidatedComptimeValueExpr | None, ValidationError | None):
    return ValidatedComptimeValueExpr([], parsed_str.span, CompleteType(Slice(), CompleteType(NamedType('u8'))), True,
                                      ExpressionMode.Value, str_to_slicevalue(parsed_str.value)), None


def validate_unop(scope: Scope, _: CompleteType | None, force_evaluation: bool, parsed_unop: ParsedUnaryOperation) -> (
        ValidatedUnaryOperationExpr | None, ValidationError | None):
    val_expr, error = validate_expression(scope, None, force_evaluation, parsed_unop.rhs)
    if error: return None, error

    if val_expr.type.is_type():
        if isinstance(parsed_unop.op, ParsedOperator) and parsed_unop.op.op != Operator.Multiply:
            return None, ValidationError(f'operator not allowed for types', parsed_unop.span)

        if isinstance(parsed_unop.op, ParsedOperator) and parsed_unop.op.op == Operator.Multiply:
            return ValidatedUnaryOperationExpr([val_expr], parsed_unop.span, val_expr.type, True, ExpressionMode.Value,
                                               parsed_unop.op.op), None

        if isinstance(parsed_unop.op, ParsedComplexOperator) and parsed_unop.op.op == ComplexOperator.Array:
            array_length_expr, error = validate_expression(scope, None, force_evaluation, parsed_unop.op.par)
            if error:
                return None, error
            expr = ValidatedUnaryOperationExpr([val_expr, array_length_expr], parsed_unop.span, val_expr.type, True,
                                               ExpressionMode.Value, parsed_unop.op.op)
            return expr, None

        if isinstance(parsed_unop.op, ParsedComplexOperator) and parsed_unop.op.op == ComplexOperator.Slice:
            return ValidatedUnaryOperationExpr([val_expr], parsed_unop.span, val_expr.type, True, ExpressionMode.Value,
                                               parsed_unop.op.op), None

        raise NotImplementedError()

    else:
        op = parsed_unop.op.op

        if op == Operator.Address:
            return ValidatedUnaryOperationExpr([val_expr], parsed_unop.span,
                                               CompleteType(Pointer(), val_expr.type), False,
                                               ExpressionMode.Value, parsed_unop.op.op), None

        if op == Operator.Multiply:
            if not val_expr.type.is_pointer():
                return None, ValidationError(f'cannot dereference type {val_expr.type}', parsed_unop.span)
            return ValidatedUnaryOperationExpr([val_expr], parsed_unop.span, val_expr.type.next, val_expr.is_comptime,
                                               ExpressionMode.Variable, parsed_unop.op.op), None

        if isinstance(val_expr, ValidatedComptimeValueExpr) and val_expr.type.is_number() and op == Operator.Minus:
            # The parser currently can only produce positive numbers. Negative numbers will be parsed as unary operation.
            # This case is handled separately to be able to apply the knowledge of the size of the number at compile time
            # to produce the best type, for example:
            # A '-3' is parsed as -( u8 ) and becomes of type i16. To avoid "oversizing" (i16 instead of i8) we can apply
            # the knowledge that the u8 is 3, and hence -3 also fits into i8.
            integer_type_name_after_unop = integer_literal_to_type(f'-{val_expr.value}')

            if not integer_type_name_after_unop:
                return None, ValidationError(
                    f'type {val_expr.type} does not support unary operation with operator {op}',
                    parsed_unop.span)
            else:
                type_after_unop = CompleteType(NamedType(integer_type_name_after_unop))

        elif not (type_after_unop := is_unary_operator_defined(val_expr.type, op)):
            return None, ValidationError(f'type {val_expr.type} does not support unary operation with operator {op}',
                                         parsed_unop.span)

        return ValidatedUnaryOperationExpr([val_expr], parsed_unop.span, type_after_unop, val_expr.is_comptime,
                                           ExpressionMode.Value, parsed_unop.op.op), None


def validate_binop(scope: Scope, type_hint: CompleteType | None, force_evaluation: bool,
                   parsed_binop: ParsedBinaryOperation) -> (
        ValidatedBinaryOperationExpr | None, ValidationError | None):
    lhs, error = validate_expression(scope, None, force_evaluation, parsed_binop.lhs)
    if error: return None, error

    rhs, error = validate_expression(scope, None, force_evaluation, parsed_binop.rhs)
    if error: return None, error

    op = parsed_binop.op.op

    if not (type_after_binop := is_binary_operator_defined(lhs.type, rhs.type, op)):
        return None, ValidationError(
            f'type {lhs.type} does no support binary operation with operator {op} and other type {rhs.type}',
            parsed_binop.span)

    return ValidatedBinaryOperationExpr([lhs, rhs], parsed_binop.span, type_after_binop,
                                        lhs.is_comptime and rhs.is_comptime, ExpressionMode.Value,
                                        parsed_binop.op.op), None


def validate_call(scope: Scope, type_hint: CompleteType | None, parsed_call: ParsedCall) -> (
        ValidatedCallExpr | ValidatedTypeInfoCallExpr | None, ValidationError | None):
    if isinstance(parsed_call.expr, ParsedName) and parsed_call.expr.value == 'typeinfo':
        if len(parsed_call.args) != 1:
            return None, ValidationError(
                f'expecte 1 argument at call to "typeinfo", got ${len(parsed_call.args)}', parsed_call.span)

        validated_arg, error = validate_expression(scope, None, True, parsed_call.args[0])
        if error: return None, error

        # FIXME: name expr?
        return ValidatedTypeInfoCallExpr([validated_arg], parsed_call.span, CompleteType(NamedType('TypeInfo')),
                                         True, mode=ExpressionMode.Value), None

    # Assumption: validated_expr will be a ValidatedNameExpr
    callee_expr, error = validate_expression(scope, None, False, parsed_call.expr, parsed_call.args)
    if error: return None, error

    if not callee_expr.type.is_function_ptr():
        return None, ValidationError(f'expression type {callee_expr.type.get()} not a function pointer',
                                     callee_expr.span)

    validated_args: list[ValidatedExpression] = []

    for arg in parsed_call.args:
        expr, error = validate_expression(scope, None, False, arg)
        if error: return None, error
        validated_args.append(expr)

    function_ptr: FunctionPointer = callee_expr.type.get()

    if len(function_ptr.pars) != len(validated_args):
        return None, ValidationError(f'Wrong number of arguments in call to function', parsed_call.span)

    if callee_expr.name == 'slicelen':
        assert (function_ptr.pars[0].is_slice())
    else:
        for idx, (a, b) in enumerate(zip(function_ptr.pars, validated_args)):
            if not a.eq_or_other_safely_convertible(b.type):
                return None, ValidationError(
                    f'Type mismatch in {idx + 1}th argument in call to function, expected={a}, got={b.type}',
                    parsed_call.span)

    if callee_expr.name == 'typeinfo':
        function_name = 'typeinfo'
        function_is_comptime = True
        comptime_par_count = 1
    elif callee_expr.name == 'slicelen':
        function_name = 'slicelen'
        function_is_comptime = True
        comptime_par_count = 1
    else:
        function = scope.get_root_scope().find_function(callee_expr.name)
        function_is_comptime = function.is_comptime
        function_name = function.name().name
        comptime_par_count = len(list(filter(lambda par: par.is_comptime, function.pars())))

    return ValidatedCallExpr([callee_expr, *validated_args], parsed_call.span, function_ptr.ret,
                             function_is_comptime, ExpressionMode.Value, function_name,
                             comptime_arg_count=comptime_par_count), None


def validate_initializer_expression(scope: Scope, type_hint: CompleteType | None,
                                    parsed_initializer_expr: ParsedInitializerExpression) -> (
        ValidatedInitializerExpr | None, ValidationError | None):
    validated_type_expr, error = validate_expression(scope, None, True, parsed_initializer_expr.expr)
    if error: return None, error

    if not validated_type_expr.type.is_type():
        return None, ValidationError(f'expression {validated_type_expr} does not evaluate to type',
                                     validated_type_expr.span)

    if not isinstance(validated_type_expr, ValidatedComptimeValueExpr):
        return ValidatedInitializerExpr([validated_type_expr], parsed_initializer_expr.span,
                                        CompleteType(NamedType('unknown')), True, ExpressionMode.Value), None

    # no pointers allowed to be elligble to comptime evaluation:
    def check_type_contains_pointers(scope: Scope, typ: CompleteType):
        if typ.is_pointer():
            return True

        if typ.is_named_type() and not typ.is_builtin():
            type_info = scope.get_type_info(typ.named_type().name)
            for field in type_info.fields:
                if check_type_contains_pointers(scope, field.type):
                    return True

        return False

    is_comptime = not check_type_contains_pointers(scope, validated_type_expr.value)

    return ValidatedInitializerExpr([validated_type_expr], parsed_initializer_expr.span, validated_type_expr.value,
                                    is_comptime, ExpressionMode.Value), None


def validate_dot_expr(scope: Scope, type_hint: CompleteType | None,
                      parsed_dot_expr: ParsedDotExpression):
    validated_expr, error = validate_expression(scope, None, False, parsed_dot_expr.expr)
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
            return ValidatedDotExpr([validated_expr, validated_name], parsed_dot_expr.span, field.type,
                                    validated_expr.is_comptime,
                                    validated_expr.mode, auto_deref), None

    return None, ValidationError(f'field {validated_name.name} not found', parsed_dot_expr.span)


def validate_index_expr(scope: Scope, type_hint: CompleteType | None,
                        parsed_index_expr: ParsedIndexExpression) -> (
        ValidatedIndexExpr | None, ValidationError | None):
    validated_expr, error = validate_expression(scope, None, False, parsed_index_expr.expr)
    if error: return None, error

    index, error = validate_expression(scope, None, False, parsed_index_expr.index)
    if error: return None, error

    if not index.type.is_integer():
        return None, ValidationError(f'expected integer as index, got {index.type}', parsed_index_expr.index.span)

    is_comptime = validated_expr.is_comptime and index.is_comptime

    if validated_expr.type.is_array() or validated_expr.type.is_slice():
        return ValidatedIndexExpr([validated_expr, index], parsed_index_expr.span, validated_expr.type.next,
                                  is_comptime, validated_expr.mode), None

    return None, ValidationError(f'cannot index {validated_expr.type}', validated_expr.span)


def validate_slice_expr(scope: Scope, type_hint: CompleteType | None,
                        parsed_slice_expr: ParsedSliceExpression) -> (
        ValidatedSliceExpr | None, ValidationError | None):
    validated_expr, error = validate_expression(scope, None, False, parsed_slice_expr.expr)
    if error: return None, error

    if not (validated_expr.type.is_slice() or validated_expr.type.is_array()):
        return None, ValidationError(f'Expression not sliceable', validated_expr.span)

    if parsed_slice_expr.start:
        start, error = validate_expression(scope, None, False, parsed_slice_expr.start)
        if error: return None, error

        if not start.type.is_integer():
            return None, ValidationError(f'expected integer as index, got {start.type}', parsed_slice_expr.start.span)
    else:
        start = SliceBoundaryPlaceholder(span=parsed_slice_expr.span)

    if parsed_slice_expr.end:
        end, error = validate_expression(scope, None, False, parsed_slice_expr.end)
        if error: return None, error

        if not end.type.is_integer():
            return None, ValidationError(f'expected integer as index, got {end.type}', parsed_slice_expr.end.span)
    else:
        end = SliceBoundaryPlaceholder(span=parsed_slice_expr.span)

    return ValidatedSliceExpr([validated_expr, start, end], parsed_slice_expr.span,
                              CompleteType(Slice(), validated_expr.type.next), validated_expr.is_comptime,
                              ExpressionMode.Value), None


def validate_name_expr(scope: Scope, type_hint: CompleteType | None, parsed_name: ParsedName,
                       args: list[ParsedExpression]) -> (
        ValidatedNameExpr | None, ValidationError | None):
    if var := scope.find_var(parsed_name.value):
        return ValidatedNameExpr([], parsed_name.span, var.type, var.is_comptime, ExpressionMode.Variable,
                                 parsed_name.value), None

    if function := scope.find_function(parsed_name.value):
        return ValidatedNameExpr([], parsed_name.span, function.type(), False, ExpressionMode.Value,
                                 parsed_name.value), None

    if comptime_function_definition := scope.find_comptime_function_definition(parsed_name.value):
        validated_function_definition, error = validate_function_definition(scope, comptime_function_definition,
                                                                            args)
        if error: return None, error
        root_scope = scope.get_root_scope()

        if not root_scope.find_function(validated_function_definition.name().name):
            root_scope.add_function(validated_function_definition)

        return ValidatedNameExpr([], parsed_name.span, validated_function_definition.type(), False,
                                 ExpressionMode.Value, validated_function_definition.name().name), None

    if parsed_name.value == 'typeinfo':
        par_type = CompleteType(NamedType('type'))
        return_type = CompleteType(NamedType('TypeInfo'))
        func_type = CompleteType(FunctionPointer([par_type], return_type))

        # val = scope.get_type_info(value_expr.value.named_type().name)
        # scope.get_root_scope().add_var(Variable(name, typ, val, True))

        return ValidatedNameExpr([], parsed_name.span, func_type, True, ExpressionMode.Value, 'typeinfo'), None

    if parsed_name.value == 'slicelen':
        par_types = [CompleteType(Slice())]
        return_type = CompleteType(NamedType('u32'))
        complete_type = CompleteType(FunctionPointer(par_types, return_type))
        return ValidatedNameExpr([], parsed_name.span, complete_type, True, ExpressionMode.Value,
                                 parsed_name.value), None

    if scope.check_type_exists(CompleteType(NamedType(parsed_name.value))):
        return ValidatedNameExpr([], parsed_name.span, CompleteType(NamedType('type')), True, ExpressionMode.Value,
                                 parsed_name.value), None

    if parsed_name.value in ['true', 'false']:
        return ValidatedNameExpr([], parsed_name.span, CompleteType(NamedType('bool')), True, ExpressionMode.Value,
                                 parsed_name.value), None

    return None, ValidationError(f"Unknown name '{parsed_name.value}'", parsed_name.span)


def validate_primary_expr(scope, type_hint: CompleteType | None, force_evaluation: bool, expr: ParsedPrimaryExpression,
                          args: list[ParsedExpression]):
    if isinstance(expr, ParsedName):
        return validate_name_expr(scope, type_hint, expr, args)
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
        validated_expr, error = validate_expression(scope, None, False, expr)
        if error: return None, error
        validated_exprs.append(validated_expr)

    is_comptime: bool = not any(map(lambda expr: not expr.is_comptime, validated_exprs))

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
                              CompleteType(Array(len(validated_exprs)), next=element_type), is_comptime,
                              ExpressionMode.Value), None


# A:
# Force expression execution if context requires it:
# - expression in type position
# - expression in compile time argument position

# B:
# Remember if expression itself requires compile time execution, but do not execute right away:
# - expression is comptime function call
# It cannot be executed right away, because when expression is at left side of assignment statement,
# we do not actually want read out the value but modify it (not a good explanation, hope you get it anyways).

# Expression has to inform assignment statement, if statement needs to be executed
# for example:
#   a := b
# Assignment statement has to be executed, if "a" is compile time only.

# 1. Force execution by context (validate_expr(..., force_execution : bool, ...))
# 2.
def validate_expression(scope: Scope, type_hint: CompleteType | None, force_evaluation: bool, expr: ParsedExpression,
                        args: list[ParsedExpression] = []) -> (
        ValidatedExpression | None, ValidationError | None):
    """Validate expression

        Arguments:
        is_comptime -- context requires compile evaluation, examples:

        Compile time functions:

        @List() : type {
            ...
        }

        Assignment to compile time variable:
        ...
        @a := something()
        ...

        In types:
        ...
        a := [calculate_array_length()]u8 {}
        b := SomeType(f1(), f2()) {}
        ...
    """

    validated_expr: Optional[ValidatedExpression] = None
    error = None

    if isinstance(expr, ParsedPrimaryExpression):
        validated_expr, error = validate_primary_expr(scope, type_hint, force_evaluation, expr, args)
    elif isinstance(expr, ParsedUnaryOperation):
        validated_expr, error = validate_unop(scope, type_hint, force_evaluation, expr)
    elif isinstance(expr, ParsedBinaryOperation):
        validated_expr, error = validate_binop(scope, type_hint, force_evaluation, expr)

    if error: return None, error

    if force_evaluation:
        return evaluate_expression(scope, validated_expr)

    return validated_expr, None


def evaluate_expression(scope: Scope, validated_expr: ValidatedExpression) -> (
        ValidatedComptimeValueExpr | None, ValidationError | None):
    if not validated_expr.is_comptime:
        return None, ValidationError(f"Compile time evaluation failure of expression {validated_expr}",
                                     validated_expr.span)

    value = do_evaluate_expr(validated_expr, scope)
    return ValidatedComptimeValueExpr([], validated_expr.span, validated_expr.type, True, ExpressionMode.Value, value), None


def try_evaluate_expression_recursively(scope: Scope, validated_expr: ValidatedExpression) -> (ValidatedExpression | None, ValidationError | None):
    if validated_expr.is_comptime:
        return evaluate_expression(scope, validated_expr)

    for i in range(len(validated_expr.children)):
        child = validated_expr.children[i]
        if not isinstance(child, ValidatedExpression):
            continue
        value_expr, error = try_evaluate_expression_recursively(scope, child)
        if error: return None, error
        validated_expr.children[i] = value_expr

    return validated_expr, None


def validate_return_stmt(scope: Scope, parsed_return_stmt: ParsedReturn) -> (
        ValidatedReturnStmt | None, ValidationError | None):
    validated_expr, error = validate_expression(scope, None, False, parsed_return_stmt.expression, [])
    if error: return None, error

    validated_expr, error = try_evaluate_expression_recursively(scope, validated_expr)
    if error: return None, error

    # we do not have 'pure' comptime return statements
    return ValidatedReturnStmt([validated_expr], parsed_return_stmt.span, False), None


def validate_variable_declaration(scope: Scope, parsed_variable_decl: ParsedVariableDeclaration) -> tuple[
    Optional[ValidatedVariableDeclarationStmt], Optional[ValidationError]]:
    # with type expression
    if parsed_variable_decl.type:
        validated_type_expr, error = validate_expression(scope, None, True, parsed_variable_decl.type)
        if error: return None, error

        init_expr, error = validate_expression(scope, validated_type_expr.value, False,
                                               parsed_variable_decl.initializer)
        if error: return None, error

        init_expr, error = try_evaluate_expression_recursively(scope, init_expr)
        if error: return None, error

        if init_expr.type.is_type() and not parsed_variable_decl.is_comptime:
            return None, ValidationError(
                f'Runtime variable cannot hold compile time only values',
                init_expr.span)

        if not validated_type_expr.value.eq_or_other_safely_convertible(init_expr.type):
            return None, ValidationError(
                f'Type mismatch in variable declaration: declaration type = {validated_type_expr.value}, initialization type = {init_expr.type}',
                parsed_variable_decl.span)

        validated_stmt = ValidatedVariableDeclarationStmt([validated_type_expr, init_expr], parsed_variable_decl.span,
                                                          parsed_variable_decl.is_comptime, parsed_variable_decl.name)
    # without type expression
    else:
        init_expr, error = validate_expression(scope, None, False, parsed_variable_decl.initializer)
        if error: return None, error

        init_expr, error = try_evaluate_expression_recursively(scope, init_expr)
        if error: return None, error

        # FIXME: types are compile time only expressions. Maybe adding a field to expression to indicate comptime_only
        #        what be a cleaner way.
        if init_expr.type.is_type() and not parsed_variable_decl.is_comptime:
            return None, ValidationError(
                f'Runtime variable cannot hold compile time only values',
                init_expr.span)

        validated_stmt = ValidatedVariableDeclarationStmt(
            [ValidatedComptimeValueExpr([], init_expr.span, CompleteType(NamedType('type')),
                                        True, ExpressionMode.Value, init_expr.type),
             init_expr], parsed_variable_decl.span,
            parsed_variable_decl.is_comptime, parsed_variable_decl.name)

    init_value = None

    if validated_stmt.is_comptime or validated_stmt.initializer().is_comptime:
        init_value_expr, error = evaluate_expression(scope, validated_stmt.initializer())
        if error: return None, error
        init_value = init_value_expr.value

    scope.add_var(Variable(validated_stmt.name, validated_stmt.type_expr().value, init_value,
                           validated_stmt.is_comptime))

    return validated_stmt, None


def validate_while_stmt(scope: Scope, parsed_while: ParsedWhile) -> tuple[
    Optional[ValidatedWhileStmt], Optional[ValidationError]]:
    condition, error = validate_expression(scope, CompleteType(NamedType('bool')), False, parsed_while.condition)
    if error: return None, error

    if not condition.type.is_bool():
        return None, ValidationError(f'expected boolean expression in while condition', parsed_while.condition.span)

    if condition.is_comptime:
        condition, error = evaluate_expression(scope, condition)
        if error: return None, error

    block, error = validate_block(scope, parsed_while.block, while_block=True)
    if error: return None, error

    # we do not have compile time while statements yet
    return ValidatedWhileStmt([condition, block], parsed_while.span, False), None


def validate_break_stmt(scope: Scope, parsed_break: ParsedBreakStatement) -> tuple[
    Optional[ValidatedBreakStmt], Optional[ValidationError]]:
    if not scope.inside_while_block:
        return None, ValidationError('break statement not in while block', parsed_break.span)

    # we do not have compile time while statements yet
    return ValidatedBreakStmt([], parsed_break.span, False), None


def validate_if_stmt(scope: Scope, parsed_if: ParsedIfStatement) -> (ValidatedIfStmt | None, ValidationError | None):
    condition, error = validate_expression(scope, CompleteType(NamedType('bool')), False, parsed_if.condition)
    if error: return None, error

    if not condition.type.is_bool():
        return None, ValidationError(f'expected boolean expression in if condition', parsed_if.condition.span)

    if condition.is_comptime:
        condition, error = evaluate_expression(scope, condition)
        if error: return None, error

    block, error = validate_block(scope, parsed_if.body)
    if error: return None, error

    return ValidatedIfStmt([condition, block], parsed_if.span, parsed_if.is_comptime), None


def validate_assignment_stmt(scope: Scope, parsed_assignment: ParsedAssignment) -> tuple[
    Optional[ValidatedAssignmentStmt], Optional[ValidationError]]:
    validated_to_expr, error = validate_expression(scope, None, False, parsed_assignment.to)
    if error: return None, error

    if validated_to_expr.mode != ExpressionMode.Variable:
        return None, ValidationError(f'cannot assign to value', parsed_assignment.to.span)

    value_expr, error = validate_expression(scope, None, False, parsed_assignment.value)
    if error: return None, error

    if value_expr.is_comptime:
        value_expr, error = evaluate_expression(scope, value_expr)
        if error: return None, error

    if not validated_to_expr.type.eq_or_other_safely_convertible(value_expr.type):
        return None, ValidationError(
            f'incompatible types in assignment: {validated_to_expr.type.to_string()} and {value_expr.type.to_string()}',
            parsed_assignment.span)

    validated_stmt = ValidatedAssignmentStmt([validated_to_expr, value_expr], parsed_assignment.span,
                                             validated_to_expr.is_comptime)

    if validated_stmt.is_comptime:
        value_expr, error = evaluate_expression(scope, validated_stmt.expr())
        if error: return None, error
        comptime_assign(scope, validated_stmt.to(), value_expr)

    return validated_stmt, None


def comptime_assign(scope: Scope, to_expr: ValidatedExpression, value_expr: ValidatedComptimeValueExpr):
    if isinstance(to_expr, ValidatedNameExpr):
        var = scope.find_var(to_expr.name)
        var.value = value_expr.value
    elif isinstance(to_expr, ValidatedIndexExpr):
        index = do_evaluate_expr(to_expr.index(), scope)
        value = get_comptime_value(scope, to_expr.expr())
        value[index] = value_expr.value
    elif isinstance(to_expr, ValidatedDotExpr):
        value = get_comptime_value(scope, to_expr.expr())
        value[to_expr.name().name] = value_expr.value
    else:
        raise NotImplementedError()


def get_comptime_value(scope: Scope, expr: ValidatedExpression) -> Value:
    if isinstance(expr, ValidatedNameExpr):
        var = scope.find_var(expr.name)
        return var.value
    elif isinstance(expr, ValidatedIndexExpr):
        index = do_evaluate_expr(expr.index(), scope)
        value = get_comptime_value(scope, expr.expr())
        if expr.expr().type.is_array():
            assert (isinstance(value, list))
            assert (index < len(value))
            return value[index]
        if expr.expr().type.is_slice():
            assert (isinstance(value, SliceValue))
            assert (isinstance(value.ptr, list))
            assert (value.start + index < len(value.ptr))
            assert (value.start + index < value.end)
            return value.ptr[value.start + index]
    elif isinstance(expr, ValidatedDotExpr):
        value = get_comptime_value(scope, expr.expr())
        return value[expr.name().name]
    elif isinstance(expr, ValidatedComptimeValueExpr):
        return expr.value
    else:
        raise NotImplementedError(expr)


def validate_block(parent_scope: Scope, block: ParsedBlock, while_block=False) -> (
        ValidatedBlock | None, ValidationError | None):
    scope = parent_scope.add_child_scope('')
    scope.inside_while_block = while_block
    stmts: list[ValidatedStatement] = []

    for stmt in block.statements:
        if isinstance(stmt, ParsedReturn):
            validated_stmt, error = validate_return_stmt(scope, stmt)
            if error: return None, error
        elif isinstance(stmt, ParsedVariableDeclaration):
            validated_stmt, error = validate_variable_declaration(scope, stmt)
            if error: return None, error
        elif isinstance(stmt, ParsedWhile):
            validated_stmt, error = validate_while_stmt(scope, stmt)
            if error: return None, error
        elif isinstance(stmt, ParsedBreakStatement):
            validated_stmt, error = validate_break_stmt(scope, stmt)
            if error: return None, error
        elif isinstance(stmt, ParsedIfStatement):
            validated_stmt, error = validate_if_stmt(scope, stmt)
            if error: return None, error
        elif isinstance(stmt, ParsedAssignment):
            validated_stmt, error = validate_assignment_stmt(scope, stmt)
            if error: return None, error
        elif isinstance(stmt, ParsedExpression):
            validated_expr, error = validate_expression(scope, None, False, stmt)
            if error: return None, error
            validated_stmt = ValidatedExpressionStmt([validated_expr], validated_expr.span, validated_expr.is_comptime)
            if validated_stmt.is_comptime:
                # dropping potential value here
                _, error = evaluate_expression(scope, validated_stmt.expr())
                if error: return None, error
        else:
            raise NotImplementedError(f'validation for statement {stmt} not implemented!')

        stmts.append(validated_stmt)
    return ValidatedBlock(stmts, block.span), None


def validate_function_declaration(scope: Scope,
                                  parsed_function_definition: ParsedFunctionDefinition | ParsedExternFunctionDeclaration) -> (
        ValidatedFunctionDefinition | None, ValidationError | None):
    validated_name, error = validate_name(parsed_function_definition.name)
    if error: return None, error

    validated_pars: list[ValidatedParameter] = []

    child_scope = scope.add_child_scope(validated_name.name)
    # check parameter types
    for par in parsed_function_definition.pars:
        validated_type_expr, error = validate_expression(child_scope, None, True, par.type)
        if error: return None, error

        if not validated_type_expr.type.is_type():
            return None, ValidationError("Expected type", validated_type_expr.span)

        validated_pars.append(
            ValidatedParameter([validated_type_expr], par.span, par.name.value, is_comptime=par.is_comptime))
        child_scope.add_var(
            Variable(validated_pars[-1].name, validated_pars[-1].type_expr().value, None, par.is_comptime))

    # check return type
    validated_return_type_expr, error = validate_expression(child_scope, None, True,
                                                            parsed_function_definition.return_type)
    if error: return None, error

    if not validated_return_type_expr.type.is_type():
        return None, ValidationError("Expected type", validated_return_type_expr.span)

    is_extern = isinstance(parsed_function_definition, ParsedExternFunctionDeclaration)

    return ValidatedFunctionDefinition(
        [validated_name, validated_return_type_expr, *validated_pars], parsed_function_definition.span, True,
        is_extern=is_extern, is_comptime=False), None


def validate_function_definition(scope: Scope, parsed_function_definition: ParsedFunctionDefinition,
                                 bindings: list[ParsedExpression] = []) -> (
        ValidatedFunctionDefinition | None, ValidationError | None):
    validated_name, error = validate_name(parsed_function_definition.name)
    if error: return None, error

    validated_pars: list[ValidatedParameter] = []
    child_scope = scope.add_child_scope(validated_name.name)

    position: int = 0
    name = ''

    # check parameter types
    for par in parsed_function_definition.pars:
        validated_type_expr, error = validate_expression(child_scope, None, True, par.type)

        if error: return None, error

        if not validated_type_expr.type.is_type():
            return None, ValidationError("Expected type", validated_type_expr.span)

        assert validated_type_expr.value is not None

        value = None

        if par.is_comptime:
            if len(bindings) > 0 and position < len(bindings):
                parsed_expr = bindings[position]

                validated_expr, error = validate_expression(child_scope, None, True, parsed_expr)
                if error: return None, error

                if not validated_type_expr.value.eq_or_other_safely_convertible(validated_expr.type):
                    return None, ValidationError("Type mismatch", validated_expr.span)

                value = do_evaluate_expr(validated_expr, child_scope)
            else:
                # FIXME: the span below points to the definition of the comptime function,
                #        but the error occurs at the call sight
                return None, ValidationError("Missing compile time argument for parameter", par.span)

        validated_pars.append(
            ValidatedParameter([validated_type_expr], par.span, par.name.value, is_comptime=par.is_comptime,
                               bound_value=value))
        child_scope.add_var(
            Variable(validated_pars[-1].name, validated_pars[-1].type_expr().value, value, par.is_comptime))

        position += 1

        if value:
            name += str(value)

    if parsed_function_definition.comptime_par_count() > 0 or parsed_function_definition.any_par_count() > 0:
        validated_name.name = validated_name.name + '__' + str(hash(name) + sys.maxsize + 1)

    # check return type
    validated_return_type_expr, error = validate_expression(child_scope, None, True,
                                                            parsed_function_definition.return_type)
    if error: return None, error

    if not validated_return_type_expr.type.is_type():
        return None, ValidationError("Expected type", validated_return_type_expr.span)

    validated_block, error = validate_block(child_scope, parsed_function_definition.body)
    if error: return None, error

    validated_return_stmt: Optional[ValidatedReturnStmt] = None

    # FIXME: Check all branches for return statements
    for validated_stmt in validated_block.statements():
        if isinstance(validated_stmt, ValidatedReturnStmt):
            validated_return_stmt = validated_stmt
            break

    if not validated_return_stmt:
        return None, ValidationError(f'missing return in function {parsed_function_definition.name}',
                                     parsed_function_definition.span)

    if not validated_return_type_expr.value.eq_or_other_safely_convertible(validated_return_stmt.expr().type):
        return None, ValidationError(
            f'Return type mismatch in function "{parsed_function_definition.name.value}": declared return type is {validated_return_type_expr.value}, but returning expression of type {validated_return_stmt.expr().type}',
            parsed_function_definition.span)

    return ValidatedFunctionDefinition(
        [validated_name, validated_return_type_expr,
         validated_block, *validated_pars], parsed_function_definition.span, is_incomplete=False, is_extern=False,
        is_comptime=parsed_function_definition.is_comptime), None


def validate_name(parsed_name: ParsedName) -> (ValidatedName | None, ValidationError | None):
    return ValidatedName([], parsed_name.span, parsed_name.value), None


def validate_struct_field(scope: Scope, parsed_field: ParsedField) -> (ValidatedField | None, ValidationError | None):
    name, error = validate_name(parsed_field.name)
    if error: return None, error

    validated_type_expr, error = validate_expression(scope, None, True, parsed_field.type)
    if error: return None, error
    if not validated_type_expr.type.is_type():
        return None, ValidationError("Expected type", validated_type_expr.span)

    return ValidatedField([name, validated_type_expr], parsed_field.span), None


def validate_struct_pre(scope: Scope, parsed_struct: ParsedStructExpression) -> (
        ValidatedStructPre | None, ValidationError | None):
    name, error = validate_name(parsed_struct.name)
    if error: return None, error
    return ValidatedStructPre([name], parsed_struct.span, name.name, CompleteType(NamedType('type'))), None


def validate_struct_expr(scope: Scope, parsed_struct: ParsedStructExpression) -> (
        ValidatedStructExpr | None, ValidationError | None):
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

    return ValidatedStructExpr([*fields], parsed_struct.span, CompleteType(NamedType('type')), True,
                               ExpressionMode.Value, name), None


def validate_module(module: ParsedModule) -> (ValidatedModule | None, ValidationError | None):
    root_scope = create_root_scope()

    root_scope.add_type_info(TypeInfo)
    root_scope.add_type_info(Field)

    body: list[Union[ValidatedFunctionDefinition, ValidatedStructExpr, ValidatedVariableDeclarationStmt]] = []

    # pre pass 1
    for stmt in module.body:
        if isinstance(stmt, ParsedStructExpression):
            validated_struct_pre, error = validate_struct_pre(root_scope, stmt)
            if error: return None, error
            root_scope.add_type_info(Struct(validated_struct_pre.name, [], ''))

    # pre pass 2
    for stmt in module.body:
        if isinstance(stmt, ParsedFunctionDefinition):
            if stmt.is_comptime or stmt.comptime_par_count() > 0 or stmt.any_par_count() > 0:
                root_scope.add_comptime_function_definition(stmt)
            else:
                validated_function_def_pre, error = validate_function_declaration(root_scope, stmt)
                if error: return None, error
                root_scope.add_function(validated_function_def_pre)
        if isinstance(stmt, ParsedExternFunctionDeclaration):
            validated_function_def_pre, error = validate_function_declaration(root_scope, stmt)
            if error: return None, error
            root_scope.add_function(validated_function_def_pre)

    # main pass 1
    for stmt in module.body:
        if isinstance(stmt, ParsedStructExpression):
            validated_struct_expr, error = validate_expression(root_scope, None, True, stmt)
            if error: return None, error
            body.append(validated_struct_expr)

    # main pass 2
    for stmt in module.body:
        if isinstance(stmt, ParsedFunctionDefinition):
            if not stmt.is_comptime and stmt.comptime_par_count() == 0 and stmt.any_par_count() == 0:
                validated_function_def, error = validate_function_definition(root_scope, stmt)
                if error: return None, error
                body.append(validated_function_def)
                root_scope.add_function(validated_function_def)
        elif isinstance(stmt, ParsedExternFunctionDeclaration):
            validated_function_decl, error = validate_function_declaration(root_scope, stmt)
            if error: return None, error
            body.append(validated_function_decl)
        elif isinstance(stmt, ParsedVariableDeclaration):
            validated_variable_decl, error = validate_variable_declaration(root_scope, stmt)
            if error: return None, error
            body.append(validated_variable_decl)

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
                assert (current.is_named_type())

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


def visit_nodes(node: ValidatedNode, visitor: Callable[[ValidatedNode], bool]):
    if not visitor(node):
        return

    for child in node.children:
        visit_nodes(child, visitor)


def evaluate_block(block: ValidatedBlock, scope) -> (int, Optional[tuple[Value, CompleteType]]):
    BLOCK_EXIT_STATUS_NONE = 0
    BLOCK_EXIT_STATUS_RETURN = 1
    BLOCK_EXIT_STATUS_BREAK = 2

    status = BLOCK_EXIT_STATUS_NONE
    val = None
    typ = None

    for stmt in block.children:
        if isinstance(stmt, ValidatedVariableDeclarationStmt):
            type_value = do_evaluate_expr(stmt.type_expr(), scope)
            init_value = do_evaluate_expr(stmt.initializer(), scope)
            scope.add_var(Variable(stmt.name, type_value, init_value, False))
        elif isinstance(stmt, ValidatedIfStmt):
            cond = do_evaluate_expr(stmt.condition(), scope)
            if cond:
                status, (val, typ) = evaluate_block(stmt.block(), scope)
                if status != BLOCK_EXIT_STATUS_NONE:
                    break
        elif isinstance(stmt, ValidatedReturnStmt):
            status = BLOCK_EXIT_STATUS_RETURN
            val = do_evaluate_expr(stmt.expr(), scope)
            typ = stmt.expr().type
            break
        elif isinstance(stmt, ValidatedBreakStmt):
            status = BLOCK_EXIT_STATUS_BREAK
            break
        elif isinstance(stmt, ValidatedAssignmentStmt):
            var = scope.find_var(stmt.to().name)
            var.value = do_evaluate_expr(stmt.expr(), scope)
        elif isinstance(stmt, ValidatedWhileStmt):
            while do_evaluate_expr(stmt.condition(), scope):
                status, (val, typ) = evaluate_block(stmt.block(), scope)
                if status == BLOCK_EXIT_STATUS_BREAK:
                    break
                if status == BLOCK_EXIT_STATUS_RETURN:
                    break
            if status == BLOCK_EXIT_STATUS_RETURN:
                break
        else:
            raise NotImplementedError(stmt)

    return status, (val, typ)


def do_evaluate_expr(expr: ValidatedExpression, scope: Scope) -> Value:
    assert (scope.parent != scope)

    if isinstance(expr, ValidatedComptimeValueExpr):
        return expr.value

    if isinstance(expr, ValidatedNameExpr):
        if expr.name == 'true':
            return True

        if expr.name == 'false':
            return False

        var = scope.find_var(expr.name)
        if var and var.is_comptime:
            return var.value

        if expr.type.is_type():
            return CompleteType(NamedType(expr.name))

        func = scope.find_function(expr.name)
        if func and func.is_comptime:
            return func

        if expr.name == 'typeinfo':
            pass

        raise NotImplementedError(expr)

    if isinstance(expr, ValidatedUnaryOperationExpr):
        if expr.type.is_type():

            if isinstance(expr.op, Operator) and expr.op != Operator.Multiply:
                raise NotImplementedError()

            if isinstance(expr.op, Operator) and expr.op == Operator.Multiply:
                nested_type = do_evaluate_expr(expr.rhs(), scope)
                return CompleteType(Pointer(), nested_type)

            if isinstance(expr.op, ComplexOperator) and expr.op == ComplexOperator.Array:
                array_length = do_evaluate_expr(expr.children[1], scope)
                nested_type = do_evaluate_expr(expr.rhs(), scope)
                return CompleteType(Array(array_length), nested_type)

            if isinstance(expr.op, ComplexOperator) and expr.op == ComplexOperator.Slice:
                nested_type = do_evaluate_expr(expr.rhs(), scope)
                return CompleteType(Slice(), nested_type)

            raise NotImplementedError()

        elif isinstance(expr.op, Operator):
            if expr.op == Operator.Plus:
                return do_evaluate_expr(expr.rhs(), scope)
            if expr.op == Operator.Minus:
                return -do_evaluate_expr(expr.rhs(), scope)

        raise NotImplementedError()

    if isinstance(expr, ValidatedBinaryOperationExpr):
        if expr.type.is_integer() or expr.type.is_floating_point() or expr.type.is_bool():
            if expr.op == Operator.Plus:
                return do_evaluate_expr(expr.lhs(), scope) + do_evaluate_expr(expr.rhs(), scope)
            if expr.op == Operator.Minus:
                return do_evaluate_expr(expr.lhs(), scope) - do_evaluate_expr(expr.rhs(), scope)
            if expr.op == Operator.Multiply:
                return do_evaluate_expr(expr.lhs(), scope) * do_evaluate_expr(expr.rhs(), scope)
            if expr.op == Operator.Divide:
                return do_evaluate_expr(expr.lhs(), scope) / do_evaluate_expr(expr.rhs(), scope)
            if expr.op == Operator.LessThan:
                return do_evaluate_expr(expr.lhs(), scope) <= do_evaluate_expr(expr.rhs(), scope)
            if expr.op == Operator.Equals:
                return do_evaluate_expr(expr.lhs(), scope) == do_evaluate_expr(expr.rhs(), scope)
            if expr.op == Operator.And:
                return do_evaluate_expr(expr.lhs(), scope) and do_evaluate_expr(expr.rhs(), scope)

        raise NotImplementedError(expr)

    if isinstance(expr, ValidatedCallExpr):

        function_definition = scope.find_function(expr.name)
        execution_scope = scope.get_root_scope().add_child_scope('call')

        for par in function_definition.pars():
            if not par.is_comptime:
                raise "Unbound parameter"
            var = Variable(par.name, par.type_expr().value, par.bound_value, True)
            execution_scope.add_var(var)

        _, (return_value, typ) = evaluate_block(function_definition.body(), execution_scope)
        return return_value

    if isinstance(expr, ValidatedTypeInfoCallExpr):
        value = do_evaluate_expr(expr.args()[0], scope)
        struct = scope.get_type_info(value.named_type().name)
        fields = list(map(lambda field: {"name": str_to_slicevalue(field.name)}, struct.fields))
        return {"name": str_to_slicevalue(struct.name), "fields": SliceValue(0, len(fields), fields)}

    if isinstance(expr, ValidatedStructExpr):
        fields = []

        signature = ''

        for field in expr.fields():
            field_type = do_evaluate_expr(field.type_expr(), scope)
            fields.append(Struct.StructField(field.name().name, field_type))
            signature += f"{field.name().name}_{field_type.to_string()}__"

        signature = hash(signature) + sys.maxsize + 1  # we want positive numbers

        name = expr.name
        if name is None:
            name = '___anonymous_struct__' + str(signature)

        struct = Struct(name, fields, '')
        scope.get_root_scope().add_type_info(struct)
        return CompleteType(NamedType(struct.name))

    if isinstance(expr, ValidatedArrayExpr):
        return list(map(lambda expr: do_evaluate_expr(expr, scope), expr.expressions()))

    if isinstance(expr, ValidatedDotExpr):
        return get_comptime_value(scope, expr)

    if isinstance(expr, ValidatedInitializerExpr):
        def comptime_initialize_value(typ: CompleteType) -> Value:
            if typ.is_integer():
                return 0
            if typ.is_bool():
                return False
            if typ.is_floating_point():
                return 0.0
            if not typ.is_builtin() and typ.is_named_type():
                dict_ = dict()
                type_info = scope.get_type_info(typ.named_type().name)
                for field in type_info.fields:
                    dict_[field.name] = comptime_initialize_value(field.type)
                return dict_
            if typ.is_array():
                array = []
                for _ in range(typ.array().length):
                    array.append(comptime_initialize_value(typ.next))
                return array
            if typ.is_slice():
                return SliceValue(0, 0, [])
            raise NotImplementedError(typ)

        return comptime_initialize_value(expr.type)

    if isinstance(expr, ValidatedIndexExpr):
        return get_comptime_value(scope, expr)

    if isinstance(expr, ValidatedSliceExpr):

        if expr.src().type.is_array():
            array = do_evaluate_expr(expr.src(), scope)

            if isinstance(expr.start(), SliceBoundaryPlaceholder):
                start = 0
            else:
                start = do_evaluate_expr(expr.start(), scope)

            if isinstance(expr.end(), SliceBoundaryPlaceholder):
                end = len(array)
            else:
                end = do_evaluate_expr(expr.end(), scope)

            return SliceValue(start, end, array)

        if expr.src().type.is_slice():
            slice_value = do_evaluate_expr(expr.src(), scope)

            if isinstance(expr.start(), SliceBoundaryPlaceholder):
                start = 0
            else:
                start = do_evaluate_expr(expr.start(), scope)

            start += slice_value.start

            if isinstance(expr.end(), SliceBoundaryPlaceholder):
                end = len(slice_value.ptr)
            else:
                end = slice_value.start + do_evaluate_expr(expr.end(), scope)

            end = min(end, slice_value.end)

            return SliceValue(start, end, slice_value.ptr)

        raise NotImplementedError()

    raise NotImplementedError()


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
