from collections import defaultdict
from typing import Union, Optional, ClassVar, Callable
from lexer import Span
from dataclasses import dataclass, field
from parser import ParsedModule, ParsedFunctionDefinition, ParsedExpression, ParsedName, ParsedBinaryOperation, \
    ParsedUnaryOperation, ParsedBlock, ParsedReturn, Operator, ParsedNumber, ParsedCall, ParsedOperator, \
    ParsedVariableDeclaration, ParsedWhile, ParsedBreakStatement, ParsedIfStatement, ParsedStruct, \
    ParsedField, ParsedType, ParsedTypeLiteral, ParsedPointerType, ParsedSliceType, ParsedArrayType, \
    ParsedStructExpression, ParsedDotExpression, ParsedPrimaryExpression, ParsedIndexExpression,\
    ParsedArray, ParsedExternFunctionDeclaration, print_parser_error, TokenSource, parse_module

from lexer import lex

builtin_types = ['i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64', 'bool']


@dataclass
class Variable:
    name: str
    type: 'CompleteType'
    scope : 'Scope' = None


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
class Namespace:
    name : str


@dataclass
class FunctionPointer:
    pars: list['CompleteType']
    ret: 'CompleteType'


@dataclass
class StructField:
    name: str
    type: 'CompleteType'


@dataclass
class Struct:
    name : str
    fields : list[StructField]
    location : str

    def try_get_field(self, name: str) -> StructField | None:
        for field in self.fields:
            if field.name == name:
                return field
        return None

    def fully_qualified_name(self):
        return self.location + '.' + self.name


@dataclass
class Builtin:
    name: str


@dataclass
class CompleteType:
    """ Complete types still require a scope context to resolve the fundamental type names.
        The complete type holds the Type that can come out of an expression.
        It's currently undecided what a Namespace should become, for now these are equivalent
        to builtin or potentially nested declared types (= fundamental types).
    """

    HoldingTypes = Union[Pointer, Array, Slice, FunctionPointer, Namespace, Struct, Builtin]

    val: HoldingTypes
    next: Optional['CompleteType'] = None

    def is_pointer(self) -> bool:
        return isinstance(self.val, Pointer)

    def is_array(self) -> bool:
        return isinstance(self.val, Array)

    def is_slice(self) -> bool:
        return isinstance(self.val, Slice)

    def is_builtin(self):
        return isinstance(self.val, Builtin)

    def is_struct(self):
        return isinstance(self.val, Struct)

    def is_integer(self) -> bool:
        return self.is_builtin() and self.builtin().name in ['i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64']

    def is_function_ptr(self) -> bool:
        return isinstance(self.val, FunctionPointer)

    def is_namespace(self) -> bool:
        return isinstance(self.val, Namespace)

    def get(self) -> HoldingTypes:
        return self.val

    def array(self) -> Array:
        assert(self.is_array())
        return self.get()

    def function_ptr(self) -> FunctionPointer:
        assert(self.is_function_ptr())
        return self.get()

    def builtin(self) -> Builtin:
        assert(self.is_builtin())
        return self.get()

    def struct(self) -> Struct:
        assert(self.is_struct())
        return self.get()

    def namespace(self) -> Namespace:
        assert(self.is_namespace())
        return self.get()

    def to_string(self) -> str:
        if self.is_pointer():
            return '*' + self.next.to_string()
        elif self.is_array():
            return f'[{self.array().length}]' + self.next.to_string()
        elif self.is_slice():
            return '[]' + self.next.to_string()
        elif self.is_struct():
            return self.struct().fully_qualified_name()
        elif self.is_builtin():
            return self.builtin().name
        elif self.is_function_ptr():
            function_ptr = self.function_ptr()
            pars = ','.join([par.to_string() for par in function_ptr.pars])
            return f'@function:({pars})->({function_ptr.ret.to_string()})'
        elif self.is_namespace():
            return f'@namespace:{self.namespace().name}'
        else:
            raise NotImplemented()

    def collect_downstream_types(self, bag: dict[str, 'CompleteType']):
        if self.to_string() in bag:
            return

        if self.is_pointer():
            bag[self.to_string()] = self
            return self.next.collect_downstream_types(bag)
        elif self.is_array():
            bag[self.to_string()] = self
            return self.next.collect_downstream_types(bag)
        elif self.is_slice():
            bag[self.to_string()] = self
            return self.next.collect_downstream_types(bag)
        elif self.is_struct():
            bag[self.to_string()] = self
            for field in self.struct().fields:
                field.type.collect_downstream_types(bag)
            return
        elif self.is_builtin():
            bag[self.to_string()] = self
            return
        elif self.is_function_ptr():
            bag[self.to_string()] = self
            for par in self.function_ptr().pars:
                par.collect_downstream_types(bag)
            self.function_ptr().ret.collect_downstream_types(bag)
            return
        elif self.is_namespace():
            pass
        else:
            raise NotImplemented()

    def eq_or_other_safely_convertible(self, other : 'CompleteType') -> bool:
        if self.is_integer() and other.is_integer():
            return check_integer_type_compatibility(self.builtin().name, other.builtin().name)

        return self.eq(other)

    def eq(self, other: 'CompleteType') -> bool:

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

        if self.is_struct() and other.is_struct():
            return self.struct().name == other.struct().name and self.struct().location == other.struct().location

        if self.is_builtin() and other.is_builtin():
            return self.builtin().name == other.builtin().name

        return False


def is_unary_operator_defined(complete_type: CompleteType, op: Operator) -> (CompleteType | None):
    if not complete_type.is_builtin() or complete_type.builtin() == 'bool':
        return None
    else:
        match op:
            case Operator.Plus:
                return complete_type
            case Operator.Minus:
                if complete_type.builtin().name == 'u64':
                    return None
                if complete_type.builtin().name in ['u8', 'u16', 'u32']:
                    conversion_table = {}
                    conversion_table['u8'] = 'i16'
                    conversion_table['u16'] = 'i32'
                    conversion_table['u32'] = 'i64'
                    return CompleteType(Builtin(conversion_table[complete_type.builtin().name]))

                return complete_type
            case _:
                return None


def is_binary_operator_defined(lhs: CompleteType, rhs: CompleteType, op: Operator) -> (CompleteType | None):
    if not lhs.is_builtin() or lhs.builtin() == 'bool':
        return None
    else :
        match op:
            case Operator.Minus | Operator.Plus | Operator.Minus | Operator.Multiply | Operator.Divide:
                return lhs
            case Operator.Equals:
                return CompleteType(Builtin('bool'))
            case _:
                return None

scope_number : int = 0

@dataclass
class Scope:
    namespace: str = ''
    functions: list[Function] = field(default_factory=list)
    variables: list[Variable] = field(default_factory=list)
    structs: list[Struct] = field(default_factory=list)
    children: list['Scope'] = field(default_factory=list)
    parent: 'Scope' = None
    inside_while_block: bool = False
    scope_number : int = 0

    scope_cnt : ClassVar[int] = 0

    def __init__(self, namespace: str = '', parent: 'Scope' = None):
        self.functions = []
        self.variables = []
        self.children = []
        self.structs = []
        self.parent = parent
        self.namespace = namespace
        self.scope_number = self.scope_cnt
        self.scope_cnt += 1

    def get_scope_id(self) -> str:
        scope_id = self.namespace if self.namespace != '' else f'anonymous{self.scope_number}'
        if self.parent:
            return self.parent.get_scope_id() + '.' + scope_id
        else: return scope_id

    def add_var(self, what: Union['ValidatedVariableDeclaration', 'ValidatedParameter']):
        var = Variable(what.name, what.type)
        self.variables.append(var)

    def update_struct_fields(self, validated_struct: 'ValidatedStruct') -> bool:
        for struct in self.structs:
            if isinstance(struct, Struct) and struct.name == validated_struct.name().name:
                struct.fields = [StructField(field.name().name, field.type) for field in validated_struct.fields()]
                return True

        return False

    def add_struct_pre(self, validated_struct_pre: 'ValidatedStructPre'):
        self.structs.append(validated_struct_pre.type.struct())

    def add_child_scope(self, name: str) -> 'Scope':
        scope = Scope(namespace=name)
        scope.parent = self
        self.children.append(scope)
        return scope

    def get_child_scope(self, name: str) -> Optional['Scope']:
        for scope in self.children:
            if scope.namespace == name:
                return scope

        return None

    def find_namespace(self, name : str) -> Optional['Scope']:
        path_rev = name.split('.')[::-1]
        return self.find_namespace_path(path_rev)

    def find_function(self, name: str) -> Function | None:
        path = name.split('.')

        if len(path) > 1:
            if scope := self.find_namespace_path(path[:-1][::-1]):  # trim off function name and then reverse
                for function in scope.functions:
                    if function.name == path[-1]:
                        return function
                else:
                    return None
            else:
                return None
        else:
            # search up for functions
            current = self

            while current != None:
                for function in current.functions:
                    if function.name == name:
                        return function

                current = current.parent

            return None

    def find_struct(self, name: str) -> Struct | None:
        path = name.split('.')

        if len(path) > 1:
            if scope := self.find_namespace_path(path[:-1][::-1]):  # trim off type name and then reverse
                for type in scope.structs:
                    if type.name == path[-1]:
                        return type
                else:
                    return None
            else:
                return None

        else:
            # search up for type
            current = self

            while current != None:
                for type in current.structs:
                    if type.name == name:
                        return type

                current = current.parent
            return None

    def find_namespace_in_scope(self, node: 'Scope', namespace: str) -> Optional['Scope']:
        if not node: return None

        for child in node.children:
            if child.namespace == namespace:
                return child

        return self.find_namespace_in_scope(node.parent, namespace)

    def find_namespace_path(self, path_rev: list[str]) -> Optional['Scope']:
        # First find the starting node, the node where the namespace path begins.
        if not (node := self.find_namespace_in_scope(self, path_rev[-1])):
            return None

        path_rev.pop()

        # Then verify if there is such a path starting from the starting node.
        while len(path_rev) > 0:
            for child in node.children:
                if child.namespace == path_rev[-1]:
                    node = child
                    path_rev.pop()
                    break
            else:
                return None

        return node

    def find_var(self, name) -> Variable | None:
        if result := self._resolve_var(name.split('.')[::-1]):
            var, _ = result
            return var

        return None

    def _resolve_var(self, path_rev: list[str]) -> (tuple[Variable, list[str]] | None):
        node = self
        allow_find_in_parent = True

        def find_namespace(n: 'Scope', namespace: str) -> Optional['Scope']:
            if not n: return None

            for child in n.children:
                if child.namespace == namespace:
                    return child

            return find_namespace(n.parent, namespace)

        node = find_namespace(node, path_rev[-1])

        if node:
            allow_find_in_parent = False
            # NS.NS.var.field.field
            # pop namespaces
            while len(path_rev) > 1 and node:
                path_rev.pop()

                for child in node.children:
                    if child.namespace == path_rev[-1]:
                        node = child
                        break  # for
                else:
                    break  # while
        else:
            node = self

        if len(path_rev) == 0:
            return None

        # var.field.field

        # At this point, the node is either initial self node or the one we found be removing the namespaces
        # from the variable path. In the first case we are still allowed to search the variable in the parent scopes of node,
        # in the second case the variable has to be in the scope of node.
        while node:
            # Find the variable in the current node.
            for var in node.variables:
                if var.name == path_rev[-1]:
                    return var, path_rev[::-1]

            # Variable not found in current node.
            if allow_find_in_parent:
                node = node.parent
            else:
                return None

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
    children : list['ValidatedNode']  # empty
    span: Span
    name: str


@dataclass
class ValidatedNameExpr:
    children : list['ValidatedNode']  # empty
    span: Span
    name: str
    type: 'CompleteType'


@dataclass
class ValidatedNumber:
    children : list['ValidatedNode']  # empty
    span: Span
    value: str
    type: 'CompleteType'


@dataclass
class ValidatedString:
    children : list['ValidatedNode']  # empty
    span: Span
    value: str
    type: 'CompleteType'


@dataclass
class ValidatedUnaryOperation:
    children : list['ValidatedNode']
    span: Span
    op: ParsedOperator
    type: 'CompleteType'

    def rhs(self) -> 'ValidatedExpression': return self.children[0]


@dataclass
class ValidatedBinaryOperation:
    children : list['ValidatedNode']
    span: Span
    op: ParsedOperator
    type: 'CompleteType'

    def lhs(self) -> 'ValidatedExpression': return self.children[0]
    def rhs(self) -> 'ValidatedExpression': return self.children[1]


@dataclass
class ValidatedCall:
    children : list['ValidatedNode']
    span: Span
    type: 'CompleteType'

    def expr(self) -> 'ValidatedExpression': return self.children[0]
    def args(self) -> list['ValidatedExpression']: return self.children[1:]


@dataclass
class ValidatedStructExpression:
    children : list['ValidatedNode']
    span: Span
    type: 'CompleteType'

    def expr(self) -> 'ValidatedExpression': return self.children[0]


@dataclass
class ValidatedDotExpression:
    children : list['ValidatedNode']
    span: Span
    type: 'CompleteType'
    auto_deref: bool

    def expr(self) -> 'ValidatedExpression': return self.children[0]
    def name(self) -> 'ValidatedName': return self.children[1]


@dataclass
class ValidatedIndexExpression:
    children : list['ValidatedNode']
    span: Span
    type: 'CompleteType'

    def expr(self) -> 'ValidatedExpression': return self.children[0]
    def index(self) -> 'ValidatedExpression': return self.children[1]


ValidatedExpression = Union[
    ValidatedNumber, ValidatedNameExpr, ValidatedUnaryOperation, ValidatedBinaryOperation, ValidatedCall, ValidatedDotExpression, ValidatedStructExpression, ValidatedIndexExpression, 'ValidatedArray']


@dataclass
class ValidatedParameter:
    children : list['ValidatedNode']  # empty
    span: Span
    name: str
    type: 'CompleteType'


@dataclass
class ValidatedReturnType:
    children: list['ValidatedNode']
    span: Span
    type: 'CompleteType'


@dataclass
class ValidatedBlock:
    children: list['ValidatedStatement']
    span: Span

    def statements(self): return self.children


@dataclass
class ValidatedFunctionDefinitionPre:
    children : list['ValidatedNode']
    span: Span

    def name(self) -> ValidatedName: return self.children[0]
    def return_type(self) -> ValidatedReturnType: return self.children[1]
    def pars(self) -> list['ValidatedParameter']: return self.children[2:]


@dataclass
class ValidatedFunctionDefinition:
    children: list['ValidatedNode']
    span: Span

    def name(self) -> ValidatedName: return self.children[0]
    def return_type(self) -> ValidatedReturnType: return self.children[1]
    def body(self) -> ValidatedBlock: return self.children[2]
    def pars(self) -> list['ValidatedParameter']: return self.children[3:]


@dataclass
class ValidatedExternFunctionDeclaration:
    children : list['ValidatedNode']
    span: Span

    def name(self) -> ValidatedName: return self.children[0]
    def return_type(self) -> ValidatedReturnType: return self.children[1]
    def pars(self) -> list['ValidatedParameter']: return self.children[2:]


OtherValidatedNodes = Union['ValidatedModule', ValidatedReturnType, ValidatedName, ValidatedBlock, ValidatedParameter, 'ValidatedField']

@dataclass
class ValidatedReturnStatement:
    children: list['ValidatedNode']
    span: Span

    def expr(self): return self.children[0]


@dataclass
class ValidatedVariableDeclaration:
    children: list['ValidatedNode']
    span: Span
    name: str
    type: CompleteType

    def expr(self) -> ValidatedExpression: return self.children[0]


@dataclass
class ValidatedWhileStatement:
    children: list['ValidatedNode']
    span: Span

    def condition(self) -> ValidatedExpression: return self.children[0]
    def block(self) -> ValidatedBlock: return self.children[1]


@dataclass
class ValidatedBreakStatement:
    children: list['ValidatedNode']
    span: Span


@dataclass
class ValidatedIfStatement:
    children: list['ValidatedNode']
    span: Span

    def condition(self) -> ValidatedExpression: return self.children[0]
    def block(self) -> ValidatedBlock: return self.children[1]


@dataclass
class ValidatedField:
    children: list['ValidatedNode']
    span: Span
    type: CompleteType

    def name(self) -> ValidatedName: return self.children[0]


@dataclass
class ValidatedStruct:
    children: list['ValidatedNode']
    span: Span
    type : CompleteType

    def name(self) -> ValidatedName: return self.children[0]
    def fields(self) -> list[ValidatedField]: return self.children[1:]


@dataclass
class ValidatedStructPre:
    children: list['ValidatedNode']
    span: Span
    type : CompleteType  # incomplete type, does not contain the fields

    def name(self) -> ValidatedName: return self.children[0]


ValidatedStatement = Union[
    ValidatedFunctionDefinition, ValidatedReturnStatement, ValidatedVariableDeclaration, ValidatedWhileStatement, ValidatedBreakStatement, ValidatedExpression, ValidatedIfStatement]

ValidatedNode = Union[ValidatedStatement, ValidatedStruct, OtherValidatedNodes]


@dataclass
class ValidatedModule:
    children: list['ValidatedNode']
    span: Span

    structs_in_topological_order : list[Struct]
    scope : Scope

    def body(self) -> list[Union[ValidatedFunctionDefinition]]: return self.children


def check_integer_type_compatibility(nominal : str, other : str) -> bool:
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


def integer_literal_too_type(literal : str) -> str | None:
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


def validate_number(type_hint: CompleteType | None, parsed_number: ParsedNumber) -> (ValidatedNumber | None, ValidationError | None):
    if '.' in parsed_number.value:
        return ValidatedNumber([], parsed_number.span, parsed_number.value, CompleteType(Builtin('f64'))), None
    else:
        if builtin_name := integer_literal_too_type(parsed_number.value):
            return ValidatedNumber([], parsed_number.span, parsed_number.value, CompleteType(Builtin(builtin_name))), None
        else:
            return None, ValidationError(f'integer number {parsed_number.value} too large', parsed_number.span)


def validate_string(type_hint: CompleteType | None, parsed_number: ParsedNumber) -> (ValidatedNumber | None, ValidationError | None):
    return ValidatedString([], parsed_number.span, parsed_number.value, CompleteType(Builtin(builtin_name))), None


def validate_unop(scope: Scope, _: CompleteType | None, parsed_unop: ParsedUnaryOperation) -> (ValidatedUnaryOperation | None, ValidationError | None):
    val_expr, error = validate_expression(scope, None, parsed_unop.rhs)
    if error: return None, error

    op = parsed_unop.op.op

    if op == Operator.And:
        return ValidatedUnaryOperation([val_expr], parsed_unop.span, parsed_unop.op, CompleteType(Pointer(), val_expr.type)), None

    if op == Operator.Multiply:
        if not val_expr.type.is_pointer():
            return None, ValidationError(f'cannot dereference type {val_expr.type}', parsed_unop.span)
        return ValidatedUnaryOperation([val_expr], parsed_unop.span, parsed_unop.op, val_expr.type.next), None

    if not (val_expr.type.is_struct() or val_expr.type.is_builtin()):
        return None, ValidationError(f'type {val_expr.type.get()} does not support unary operation with operator {op}', parsed_unop.span)

    if val_expr.type.is_struct():
        # TODO: Do we still need to check this here? I feel like if we have a CompleteType that
        #       that is known to be a struct, then it should have already be validated.
        if not scope.find_struct(val_expr.type.struct().name):
            raise NotImplementedError('Leaving this here to confirm my hypothesis above.')

    if isinstance(val_expr, ValidatedNumber) and op == Operator.Minus:
        # The parser currently can only produce positive numbers. Negative numbers will be parsed as unary operation.
        # This case is handled separately to be able to apply the knowledge of the size of the number at compile time
        # to produce the best type, for example:
        # A '-3' is parsed as -( u8 ) and becomes of type i16. To avoid "oversizing" (i16 instead of i8) we can apply
        # the knowledge that the u8 is 3, and hence -3 also fits into i8.
        integer_type_name_after_unop = integer_literal_too_type(f'-{val_expr.value}')

        if not integer_type_name_after_unop:
            return None, ValidationError(f'type {val_expr.type} does not support unary operation with operator {op}', parsed_unop.span)
        else:
            type_after_unop = CompleteType(Builtin(integer_type_name_after_unop))

    elif not (type_after_unop := is_unary_operator_defined(val_expr.type, op)):
        return None, ValidationError(f'type {val_expr.type} does not support unary operation with operator {op}', parsed_unop.span)

    return ValidatedUnaryOperation([val_expr], parsed_unop.span, parsed_unop.op, type_after_unop), None


def validate_binop(scope: Scope, type_hint: CompleteType | None, parsed_binop: ParsedBinaryOperation) -> (ValidatedBinaryOperation | None, ValidationError | None):
    lhs, error = validate_expression(scope, None, parsed_binop.lhs)
    if error: return None, error

    rhs, error = validate_expression(scope, None, parsed_binop.rhs)
    if error: return None, error

    op = parsed_binop.op.op

    if not(type_after_binop := is_binary_operator_defined(lhs.type, rhs.type, op)):
        return None, ValidationError(
            f'type {lhs.type.get()} does no support binary operation with operator {op} and other type {rhs.type.get()}', parsed_binop.span)

    return ValidatedBinaryOperation([lhs, rhs], parsed_binop.span, parsed_binop.op, type_after_binop), None


def validate_call(scope: Scope, type_hint: CompleteType | None, parsed_call: ParsedCall) -> (ValidatedCall | None, ValidationError | None):
    validated_expr, error = validate_expression(scope, None, parsed_call.expr)
    if error: return None, error

    if not validated_expr.type.is_function_ptr():
        return None, ValidationError(f'expression type {validated_expr.type.get()} not a function pointer', validated_expr.span)

    validated_args: list[ValidatedExpression] = []

    for arg in parsed_call.args:
        expr, error = validate_expression(scope, None, arg)
        if error: return None, error
        validated_args.append(expr)

    function_ptr : FunctionPointer = validated_expr.type.get()

    if len(function_ptr.pars) != len(validated_args):
        return None, ValidationError(f'Wrong number of arguments in call to function', parsed_call.span)

    for idx, (a, b) in enumerate(zip(function_ptr.pars, validated_args)):
        if not a.eq_or_other_safely_convertible(b.type):
            return None, ValidationError(f'Type mismatch in {idx + 1}th argument in call to function', parsed_call.span)

    return ValidatedCall([validated_expr, *validated_args], parsed_call.span, function_ptr.ret), None


def validate_struct_expression(scope: Scope,type_hint: CompleteType | None, parsed_struct_expression: ParsedStructExpression) -> (ValidatedStructExpression | None, ValidationError | None):
    validated_expr, error = validate_expression(scope, None, parsed_struct_expression.expr)
    if error: return None, error

    if not validated_expr.type.is_namespace():
        return None, ValidationError(f'expression {validated_expr} does not evaluate to namespace/type', validated_expr.span)

    if not (struct := scope.find_struct(validated_expr.type.namespace().name)):
        return None, ValidationError(f'type {validated_expr.type.namespace().name} not found in scope {scope.namespace}', validated_expr.span)

    return ValidatedStructExpression([validated_expr], parsed_struct_expression.span, CompleteType(struct)), None


def validate_dot_expr(scope: Scope, type_hint: CompleteType | None, parsed_dot_expr: ParsedDotExpression):

    validated_expr, error = validate_expression(scope, None, parsed_dot_expr.expr)
    if error: return None, error

    validated_name, error = validate_name(parsed_dot_expr.name)
    if error: return None, error

    if validated_expr.type.is_namespace():
        namespace = validated_expr.type.namespace().name + '.' + validated_name.name
        if scope.find_namespace(namespace):
            return ValidatedDotExpression([validated_expr, validated_name], parsed_dot_expr.span, CompleteType(Namespace(namespace)), False), None
        else:
            return None, ValidationError(f'namespace {namespace} not found from scope {scope.namespace}', parsed_dot_expr.span)

    dot_into = None

    auto_deref = False

    # pointers should always have a next type
    if validated_expr.type.is_pointer() and validated_expr.type.next.is_struct():
        auto_deref = True
        dot_into = validated_expr.type.next

    if validated_expr.type.is_struct():
        dot_into = validated_expr.type

    if not dot_into:
        return None, ValidationError(f'cannot dot into type {validated_expr.type}', parsed_dot_expr.span)

    if field := dot_into.struct().try_get_field(validated_name.name):
        return ValidatedDotExpression([validated_expr, validated_name], parsed_dot_expr.span, field.type, auto_deref), None

    return None, ValidationError(f'field {validated_name.name} not found', parsed_dot_expr.span)


def validate_index_expr(scope : Scope, type_hint: CompleteType | None, parsed_index_expr : ParsedIndexExpression) -> (ValidatedIndexExpression | None, ValidationError | None):
    validated_expr, error = validate_expression(scope, None, parsed_index_expr.expr)
    if error: return None, error

    index, error = validate_expression(scope, None, parsed_index_expr.index)
    if error: return None, error

    if not index.type.is_integer():
        return None, ValidationError(f'expected integer as index, got {index.type}', parsed_index_expr.index.span)

    if validated_expr.type.is_array() or validated_expr.type.is_slice():
        return ValidatedIndexExpression([validated_expr, index], parsed_index_expr.span, validated_expr.type.next), None

    return None, ValidationError(f'cannot index {validated_expr.type}', validated_expr.span)


def validate_name_expr(scope: Scope, type_hint: CompleteType | None, parsed_name: ParsedName) -> (ValidatedNameExpr | None, ValidationError | None):

    if var := scope.find_var(parsed_name.value):
        return ValidatedNameExpr([], parsed_name.span, parsed_name.value, var.type), None

    if scope.find_namespace(parsed_name.value):
        return ValidatedNameExpr([], parsed_name.span, parsed_name.value, CompleteType(Namespace(parsed_name.value))), None

    if scope.find_struct(parsed_name.value):
        return ValidatedNameExpr([], parsed_name.span, parsed_name.value, CompleteType(Namespace(parsed_name.value))), None

    if parsed_name.value in builtin_types:
        return ValidatedNameExpr([], parsed_name.span, parsed_name.value, CompleteType(Namespace(parsed_name.value))), None

    if function := scope.find_function(parsed_name.value):
        return ValidatedNameExpr([], parsed_name.span,parsed_name.value, function.type), None

    if parsed_name.value in ['true', 'false']:
        return ValidatedNameExpr([], parsed_name.span, parsed_name.value, CompleteType(Builtin('bool'))), None

    return None, ValidationError(f'name {parsed_name.value} not found in scope {scope.namespace}', parsed_name.span)


def validate_primary_expr(scope, type_hint: CompleteType | None, expr : ParsedPrimaryExpression):

    if isinstance(expr, ParsedName):
        return validate_name_expr(scope, type_hint, expr)
    if isinstance(expr, ParsedNumber):
        return validate_number(type_hint, expr)
    if isinstance(expr, ParsedCall):
        return validate_call(scope, type_hint, expr)
    if isinstance(expr, ParsedStructExpression):
        return validate_struct_expression(scope, type_hint, expr)
    if isinstance(expr, ParsedDotExpression):
        return validate_dot_expr(scope, type_hint, expr)
    if isinstance(expr, ParsedIndexExpression):
        return validate_index_expr(scope, type_hint, expr)
    if isinstance(expr, ParsedArray):
        return validate_array(scope, type_hint, expr)

    raise NotImplementedError(expr)


@dataclass
class ValidatedArray:
    children : list[ValidatedExpression]
    span : Span
    type : CompleteType

    def expressions(self): return self.children


def validate_array(scope, type_hint: CompleteType | None, array: ParsedArray) -> (ValidatedArray | None, ValidationError | None):

    validated_exprs : list[ValidatedExpression] = []
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

            integer_name = expr.type.builtin().name
            integer_candidates = integer_candidates.intersection(integer_compatibility[integer_name])

        if len(integer_candidates) == 0:
            return None, ValidationError('unable to find a common integer type for expressions array', array.span)

        # Choose the hinted type if provided and possible.
        if type_hint and type_hint.is_array() and type_hint.next.is_integer() and type_hint.next.builtin().name in integer_candidates:
            chosen_name = type_hint.next.builtin().name
        else:
            # We might have multiple candidates left, pick in order of 'all_integer_types'.
            chosen_name = 'i64'

            for name in all_integer_types:
                if name in integer_candidates:
                    chosen_name = name
                    break

        element_type = CompleteType(Builtin(chosen_name))

    return ValidatedArray(validated_exprs, array.span, CompleteType(Array(len(validated_exprs)), next=element_type)), None


def validate_expression(scope: Scope, type_hint: CompleteType | None, expr: ParsedExpression) -> (ValidatedExpression | None, ValidationError | None):
    if isinstance(expr, ParsedUnaryOperation):
        return validate_unop(scope, type_hint, expr)
    if isinstance(expr, ParsedBinaryOperation):
        return validate_binop(scope, type_hint, expr)
    if isinstance(expr, ParsedPrimaryExpression):
        return validate_primary_expr(scope, type_hint, expr)

    raise NotImplementedError(expr)


def validate_return_stmt(scope: Scope, parsed_return_stmt: ParsedReturn) -> (ValidatedReturnStatement | None, ValidationError | None):
    validated_expr, error = validate_expression(scope, None, parsed_return_stmt.expression)
    if error: return None, error
    return ValidatedReturnStatement([validated_expr], parsed_return_stmt.span), None


def validate_variable_declaration(scope: Scope, parsed_variable_decl: ParsedVariableDeclaration) -> tuple[Optional[ValidatedVariableDeclaration], Optional[ValidationError]]:
    validated_type, error = validate_type(scope, parsed_variable_decl.type)
    if error: return None, error

    init_expr, error = validate_expression(scope, validated_type, parsed_variable_decl.initializer)
    if error: return None, error

    if not validated_type.eq_or_other_safely_convertible(init_expr.type):
        return None, ValidationError(
            f'Type mismatch in variable declaration: declaration type = {validated_type}, initialization type = {init_expr.type}', parsed_variable_decl.span)

    return ValidatedVariableDeclaration([init_expr], parsed_variable_decl.span, parsed_variable_decl.name, validated_type), None


def validate_while_stmt(scope: Scope, parsed_while: ParsedWhile) -> tuple[
    Optional[ValidatedWhileStatement], Optional[ValidationError]]:
    condition, error = validate_expression(scope, CompleteType(Builtin('bool')), parsed_while.condition)
    if error: return None, error

    if not condition.type.is_builtin() or condition.type.builtin().name != 'bool':
        return None, ValidationError(f'expected boolean expression in while condition', parsed_while.condition.span)

    block, error = validate_block(scope, parsed_while.block, while_block=True)
    if error: return None, error

    return ValidatedWhileStatement([condition, block], parsed_while.span), None


def validate_break_stmt(scope: Scope, parsed_break: ParsedBreakStatement) -> tuple[
    Optional[ValidatedBreakStatement], Optional[ValidationError]]:
    if scope.inside_while_block:
        return ValidatedBreakStatement([], parsed_break.span), None

    return None, ValidationError('break statement not in while block', parsed_break.span)


def validate_if_stmt(scope: Scope, parsed_if: ParsedIfStatement) -> tuple[
    Optional[ValidatedIfStatement], Optional[ValidationError]]:
    condition, error = validate_expression(scope, CompleteType(Builtin('bool')), parsed_if.condition)
    if error: return None, error

    if not condition.type.is_builtin() or condition.type.builtin().name != 'bool':
        return None, ValidationError(f'expected boolean expression in while condition', parsed_if.condition.span)

    block, error = validate_block(scope, parsed_if.body)
    if error: return None, error

    return ValidatedIfStatement([condition, block], parsed_if.span), None


def validate_block(parent_scope: Scope, block: ParsedBlock, while_block=False) -> (ValidatedBlock | None, ValidationError | None):
    scope = Scope('', parent_scope)
    scope.inside_while_block = while_block
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
        elif isinstance(stmt, ParsedExpression):
            validated_expr, error = validate_expression(scope, None, stmt)
            if error: return None, error
            stmts.append(validated_expr)
        else:
            raise NotImplementedError(f'validation for statement {stmt} not implemented!')

    return ValidatedBlock(stmts, block.span), None


def validate_function_definition_pre(scope: Scope, function_definition: ParsedFunctionDefinition | ParsedExternFunctionDeclaration) -> (ValidatedFunctionDefinitionPre | None, ValidationError | None):
    validated_name, error = validate_name(function_definition.name)
    if error: return None, error

    validated_pars: list[ValidatedParameter] = []

    # check parameter types
    for par in function_definition.pars:
        validated_type, error = validate_type(scope, par.type)
        if error: return None, error
        validated_pars.append(ValidatedParameter([], par.span, par.name.value, validated_type))

    # check return type
    validated_return_type, error = validate_type(scope, function_definition.return_type)
    if error: return None, error

    return ValidatedFunctionDefinitionPre([validated_name, ValidatedReturnType([], function_definition.return_type.span, validated_return_type), *validated_pars], function_definition.span), None


def validate_function_definition(scope: Scope, function_definition: ParsedFunctionDefinition) -> (ValidatedFunctionDefinition | None, ValidationError | None):
    validated_name, error = validate_name(function_definition.name)
    if error: return None, error

    validated_pars: list[ValidatedParameter] = []

    # check parameter types
    for par in function_definition.pars:
        validated_type, error = validate_type(scope, par.type)
        if error: return None, error
        validated_pars.append(ValidatedParameter([], par.span, par.name.value, validated_type))

    # check return type
    validated_return_type, error = validate_type(scope, function_definition.return_type)
    if error: return None, error

    # add parameters and
    child_scope = Scope('', scope)

    for validated_par in validated_pars:
        child_scope.add_var(validated_par)

    # TODO: add function to scope for recursive functions

    validated_block, error = validate_block(child_scope, function_definition.body)
    if error: return None, error

    validated_return_stmt: Optional[ValidatedReturnStatement] = None

    for validated_stmt in validated_block.statements():
        if isinstance(validated_stmt, ValidatedReturnStatement):
            validated_return_stmt = validated_stmt
            break

    if not validated_return_stmt:
        return None, ValidationError(f'missing return in function {function_definition.name}', function_definition.span)

    if not validated_return_type.eq_or_other_safely_convertible(validated_return_stmt.expr().type):
        return None, ValidationError(f'Return type mismatch in function "{function_definition.name.value}": declared return type is {validated_return_type}, but returning expression of type {validated_return_stmt.expr().type}', function_definition.span)

    return ValidatedFunctionDefinition([validated_name, ValidatedReturnType([], function_definition.return_type.span, validated_return_type), validated_block, *validated_pars], function_definition.span), None


def validate_extern_function_declaration(scope: Scope, extern_function_declaration: ParsedExternFunctionDeclaration) -> (ValidatedExternFunctionDeclaration | None, ValidationError | None):
    validated_name, error = validate_name(extern_function_declaration.name)
    if error: return None, error

    validated_pars: list[ValidatedParameter] = []

    # check parameter types
    for par in extern_function_declaration.pars:
        validated_type, error = validate_type(scope, par.type)
        if error: return None, error
        validated_pars.append(ValidatedParameter([], par.span, par.name.value, validated_type))

    # check return type
    validated_return_type, error = validate_type(scope, extern_function_declaration.return_type)
    if error: return None, error

    return ValidatedExternFunctionDeclaration([validated_name, ValidatedReturnType([], extern_function_declaration.return_type.span,  validated_return_type), *validated_pars], extern_function_declaration.span), None


def validate_name(parsed_name: ParsedName) -> (ValidatedName | None, ValidationError | None):
    return ValidatedName([], parsed_name.span, parsed_name.value), None


def validate_array_type(scope: Scope, parsed_array_type: ParsedArrayType) -> (CompleteType | None, ValidationError | None):
    validated_type, error = validate_type(scope, parsed_array_type.parsed_type)
    if error: return None, error

    validated_expr, error = validate_expression(scope, None, parsed_array_type.length)
    if error: return None, error

    # TODO: What types should we allow for array lengths?
    if not validated_expr.type.is_integer():
        return None, ValidationError(f'array length has to be an integer, found {validated_expr.type}', validated_expr.span)

    if not isinstance(validated_expr, ValidatedNumber):
        return None, ValidationError(f'array length has to be known at compile time', validated_expr.span)

    return CompleteType(Array(int(validated_expr.value)), validated_type), None


def validate_pointer_type(scope: Scope, parsed_pointer_type: ParsedPointerType) -> (CompleteType | None, ValidationError | None):
    validated_type, error = validate_type(scope, parsed_pointer_type.parsed_type)
    if error: return None, error
    return CompleteType(Pointer(), validated_type), None


def validate_slice_type(scope: Scope, parsed_slice_type: ParsedSliceType) -> (CompleteType | None, ValidationError | None):
    validated_type, error = validate_type(scope, parsed_slice_type.parsed_type)
    if error: return None, error
    return CompleteType(Slice(), validated_type), None


def validate_type_literal(scope: Scope, parsed_type_literal: ParsedTypeLiteral) -> (CompleteType | None, ValidationError | None):
    if isinstance(parsed_type_literal, ParsedArrayType):
        return validate_array_type(scope, parsed_type_literal)
    elif isinstance(parsed_type_literal, ParsedPointerType):
        return validate_pointer_type(scope, parsed_type_literal)
    elif isinstance(parsed_type_literal, ParsedSliceType):
        return validate_slice_type(scope, parsed_type_literal)
    else:
        raise NotImplementedError()


def validate_type(scope: Scope, parsed_type: ParsedType) -> (CompleteType | None, ValidationError | None):
    if isinstance(parsed_type, ParsedExpression):
        validated_expr, error = validate_expression(scope, None, parsed_type)
        if error: return None, error

        if not validated_expr.type.is_namespace():
            return None, ValidationError(f'expression {validated_expr} does not evaluate to namespace/type', parsed_type.span)

        what = validated_expr.type.namespace().name
        if what in builtin_types:
            return CompleteType(Builtin(what)), None

        if not (struct := scope.find_struct(what)):
            return None, ValidationError(f'type {validated_expr.type.namespace().name} not found in scope {scope.namespace}', parsed_type.span)

        return CompleteType(struct), None
    elif isinstance(parsed_type, ParsedTypeLiteral):
        return validate_type_literal(scope, parsed_type)
    else:
        raise NotImplementedError()


def validate_struct_field(scope: Scope, parsed_field: ParsedField) -> (ValidatedField | None, ValidationError | None):
    name, error = validate_name(parsed_field.name)
    if error: return None, error

    validated_type, error = validate_type(scope, parsed_field.type)
    if error: return None, error

    return ValidatedField([name], parsed_field.span, validated_type), None


def validate_struct_pre(scope: Scope, parsed_struct: ParsedStruct) -> (ValidatedStructPre | None, ValidationError | None):
    name, error = validate_name(parsed_struct.name)
    if error: return None, error

    child_scope = scope.add_child_scope(name.name)

    for struct in parsed_struct.structs:
        validated_struct_pre, error = validate_struct_pre(child_scope, struct)
        if error: return None, error
        child_scope.add_struct_pre(validated_struct_pre)

    struct = Struct(parsed_struct.name.value, [], scope.get_scope_id())
    type = CompleteType(struct)

    return ValidatedStructPre([name], parsed_struct.span, type), None


def validate_struct(scope: Scope, parsed_struct: ParsedStruct) -> (ValidatedStruct | None, ValidationError | None):
    name, error = validate_name(parsed_struct.name)
    if error: return None, error

    if not (child_scope := scope.get_child_scope(name.name)):
        return None, ValidationError(f'child scope {name.name} not found in scope {scope.namespace}', parsed_struct.span)

    structs: list[ValidatedStruct] = []
    for struct in parsed_struct.structs:
        validated_struct, error = validate_struct(child_scope, struct)
        if error: return None, error
        structs.append(validated_struct)
        if not child_scope.update_struct_fields(validated_struct):
            raise "failed to update struct fields"

    fields: list[ValidatedField] = []
    for field in parsed_struct.fields:
        validated_field, error = validate_struct_field(child_scope, field)
        if error: return None, error
        fields.append(validated_field)

    struct = Struct(name.name, [StructField(field.name().name, field.type) for field in fields], scope.get_scope_id())
    return ValidatedStruct([name, *fields], parsed_struct.span, CompleteType(struct)), None


def validate_module(module: ParsedModule) -> (ValidatedModule | None, ValidationError | None):
    root_scope = create_root_scope()
    body: list[Union[ValidatedFunctionDefinition, ValidatedStruct, ValidatedVariableDeclaration]] = []

    # pre pass 1
    for stmt in module.body:
        if isinstance(stmt, ParsedStruct):
            validated_struct_pre, error = validate_struct_pre(root_scope, stmt)
            if error: return None, error
            root_scope.add_struct_pre(validated_struct_pre)

    # pre pass 2
    for stmt in module.body:
        if isinstance(stmt, ParsedFunctionDefinition) or isinstance(stmt, ParsedExternFunctionDeclaration):
            validated_function_def_pre, error = validate_function_definition_pre(root_scope, stmt)
            if error: return None, error
            function_par_types = [par.type for par in validated_function_def_pre.pars()]
            function_type = CompleteType(FunctionPointer(function_par_types, validated_function_def_pre.return_type().type))
            root_scope.functions.append(Function(validated_function_def_pre.name().name, function_type))

    # main pass 1
    for stmt in module.body:
        if isinstance(stmt, ParsedStruct):
            validated_struct, error = validate_struct(root_scope, stmt)
            if error: return None, error
            body.append(validated_struct)
            root_scope.update_struct_fields(validated_struct)

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

    struct_dict : dict[str, Struct] = {}

    # Collect all structs in all scopes.
    while len(scopes) > 0:
        scope = scopes.pop()
        for struct in scope.structs:
            struct_dict[struct.fully_qualified_name()] = struct

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

                if current.is_struct():
                    dependency = current.struct().fully_qualified_name()
                    if dependency in struct_dict:
                        out_edges[struct_id].append(dependency)
                        in_edges[dependency].append(struct_id)
                    break

                # Decay Arrays down to the fundamental types.
                if current.is_array():
                    current = current.next
                    continue

                raise NotImplementedError("Unreachable")

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


def visit_nodes(node : ValidatedNode, visitor : Callable[[ValidatedNode], None]):

    visitor(node)

    for child in node.children:
        visit_nodes(child, visitor)



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
