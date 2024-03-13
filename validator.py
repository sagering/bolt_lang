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

builtin_types = ['i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64', 'bool', 'f32', 'f64', 'void', 'type', 'typeset']

# TODO: Make sure to copy list when assigning array values
ArrayValue = list['Value']
StructValue = dict[str, 'Value']


@dataclass
class SliceValue:
    start: int
    end: int
    ptr: ArrayValue
    byte_offset = 0


@dataclass
class TypeSet:
    types: list['CompleteType']
    all: bool = False

    def includes(self, other: 'TypeSet') -> bool:
        if self.all:
            return True
        if other.all:
            return False
        if len(other.types) > len(self.types):
            return False
        for other_type in other.types:
            found = False
            for my_type in self.types:
                if other_type.eq(my_type):
                    found = True
                    break
            if not found:
                return False
        return True

    def is_single_wrapped_type(self) -> bool:
        if self.all:
            return False
        return len(self.types) == 1


Value = Union[int, float, bool, 'CompleteType', ArrayValue, StructValue, 'SliceValue', TypeSet]


@dataclass
class VariableType:
    expression: 'ValidatedExpression'
    from_expression_value: bool

    def type(self, scope: 'Scope') -> 'CompleteType':
        if self.from_expression_value:
            result = do_evaluate_expr(self.expression, scope)
            return result
        return self.expression.type


@dataclass
class Variable:
    name: str
    value: Optional[Value]
    is_comptime: bool
    type: VariableType


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
    is_incomplete: bool = False
    is_comptime: bool = False
    comptime_par_cnt: int = 0


@dataclass
class Struct:
    @dataclass
    class StructField:
        name: str
        type: 'CompleteType'

    name: Optional[str]
    is_comptime: bool
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
class TypeSetValue:
    name: str
    typeset: TypeSet


@dataclass
class Untyped:
    pass


@dataclass
class CompleteType:
    """ Complete types still require a scope context to resolve the fundamental type names.
        The complete type holds the Type that can come out of an expression.
        It's currently undecided what a Namespace should become, for now these are equivalent
        to builtin or potentially nested declared types (= fundamental types).
    """

    HoldingTypes = Union[Pointer, Array, Slice, FunctionPointer, NamedType, TypeSet, TypeSetValue, Untyped]

    val: HoldingTypes
    next: Optional['CompleteType'] = None

    def is_untype(self):
        return isinstance(self.val, Untyped)

    def is_pointer(self) -> bool:
        if isinstance(self.val, TypeSetValue):
            if self.val.typeset.all:
                return False
            return all(map(lambda t: t.is_pointer(), self.val.typeset.types))
        return isinstance(self.val, Pointer)

    def is_array(self) -> bool:
        return isinstance(self.val, Array)

    def is_slice(self) -> bool:
        if isinstance(self.val, TypeSetValue):
            if self.val.typeset.all:
                return False
            return all(map(lambda t: t.is_slice(), self.val.typeset.types))
        return isinstance(self.val, Slice)

    def is_builtin(self) -> bool:
        assert not isinstance(self.val, TypeSetValue)
        return self.is_named_type() and self.named_type().name in builtin_types

    def is_named_type(self) -> bool:
        return isinstance(self.val, NamedType)

    def is_struct(self) -> bool:
        assert not isinstance(self.val, TypeSetValue)
        return self.is_named_type() and not self.named_type().name in builtin_types

    def is_type(self) -> bool:
        return isinstance(self.val, TypeSet)

    def is_typeset(self) -> bool:
        return isinstance(self.val, NamedType) and self.named_type().name == 'typeset'

    def is_typesetvalue(self) -> bool:
        # intersection of all types in
        return isinstance(self.val, TypeSetValue)

    def is_integer(self) -> bool:
        if isinstance(self.val, TypeSetValue):
            if self.val.typeset.all:
                return False
            return all(map(lambda t: t.is_integer(), self.val.typeset.types))
        return self.is_named_type() and self.named_type().name in ['i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64']

    def is_floating_point(self) -> bool:
        if isinstance(self.val, TypeSetValue):
            if self.val.typeset.all:
                return False
            return all(map(lambda t: t.is_floating_point(), self.val.typeset.types))
        return self.is_named_type() and self.named_type().name in ['f32', 'f64']

    def is_number(self) -> bool:
        if isinstance(self.val, TypeSetValue):
            if self.val.typeset.all:
                return False
            return all(map(lambda t: t.is_number(), self.val.typeset.types))
        return self.is_integer() or self.is_floating_point()

    def is_bool(self) -> bool:
        if isinstance(self.val, TypeSetValue):
            if self.val.typeset.all:
                return False
            return all(map(lambda t: t.is_bool(), self.val.typeset.types))
        return self.is_named_type() and self.named_type().name == 'bool'

    def is_u8(self) -> bool:
        if isinstance(self.val, TypeSetValue):
            if self.val.typeset.all:
                return False
            return all(map(lambda t: t.is_u8(), self.val.typeset.types))
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
        elif self.is_typesetvalue():
            tsv = self.get()
            return f"typesetvalue({tsv.name}){{{','.join([t.to_string() for t in tsv.typeset.types])}}}"
        elif self.is_type():
            return f"type{{{','.join([t.to_string() for t in self.get().types])}}}"
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

    def eq_or_safely_convertible_from(self, other: 'CompleteType') -> bool:
        if self.is_untype() or other.is_untype():
            return True

        # TODO: the other way around
        if self.is_type() and not (other.is_type() or other.is_typeset()):
            if self.get().includes(TypeSet([other], all=False)):
                return True
            for mytype in self.get().types:
                if mytype.eq_or_safely_convertible_from(other):
                    return True
            return False

        # TODO: add documentation why this is necessary
        if self.is_typesetvalue() and other.is_typesetvalue() and self.get().name == other.get().name:
            return True

        if self.is_integer() and other.is_integer():
            if other.is_typesetvalue():
                for other_t in other.get().typeset.types:
                    if self.is_typesetvalue():
                        for my_t in self.get().typeset.types:
                            if not is_integer_convertible_from(my_t.named_type().name, other_t.named_type().name):
                                return False
                    else:
                        if not is_integer_convertible_from(self.named_type().name, other_t.named_type().name):
                            return False
            else:
                if self.is_typesetvalue():
                    for my_t in self.get().typeset.types:
                        if not is_integer_convertible_from(my_t.named_type().name, other.named_type().name):
                            return False
                else:
                    if not is_integer_convertible_from(self.named_type().name, other.named_type().name):
                        return False
            return True

        if self.is_type() and other.is_type():
            return self.val.includes(other.val)

        # TODO: Same for arrays, pointers, etc.
        if self.is_slice() and other.is_slice():
            return self.next.eq_or_safely_convertible_from(other.next)

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

        if self.is_type() and other.is_type():
            return self.get().includes(other.get()) and other.get().includes(self.get())

        return False

    def into_type(self):
        if self.is_type():
            return self

        assert not self.is_typeset()
        return CompleteType(TypeSet([self]))

    def is_binary_operator_defined(lhs: 'CompleteType', rhs: 'CompleteType', op: Operator) -> Optional['CompleteType']:

        if lhs.is_bool() and rhs.is_bool() and op == Operator.And:
            return CompleteType(NamedType('bool'))

        if lhs.is_type() and rhs.is_type() and op == Operator.Equals:
            return CompleteType(NamedType('bool'))

        if (lhs.is_type() or lhs.is_typeset()) and (rhs.is_type() or rhs.is_typeset()) and op == Operator.SetUnion:
            return CompleteType(NamedType('typeset'))

        if lhs.is_number() and rhs.is_number():
            match op:
                case Operator.Minus | Operator.Plus | Operator.Minus | Operator.Multiply | Operator.Divide:
                    return lhs
                case Operator.Equals | Operator.LessThan:
                    return CompleteType(NamedType('bool'))
                case _:
                    return None

        return None


Field = Struct("Field", True, [
    Struct.StructField("name", CompleteType(Slice(), CompleteType(NamedType("u8")))),
    Struct.StructField("type", CompleteType(TypeSet([], all=True)))
], '')

TypeInfo = Struct("TypeInfo", True, [
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


@dataclass
class Scope:
    name: str = ''
    children: list['Scope'] = field(default_factory=list)
    parent: 'Scope' = None

    evaluation_allowed: bool = False
    inside_while_block: bool = False
    scope_number: int = 0

    functions: list['ValidatedFunctionDefinition'] = field(default_factory=list)

    type_infos: dict[str, Struct] = field(default_factory=dict)
    var_value_maps: list[dict[str, 'Value']] = field(default_factory=dict)

    scope_cnt: ClassVar[int] = 0

    def __init__(self, name: str = '', parent: 'Scope' = None):
        self.functions = []
        self.vars = []
        self.var_value_maps = []
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

    def push_var_value_map(self) -> None:
        self.var_value_maps.append({})

    def pop_var_value_map(self) -> None:
        assert len(self.var_value_maps) > 0
        self.var_value_maps.pop()

    def put_var_value(self, name: str, value: 'Value', value_map_index: int = -1) -> None:
        self.var_value_maps[value_map_index][name] = value

    def get_var_value(self, name: str, value_map_index: int = -1) -> Value | None:
        if value_map_index == -1:
            value_map_index = len(self.var_value_maps) - 1
        while value_map_index >= 0:
            if self.var_value_maps[value_map_index].get(name):
                return self.var_value_maps[value_map_index][name]
            value_map_index -= 1
        if self.parent:
            return self.parent.get_var_value(name)
        return None

    def add_var(self, var: Variable):
        self.vars.append(var)

    def add_child_scope(self, name: str) -> 'Scope':
        scope = Scope(name=name)
        scope.parent = self
        scope.evaluation_allowed = self.evaluation_allowed
        self.children.append(scope)
        return scope

    def add_function(self, function):
        self.functions.append(function)

    def find_function(self, name) -> Optional['ValidatedFunctionDefinition']:
        for function in reversed(self.functions):
            if function.lookup_name == name:
                return function
        if self.parent:
            return self.parent.find_function(name)
        return None

    def get_child_scope(self, name: str) -> 'Scope':
        for scope in self.children:
            if scope.name == name:
                return scope
        raise ValueError(f"Child scope '{name}' not found")

    def find_local_var(self, name) -> Variable | None:
        for var in reversed(self.vars):
            if var.name == name:
                return var
        return None

    def _lookup_var_rec(self, name) -> Variable | None:
        for var in reversed(self.vars):
            if var.name == name:
                return var
        if self.parent:
            return self.parent._lookup_var_rec(name)
        return None

    def lookup_var(self, name) -> tuple[Variable, 'Value'] | None:
        var = self._lookup_var_rec(name)
        if var is None:
            return None
        var_value = var.value
        if var_value is None:
            var_value = self.get_var_value(name)
        return var, var_value


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

    """ 
    Idea is to lazily resolve types for all expressions to allow for type narrowing:
    
    example(@T : type, slice : []T) : i32 {
        a := &slice[0]
        
        if (T == i32) {
            return *a
        }
        
        return 0
    }
    
    The NameExpr for 'slice' will be associated with a TypeExpr '[]T'.
    """


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
class ValidatedAddressExpression:
    children: list['ValidatedExpression']
    span: Span
    type: CompleteType
    is_comptime: bool
    mode: ExpressionMode

    def rhs(self) -> 'ValidatedExpression': return self.children[0]


@dataclass
class ValidatedDerefExpression:
    children: list['ValidatedExpression']
    span: Span
    type: CompleteType
    is_comptime: bool
    mode: ExpressionMode

    def rhs(self) -> 'ValidatedExpression': return self.children[0]


@dataclass
class ValidatedPointerTypeExpression:
    children: list['ValidatedExpression']
    span: Span
    type: CompleteType
    is_comptime: bool
    mode: ExpressionMode

    def rhs(self) -> 'ValidatedExpression': return self.children[0]


@dataclass
class ValidatedArrayTypeExpression:
    children: list['ValidatedExpression']
    span: Span
    type: CompleteType
    is_comptime: bool
    mode: ExpressionMode

    def rhs(self) -> 'ValidatedExpression': return self.children[0]

    def length(self) -> 'ValidatedExpression': return self.children[1]


@dataclass
class ValidatedSliceTypeExpression:
    children: list['ValidatedExpression']
    span: Span
    type: CompleteType
    is_comptime: bool
    mode: ExpressionMode

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
    function_lookup_name: str

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
class ValidatedLenCallExpr:
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

    def type_expr(self) -> 'ValidatedExpression': return self.children[0]


@dataclass
class ValidatedDotExpr:
    children: list['ValidatedNode']
    span: Span
    type: CompleteType
    is_comptime: bool
    mode: ExpressionMode

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
    ValidatedLenCallExpr,
    ValidatedCallExpr, ValidatedDotExpr, ValidatedInitializerExpr, ValidatedIndexExpr, ValidatedArrayExpr,
    ValidatedStructExpr]


@dataclass
class ValidatedParameter:
    children: list['ValidatedNode']
    span: Span
    name: str
    bound_value: Value | None
    is_comptime: bool

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

    # named used internally, different from the name inside ValidatedName for monomorphized functions
    lookup_name: str

    is_declaration: bool
    is_incomplete: bool
    is_extern: bool
    is_comptime: bool

    validation_scope: Scope
    parsed_function_def: ParsedFunctionDefinition

    def name(self) -> ValidatedName:
        return self.children[0]

    def return_type(self) -> ValidatedExpression:
        return self.children[1]

    def body(self) -> ValidatedBlock:
        if self.is_declaration:
            raise LookupError()
        return self.children[2]

    def pars(self) -> list['ValidatedParameter']:
        if self.is_declaration:
            return self.children[2:]
        return self.children[3:]

    def type(self) -> CompleteType:
        par_types = [p.type_expr().value for p in self.pars()]
        assert isinstance(self.return_type(), ValidatedComptimeValueExpr)
        ret_type = self.return_type().value
        comptime_par_cnt = sum([1 if p.is_comptime else 0 for p in self.pars()])
        return CompleteType(
            FunctionPointer(par_types, ret_type, self.is_incomplete, self.is_comptime, comptime_par_cnt))


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

    def type_expr(self) -> ValidatedExpression:
        if len(self.children) == 2:
            return self.children[0]
        return ValidatedComptimeValueExpr([], self.span, CompleteType(TypeSet([], all=True)), True,
                                          ExpressionMode.Value, self.initializer().type)

    def initializer(self) -> ValidatedExpression: return self.children[len(self.children) - 1]


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
    type: CompleteType  # incomplete type, does not contain the fields
    name: str


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
    ValidatedBreakStmt, ValidatedIfStmt, ValidatedAssignmentStmt, ValidatedExpressionStmt, ValidatedBlock]

ValidatedNode = Union[ValidatedStatement, ValidatedExpression, SliceBoundaryPlaceholder, OtherValidatedNodes]


@dataclass
class ValidatedModule:
    children: list['ValidatedNode']
    span: Span

    structs_in_topological_order: list[Struct]
    scope: Scope

    def body(self) -> list[Union[ValidatedFunctionDefinition]]: return self.children


def is_integer_convertible_from(nominal: str, other: str) -> bool:
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
        return ValidatedComptimeValueExpr([], parsed_number.span, CompleteType(NamedType('f64')), True,
                                          ExpressionMode.Value,
                                          float(parsed_number.value)), None
    else:
        if builtin_name := integer_literal_to_type(parsed_number.value):
            return ValidatedComptimeValueExpr([], parsed_number.span,
                                              CompleteType(NamedType(builtin_name)), True, ExpressionMode.Value,
                                              int(parsed_number.value)), None
        else:
            return None, ValidationError(f'integer number {parsed_number.value} too large', parsed_number.span)


def str_to_slicevalue(s: str) -> SliceValue:
    assert (isinstance(s, str))
    bytes = s.encode('utf-8')
    return SliceValue(0, len(bytes), list(bytes))


def validate_string(type_hint: CompleteType | None, parsed_str: ParsedString) -> (
        ValidatedComptimeValueExpr | None, ValidationError | None):
    return ValidatedComptimeValueExpr([], parsed_str.span, CompleteType(Slice(), CompleteType(NamedType('u8'))), True,
                                      ExpressionMode.Value, str_to_slicevalue(parsed_str.value)), None


def validate_unop(scope: Scope, _: CompleteType | None, parsed_unop: ParsedUnaryOperation) -> (
        ValidatedUnaryOperationExpr | None, ValidationError | None):
    val_expr, error = validate_expression(scope, None, False, parsed_unop.rhs)
    if error: return None, error
    if val_expr.type.is_type() or val_expr.type.is_typeset():
        if isinstance(parsed_unop.op, ParsedOperator) and parsed_unop.op.op != Operator.Multiply:
            return None, ValidationError(f'operator not allowed for types', parsed_unop.span)

        if isinstance(parsed_unop.op, ParsedOperator) and parsed_unop.op.op == Operator.Multiply:
            return ValidatedPointerTypeExpression([val_expr], parsed_unop.span, val_expr.type, True,
                                                  ExpressionMode.Value), None

        if isinstance(parsed_unop.op, ParsedComplexOperator) and parsed_unop.op.op == ComplexOperator.Array:
            array_length_expr, error = validate_expression(scope, None, False, parsed_unop.op.par)
            if error: return None, error
            expr = ValidatedArrayTypeExpression([val_expr, array_length_expr], parsed_unop.span, val_expr.type, True,
                                                ExpressionMode.Value)
            return expr, None

        if isinstance(parsed_unop.op, ParsedComplexOperator) and parsed_unop.op.op == ComplexOperator.Slice:
            return ValidatedSliceTypeExpression([val_expr], parsed_unop.span, val_expr.type, True,
                                                ExpressionMode.Value), None

        raise NotImplementedError()
    else:
        op = parsed_unop.op.op

        if op == Operator.Address:
            if val_expr.is_comptime:
                return None, ValidationError(f'cannot take the address of compile time value', parsed_unop.span)
            return ValidatedAddressExpression([val_expr], parsed_unop.span, CompleteType(Pointer(), val_expr.type),
                                              False, ExpressionMode.Value), None

        if op == Operator.Multiply:
            if not val_expr.type.is_pointer():
                return None, ValidationError(f'cannot dereference type {val_expr.type}', parsed_unop.span)
            return ValidatedDerefExpression([val_expr], parsed_unop.span, val_expr.type.next, val_expr.is_comptime,
                                            ExpressionMode.Variable), None

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
            return ValidatedComptimeValueExpr([val_expr], parsed_unop.span,
                                              CompleteType(NamedType(integer_type_name_after_unop)),
                                              val_expr.is_comptime, ExpressionMode.Value, -val_expr.value), None

        elif not (type_after_unop := is_unary_operator_defined(val_expr.type, op)):
            return None, ValidationError(f'type {val_expr.type} does not support unary operation with operator {op}',
                                         parsed_unop.span)

        return ValidatedUnaryOperationExpr([val_expr], parsed_unop.span, type_after_unop, val_expr.is_comptime,
                                           ExpressionMode.Value, parsed_unop.op.op), None


def validate_binop(scope: Scope, type_hint: CompleteType | None,
                   parsed_binop: ParsedBinaryOperation) -> (
        ValidatedBinaryOperationExpr | None, ValidationError | None):
    lhs, error = validate_expression(scope, None, False, parsed_binop.lhs)
    if error: return None, error

    rhs, error = validate_expression(scope, None, False, parsed_binop.rhs)
    if error: return None, error

    op = parsed_binop.op.op
    type_after_binop = lhs.type.is_binary_operator_defined(rhs.type, op)

    if not type_after_binop:
        return None, ValidationError(
            f'type {lhs.type} does not support binary operation with operator {op} and other type {rhs.type}',
            parsed_binop.span)

    # if lhs is name pointing to variable of type typeset and rhs is a concrete type, mark as "type narrowing"
    return ValidatedBinaryOperationExpr([lhs, rhs], parsed_binop.span, type_after_binop,
                                        lhs.is_comptime and rhs.is_comptime, ExpressionMode.Value,
                                        parsed_binop.op.op), None


def check_struct_has_field(scope: Scope, struct_type: CompleteType, field_name: str) -> bool:
    assert struct_type.is_struct()
    ti = scope.get_type_info(struct_type.named_type().name)
    assert ti is not None
    for field in ti.fields:
        if field.name == field_name:
            return True
    return False


def get_struct_field_type(scope: Scope, struct_type: CompleteType, field_name: str) -> CompleteType:
    assert struct_type.is_struct()
    ti = scope.get_type_info(struct_type.named_type().name)
    assert ti is not None
    for field in ti.fields:
        if field.name == field_name:
            return field.type
    assert False, "not reached"


def validate_call(scope: Scope, type_hint: CompleteType | None, parsed_call: ParsedCall) -> (
        ValidatedCallExpr | ValidatedTypeInfoCallExpr | ValidatedLenCallExpr | None, ValidationError | None):
    if isinstance(parsed_call.expr, ParsedName) and parsed_call.expr.value == 'typeinfo':
        if len(parsed_call.args) != 1:
            return None, ValidationError(
                f'expected 1 argument at call to "typeinfo", got ${len(parsed_call.args)}', parsed_call.span)

        validated_arg, error = validate_expression(scope, None, True, parsed_call.args[0])
        if error: return None, error

        return ValidatedTypeInfoCallExpr([validated_arg], parsed_call.span, CompleteType(NamedType('TypeInfo')),
                                         True, mode=ExpressionMode.Value), None

    if isinstance(parsed_call.expr, ParsedName) and parsed_call.expr.value == 'field':
        if len(parsed_call.args) != 2:
            return None, ValidationError(
                f'expected 2 arguments at call to "field", got ${len(parsed_call.args)}', parsed_call.span)

        validated_arg0, error = validate_expression(scope, None, False, parsed_call.args[0])
        if error: return None, error

        if scope.evaluation_allowed:
            validated_arg0, error = try_evaluate_expression_recursively(scope, validated_arg0)
            if error: return None, error

        validated_arg1, error = validate_expression(scope, None, False, parsed_call.args[1])
        if error: return None, error

        if scope.evaluation_allowed:
            validated_arg1, error = try_evaluate_expression_recursively(scope, validated_arg1)
            if error: return None, error
        arg1_type = validated_arg1.type
        if not (arg1_type.is_slice() and arg1_type.next.is_u8()):
            return None, ValidationError(
                f'expected second argument to be a []u8', parsed_call.span)

        if scope.evaluation_allowed:
            field_name = ''.join(map(chr, validated_arg1.value.ptr))
            arg0_type = validated_arg0.type

            if not arg0_type.is_struct():
                return None, ValidationError(
                    f'expected first argument to be a struct', parsed_call.span)

            if not check_struct_has_field(scope, arg0_type, field_name):
                return None, ValidationError(
                    f'struct has no field named "{field_name}"', parsed_call.span)

            field_type = get_struct_field_type(scope, arg0_type, field_name)
        else:
            field_name = 'unknown'
            field_type = CompleteType(Untyped())

        return ValidatedDotExpr([validated_arg0, ValidatedName([], validated_arg1.span, field_name)], parsed_call.span,
                                field_type,
                                validated_arg0.is_comptime, ExpressionMode.Variable), None

    if isinstance(parsed_call.expr, ParsedName) and parsed_call.expr.value == 'len':
        if len(parsed_call.args) != 1:
            return None, ValidationError(
                f'expected 1 argument at call to "len", got ${len(parsed_call.args)}', parsed_call.span)

        validated_arg, error = validate_expression(scope, None, False, parsed_call.args[0])
        if error: return None, error
        arg_type = validated_arg.type
        if not (arg_type.is_array() or arg_type.is_slice()):
            return None, ValidationError(
                f'expression type is not an array or slice', parsed_call.span)

        return ValidatedLenCallExpr([validated_arg], parsed_call.span, CompleteType(NamedType('u32')),
                                    validated_arg.is_comptime, mode=ExpressionMode.Value), None

    # Assumption: validated_expr will be a ValidatedNameExpr
    callee_expr, error = validate_expression(scope, None, False, parsed_call.expr)
    if error: return None, error

    calle_expr_type = callee_expr.type

    if not calle_expr_type.is_function_ptr():
        return None, ValidationError(f'expression type {calle_expr_type.get()} not a function pointer',
                                     callee_expr.span)

    validated_args: list[ValidatedExpression] = []

    for arg in parsed_call.args:
        expr, error = validate_expression(scope, None, False, arg)
        if error: return None, error
        validated_args.append(expr)

    function_ptr: FunctionPointer = calle_expr_type.get()

    if len(function_ptr.pars) != len(validated_args):
        return None, ValidationError(f'Wrong number of arguments in call to function', parsed_call.span)

    function = scope.get_root_scope().find_function(callee_expr.name)

    if function_ptr.is_incomplete or function_ptr.is_comptime:
        assert function_ptr.is_comptime or function_ptr.comptime_par_cnt > 0

        if scope.evaluation_allowed:
            bindings: list['ValidatedComptimeValueExpr'] = []

            for par_index in range(function_ptr.comptime_par_cnt):
                comptime_expr, error = try_evaluate_expression_recursively(scope, validated_args[par_index])
                if error: return None, error
                bindings.append(comptime_expr)

            function, error = validate_function_definition(function.validation_scope, function.parsed_function_def,
                                                           bindings)
            if error: return None, error

            function_ptr = function.type().get()

            assert function_ptr.is_comptime or not function_ptr.is_incomplete

            if scope.find_function(function.lookup_name) is None:
                scope.get_root_scope().add_function(function)

    for idx, (par_type, arg) in enumerate(zip(function_ptr.pars, validated_args)):
        if not par_type.eq_or_safely_convertible_from(arg.type):
            return None, ValidationError(
                f'Type mismatch in {idx + 1}th argument in call to function, expected={par_type}, got={arg.type}',
                parsed_call.span)

    function_is_comptime = function.is_comptime
    function_name = function.name().name
    comptime_par_count = len(list(filter(lambda par: par.is_comptime, function.pars())))

    return ValidatedCallExpr([callee_expr, *validated_args], parsed_call.span, function_ptr.ret,
                             function_is_comptime, ExpressionMode.Value, function_name,
                             function_lookup_name=function.lookup_name,
                             comptime_arg_count=comptime_par_count), None


def validate_initializer_expression(scope: Scope, type_hint: CompleteType | None,
                                    parsed_initializer_expr: ParsedInitializerExpression) -> (
        ValidatedInitializerExpr | None, ValidationError | None):
    validated_type_expr, error = validate_expression(scope, None, True, parsed_initializer_expr.expr)
    if error: return None, error

    if not validated_type_expr.type.is_type():
        return None, ValidationError(f'expression {validated_type_expr} does not evaluate to type',
                                     validated_type_expr.span)

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
    validated_expr_type = validated_expr.type

    if validated_expr_type.is_type():
        return None, ValidationError(f'not implemented {validated_expr}', validated_expr.span)

    validated_name, error = validate_name(parsed_dot_expr.name)
    if error: return None, error

    dot_into = None
    auto_deref = False

    if validated_expr_type.is_pointer():
        auto_deref = True
        dot_into = validated_expr.type.next

    if validated_expr_type.is_named_type():
        dot_into = validated_expr.type

    if not dot_into:
        return None, ValidationError(f'cannot dot into type {validated_expr_type}', parsed_dot_expr.span)

    if type_info := scope.get_type_info(dot_into.named_type().name):
        if field := type_info.try_get_field(validated_name.name):
            if auto_deref:
                validated_expr = ValidatedDerefExpression([validated_expr], parsed_dot_expr.span, dot_into,
                                                          validated_expr.is_comptime,
                                                          validated_expr.mode)

            dot_expr = ValidatedDotExpr([validated_expr, validated_name], parsed_dot_expr.span, field.type,
                                        validated_expr.is_comptime,
                                        validated_expr.mode)
            return dot_expr, None

    return None, ValidationError(f'field {validated_name.name} not found', parsed_dot_expr.span)


def validate_index_expr(scope: Scope, type_hint: CompleteType | None,
                        parsed_index_expr: ParsedIndexExpression) -> (
        ValidatedIndexExpr | None, ValidationError | None):
    validated_expr, error = validate_expression(scope, None, False, parsed_index_expr.expr)
    if error: return None, error

    index, error = validate_expression(scope, None, False, parsed_index_expr.index)
    if error: return None, error

    index_type = index.type
    if not index_type.is_integer():
        return None, ValidationError(f'expected integer as index, got {index_type}', parsed_index_expr.index.span)

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
        start_type = start.type

        if not start_type.is_integer():
            return None, ValidationError(f'expected integer as index, got {start_type}', parsed_slice_expr.start.span)
    else:
        start = SliceBoundaryPlaceholder(span=parsed_slice_expr.span)

    if parsed_slice_expr.end:
        end, error = validate_expression(scope, None, False, parsed_slice_expr.end)
        if error: return None, error
        end_type = end.type
        if not end_type.is_integer():
            return None, ValidationError(f'expected integer as index, got {end_type}', parsed_slice_expr.end.span)
    else:
        end = SliceBoundaryPlaceholder(span=parsed_slice_expr.span)

    return ValidatedSliceExpr([validated_expr, start, end], parsed_slice_expr.span,
                              CompleteType(Slice(), validated_expr.type.next),
                              validated_expr.is_comptime,
                              ExpressionMode.Value), None


def validate_name_expr(scope: Scope, type_hint: CompleteType | None, parsed_name: ParsedName) -> (
        ValidatedNameExpr | None, ValidationError | None):
    # TODO: Respect type hint in all cases
    if var_lookup_result := scope.lookup_var(parsed_name.value):
        (var, _) = var_lookup_result
        var_type = var.type.type(scope)
        return ValidatedNameExpr([], parsed_name.span, var_type, var.is_comptime, ExpressionMode.Variable,
                                 parsed_name.value), None

    if function := scope.find_function(parsed_name.value):
        return ValidatedNameExpr([], parsed_name.span, function.type(), False, ExpressionMode.Value,
                                 parsed_name.value), None

    if parsed_name.value == 'typeinfo':
        par_type = CompleteType(NamedType('type'))
        return_type = CompleteType(NamedType('TypeInfo'))
        func_type = CompleteType(FunctionPointer([par_type], return_type, is_comptime=True, comptime_par_cnt=1))
        return ValidatedNameExpr([], parsed_name.span, func_type, True, ExpressionMode.Value, 'typeinfo'), None

    if parsed_name.value == 'slicelen':
        par_types = [CompleteType(Slice())]
        return_type = CompleteType(NamedType('u32'))
        complete_type = CompleteType(FunctionPointer(par_types, return_type))
        return ValidatedNameExpr([], parsed_name.span, complete_type, True, ExpressionMode.Value,
                                 parsed_name.value), None

    if parsed_name.value == 'typeset':
        return ValidatedNameExpr([], parsed_name.span, CompleteType(NamedType('typeset')), True, ExpressionMode.Value,
                                 parsed_name.value), None

    if parsed_name.value == 'type':
        return ValidatedNameExpr([], parsed_name.span, CompleteType(NamedType('typeset')), True, ExpressionMode.Value,
                                 parsed_name.value), None

    if scope.check_type_exists(CompleteType(NamedType(parsed_name.value))):
        typ = CompleteType(TypeSet([CompleteType(NamedType(parsed_name.value))]))
        if type_hint is not None:
            if type_hint.eq_or_safely_convertible_from(typ):
                return ValidatedNameExpr([], parsed_name.span, type_hint, True, ExpressionMode.Value,
                                         parsed_name.value), None
        return ValidatedNameExpr([], parsed_name.span, typ, True, ExpressionMode.Value,
                                 parsed_name.value), None

    if parsed_name.value in ['true', 'false']:
        return ValidatedNameExpr([], parsed_name.span, CompleteType(NamedType('bool')), True, ExpressionMode.Value,
                                 parsed_name.value), None

    return None, ValidationError(f"Unknown name '{parsed_name.value}'", parsed_name.span)


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
    if type_hint is not None and type_hint.is_array():
        element_type = type_hint.next
    else:
        element_type = None

    validated_exprs: list[ValidatedExpression] = []
    for expr in array.exprs:
        validated_expr, error = validate_expression(scope, element_type, False, expr)
        if error: return None, error
        validated_exprs.append(validated_expr)

    is_comptime: bool = not any(map(lambda expr: not expr.is_comptime, validated_exprs))

    # Only want to allow safe implicit conversions between integer type. This means that we do not allow implicit
    # conversions between floats and integers, or between any other types.

    if type_hint is not None and type_hint.is_array():
        element_type = type_hint.next
        for expr in validated_exprs:
            if not element_type.eq_or_safely_convertible_from(expr.type):
                return None, ValidationError('types do not match declared type', array.span)
    else:
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
def validate_expression(scope: Scope, type_hint: CompleteType | None, force_evaluation: bool,
                        expr: ParsedExpression) -> (
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
        validated_expr, error = validate_primary_expr(scope, type_hint, expr)
    elif isinstance(expr, ParsedUnaryOperation):
        validated_expr, error = validate_unop(scope, type_hint, expr)
    elif isinstance(expr, ParsedBinaryOperation):
        validated_expr, error = validate_binop(scope, type_hint, expr)

    if error:
        return None, error

    if force_evaluation:
        return evaluate_expression(scope, validated_expr)

    return validated_expr, None


def evaluate_expression(scope: Scope, validated_expr: ValidatedExpression) -> (
        ValidatedComptimeValueExpr | None, ValidationError | None):
    if not validated_expr.is_comptime:
        return None, ValidationError(f"Compile time evaluation failure of expression {validated_expr}",
                                     validated_expr.span)

    value = do_evaluate_expr(validated_expr, scope)
    assert value is not None
    type = validated_expr.type
    return ValidatedComptimeValueExpr([], validated_expr.span, type, True, ExpressionMode.Value,
                                      value), None


def try_evaluate_expression_recursively(scope: Scope, validated_expr: ValidatedExpression) -> (
        ValidatedExpression | None, ValidationError | None):
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


def validate_return_stmt(scope: Scope, evaluation_allowed: bool, parsed_return_stmt: ParsedReturn) -> (
        ValidatedReturnStmt | None, ValidationError | None):
    validated_expr, error = validate_expression(scope, None, False, parsed_return_stmt.expression)
    if error: return None, error

    if evaluation_allowed:
        validated_expr, error = try_evaluate_expression_recursively(scope, validated_expr)
        if error: return None, error

    # we do not have 'pure' comptime return statements
    return ValidatedReturnStmt([validated_expr], parsed_return_stmt.span, False), None


def validate_type_expr(scope: Scope, parsed_expr: ParsedExpression) -> (
        ValidatedExpression | None, ValidationError | None):
    validated_type_expr, error = validate_expression(scope, None, False, parsed_expr)
    if error: return None, None, error

    if not validated_type_expr.is_comptime:
        return None, None, ValidationError(f'expression is not comptime evaluable', validated_type_expr.span)

    type = validated_type_expr.type
    if not type.is_type() and not type.is_typeset():
        return None, None, ValidationError(f'expression does not evaluate to type or typeset', validated_type_expr.span)

    comptime_value_expr, error = evaluate_expression(scope, validated_type_expr)
    if error: return None, None, error

    return validated_type_expr, comptime_value_expr, None


def check_assignability(lhs_type: CompleteType, rhs_type: CompleteType) -> bool:
    return lhs_type.eq_or_safely_convertible_from(rhs_type)


def validate_variable_declaration(scope: Scope, evaluation_allowed: bool,
                                  parsed_variable_decl: ParsedVariableDeclaration) -> tuple[
    Optional[ValidatedVariableDeclarationStmt], Optional[ValidationError]]:
    # with type expression
    if parsed_variable_decl.type:
        validated_type_expr, type_value_expr, error = validate_type_expr(scope, parsed_variable_decl.type)
        if error: return None, error

        init_expr, error = validate_expression(scope, type_value_expr.value, False, parsed_variable_decl.initializer)
        if error: return None, error

        if init_expr.type.is_type() and not parsed_variable_decl.is_comptime:
            return None, ValidationError(
                f'Runtime variable cannot hold compile time only values',
                init_expr.span)

        if not check_assignability(type_value_expr.value, init_expr.type):
            return None, ValidationError(
                f'Type mismatch in variable declaration: declaration type = {type_value_expr.value}, initialization type = {init_expr.type}',
                parsed_variable_decl.span)

        variable_type = VariableType(validated_type_expr, True)
        validated_stmt = ValidatedVariableDeclarationStmt([type_value_expr, init_expr], parsed_variable_decl.span,
                                                          parsed_variable_decl.is_comptime, parsed_variable_decl.name)
    # without type expression
    else:
        init_expr, error = validate_expression(scope, None, False, parsed_variable_decl.initializer)
        if error: return None, error

        if init_expr.type.is_type() and not parsed_variable_decl.is_comptime:
            return None, ValidationError(
                f'Runtime variable cannot hold compile time only values',
                init_expr.span)

        variable_type = VariableType(init_expr, False)
        validated_stmt = ValidatedVariableDeclarationStmt(
            [init_expr], parsed_variable_decl.span,
            parsed_variable_decl.is_comptime, parsed_variable_decl.name)

    init_value = None
    if evaluation_allowed:
        init_expr, error = try_evaluate_expression_recursively(scope, init_expr)
        if error: return None, error
        init_expr_index = len(validated_stmt.children) - 1
        validated_stmt.children[init_expr_index] = init_expr
        if isinstance(init_expr, ValidatedComptimeValueExpr):
            init_value = init_expr.value

    scope.add_var(Variable(validated_stmt.name, init_value, validated_stmt.is_comptime, variable_type))

    return validated_stmt, None


def validate_while_stmt(scope: Scope, evaluation_allowed: bool, parsed_while: ParsedWhile) -> tuple[
    Optional[ValidatedWhileStmt], Optional[ValidationError]]:
    if evaluation_allowed and parsed_while.is_comptime:
        blocks = []

        while True:
            condition, error = validate_expression(scope, CompleteType(NamedType('bool')), True, parsed_while.condition)
            if error: return None, error

            if not condition.type.is_bool():
                return None, ValidationError(f'expected boolean expression in while condition',
                                             parsed_while.condition.span)

            if not condition.value:
                break

            block, error = validate_block(scope, evaluation_allowed, parsed_while.block, while_block=True)
            if error: return None, error

            blocks.append(block)

        validated_block = ValidatedBlock([*blocks], parsed_while.block.span)
        return ValidatedWhileStmt([condition, validated_block], parsed_while.span, True), None

    condition, error = validate_expression(scope, CompleteType(NamedType('bool')), False, parsed_while.condition)
    if error: return None, error

    if evaluation_allowed:
        condition, error = try_evaluate_expression_recursively(scope, condition)
        if error: return None, error

    if not condition.type.is_bool():
        return None, ValidationError(f'expected boolean expression in while condition', parsed_while.condition.span)

    block, error = validate_block(scope, evaluation_allowed, parsed_while.block, while_block=True)
    if error: return None, error

    # we do not have compile time while statements yet
    return ValidatedWhileStmt([condition, block], parsed_while.span, False), None


def validate_break_stmt(scope: Scope, parsed_break: ParsedBreakStatement) -> tuple[
    Optional[ValidatedBreakStmt], Optional[ValidationError]]:
    if not scope.inside_while_block:
        return None, ValidationError('break statement not in while block', parsed_break.span)

    # we do not have compile time while statements yet
    return ValidatedBreakStmt([], parsed_break.span, False), None


VariableDeclaringNode = ValidatedVariableDeclarationStmt | ValidatedParameter


def hallo(a: VariableDeclaringNode):
    if isinstance(a, ValidatedVariableDeclarationStmt):
        type_expr = a.type_expr()

    pass


def validate_if_stmt(scope: Scope, evaluation_allowed: bool, parsed_if: ParsedIfStatement) -> (
        ValidatedIfStmt | None, ValidationError | None):
    condition, error = validate_expression(scope, CompleteType(NamedType('bool')), False, parsed_if.condition)
    if error: return None, error

    if not condition.type.is_bool():
        return None, ValidationError(f'expected boolean expression in if condition', parsed_if.condition.span)

    if evaluation_allowed:
        condition, error = try_evaluate_expression_recursively(scope, condition)
        if error: return None, error
        if isinstance(condition, ValidatedComptimeValueExpr) and not condition.value:
            return None, None

    scope.push_var_value_map()

    if isinstance(condition, ValidatedBinaryOperationExpr) and condition.op == Operator.Equals:
        if isinstance(condition.lhs(), ValidatedNameExpr) and condition.lhs().type.is_type():
            rhs_type = condition.rhs().type
            if rhs_type.is_type() and rhs_type.get().is_single_wrapped_type():
                var_lookup_result = scope.lookup_var(condition.lhs().name)
                scope.put_var_value(condition.lhs().name, rhs_type.get().types[0])

    block, error = validate_block(scope, evaluation_allowed, parsed_if.body)

    scope.pop_var_value_map()

    if error: return None, error

    return ValidatedIfStmt([condition, block], parsed_if.span, parsed_if.is_comptime), None


def validate_assignment_stmt(scope: Scope, evaluation_allowed: bool, parsed_assignment: ParsedAssignment) -> tuple[
    Optional[ValidatedAssignmentStmt], Optional[ValidationError]]:
    validated_to_expr, error = validate_expression(scope, None, False, parsed_assignment.to)
    if error: return None, error

    if validated_to_expr.mode != ExpressionMode.Variable:
        return None, ValidationError(f'cannot assign to value', parsed_assignment.to.span)

    assignment_target_type = validated_to_expr.type

    value_expr, error = validate_expression(scope, None, False, parsed_assignment.value)
    if error: return None, error

    value_expr_type = value_expr.type

    if evaluation_allowed:
        value_expr, error = try_evaluate_expression_recursively(scope, value_expr)
        if error: return None, error

    if not assignment_target_type.eq_or_safely_convertible_from(value_expr_type):
        return None, ValidationError(
            f'incompatible types in assignment: {assignment_target_type.to_string()} and {value_expr_type.to_string()}',
            parsed_assignment.span)

    validated_stmt = ValidatedAssignmentStmt([validated_to_expr, value_expr], parsed_assignment.span,
                                             validated_to_expr.is_comptime)

    if evaluation_allowed and validated_stmt.is_comptime:
        comptime_assign(scope, validated_stmt.to(), value_expr)

    return validated_stmt, None


def comptime_assign(scope: Scope, to_expr: ValidatedExpression, value_expr: ValidatedComptimeValueExpr):
    if isinstance(to_expr, ValidatedNameExpr):
        var, _ = scope.lookup_var(to_expr.name)
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
        var, _ = scope.lookup_var(expr.name)
        assert var.value is not None
        return var.value
    elif isinstance(expr, ValidatedIndexExpr):
        index = do_evaluate_expr(expr.index(), scope)
        value = get_comptime_value(scope, expr.expr())
        if expr.expr().type.is_array():
            assert (isinstance(value, list))
            assert (index < len(value))
            value = value[index]
            assert value is not None
            return value
        if expr.expr().type.is_slice():
            assert (isinstance(value, SliceValue))
            assert (isinstance(value.ptr, list))
            assert (value.start + index < len(value.ptr))
            assert (value.start + index < value.end)
            value = value.ptr[value.start + index]
            assert value is not None
            return value
        raise NotImplementedError(f"Index expr: {expr.type}")
    elif isinstance(expr, ValidatedDotExpr):
        value = get_comptime_value(scope, expr.expr())
        value = value[expr.name().name]
        assert value is not None
        return value
    elif isinstance(expr, ValidatedComptimeValueExpr):
        assert expr.value is not None
        return expr.value
    else:
        raise NotImplementedError(expr)


def validate_block(parent_scope: Scope, evaluation_allowed: bool, block: ParsedBlock, while_block=False) -> (
        ValidatedBlock | None, ValidationError | None):
    scope = parent_scope.add_child_scope('')
    scope.inside_while_block = while_block
    stmts: list[ValidatedStatement] = []

    for stmt in block.statements:
        if isinstance(stmt, ParsedReturn):
            validated_stmt, error = validate_return_stmt(scope, evaluation_allowed, stmt)
            if error: return None, error
        elif isinstance(stmt, ParsedVariableDeclaration):
            validated_stmt, error = validate_variable_declaration(scope, evaluation_allowed, stmt)
            if error: return None, error
        elif isinstance(stmt, ParsedWhile):
            validated_stmt, error = validate_while_stmt(scope, evaluation_allowed, stmt)
            if error: return None, error
            if validated_stmt.is_comptime:
                stmts.append(validated_stmt.block())
                continue  # do not add this stmt to the current block
        elif isinstance(stmt, ParsedBreakStatement):
            validated_stmt, error = validate_break_stmt(scope, stmt)
            if error: return None, error
        elif isinstance(stmt, ParsedIfStatement):
            validated_stmt, error = validate_if_stmt(scope, evaluation_allowed, stmt)
            if error: return None, error
            # if can reduce to nothing
            if not validated_stmt:
                continue
        elif isinstance(stmt, ParsedAssignment):
            validated_stmt, error = validate_assignment_stmt(scope, evaluation_allowed, stmt)
            if error: return None, error
        elif isinstance(stmt, ParsedExpression):
            validated_expr, error = validate_expression(scope, None, False, stmt)
            if error: return None, error
            if evaluation_allowed:
                validated_expr, error = try_evaluate_expression_recursively(scope, validated_expr)
                if error:
                    return None, error
            validated_stmt = ValidatedExpressionStmt([validated_expr], validated_expr.span, validated_expr.is_comptime)
        else:
            raise NotImplementedError(f'validation for statement {stmt} not implemented!')

        stmts.append(validated_stmt)
    return ValidatedBlock(stmts, block.span), None


def validate_function_declaration(scope: Scope,
                                  parsed_function_definition: ParsedFunctionDefinition | ParsedExternFunctionDeclaration) -> (
        ValidatedFunctionDefinition | None, ValidationError | None):
    validated_name, error = validate_name(parsed_function_definition.name)
    if error: return None, error

    lookup_name = validated_name.name

    validated_pars: list[ValidatedParameter] = []

    child_scope = scope.add_child_scope(validated_name.name)

    # check parameter types
    for par in parsed_function_definition.pars:
        validated_type_expr, par_type_value_expr, error = validate_type_expr(child_scope, par.type)
        if error: return None, error

        validated_pars.append(
            ValidatedParameter([par_type_value_expr], par.span, par.name.value, bound_value=None,
                               is_comptime=par.is_comptime))

        child_scope.add_var(
            Variable(validated_pars[-1].name, None, par.is_comptime, VariableType(validated_type_expr, True)))

    # check return type
    validated_return_type_expr, return_type_value_expr, error = validate_type_expr(child_scope,
                                                                                   parsed_function_definition.return_type)
    if error: return None, error

    has_comptime_pars = any(map(lambda par: par.is_comptime, parsed_function_definition.pars))
    is_extern = isinstance(parsed_function_definition, ParsedExternFunctionDeclaration)
    is_comptime = not is_extern and parsed_function_definition.is_comptime
    is_incomplete = has_comptime_pars and not is_comptime

    return ValidatedFunctionDefinition(
        [validated_name, return_type_value_expr, *validated_pars], parsed_function_definition.span,
        lookup_name=lookup_name, is_declaration=True, is_incomplete=is_incomplete,
        is_extern=is_extern, is_comptime=is_comptime, validation_scope=scope,
        parsed_function_def=parsed_function_definition), None


def validate_function_definition(scope: Scope, parsed_function_definition: ParsedFunctionDefinition,
                                 bindings: list[ValidatedComptimeValueExpr] | None = None) -> (
        ValidatedFunctionDefinition | None, ValidationError | None):
    validated_name, error = validate_name(parsed_function_definition.name)
    if error: return None, error

    lookup_name = validated_name.name

    validated_pars: list[ValidatedParameter] = []
    child_scope = scope.add_child_scope(validated_name.name)

    # TODO: Make sure comptime parameters come first
    # check parameter types
    par_index = 0
    name_suffix = ""
    for par in parsed_function_definition.pars:
        validated_type_expr, par_type_value_expr, error = validate_type_expr(child_scope, par.type)
        if error: return None, error

        value = None
        if bindings is not None and par.is_comptime:
            assert len(bindings) > 0 and par_index < len(bindings), "should be checked at call site"
            arg = bindings[par_index]

            if not par_type_value_expr.value.eq_or_safely_convertible_from(arg.type):
                return None, ValidationError("Type mismatch", arg.span)

            value = arg.value
            par_index += 1
            name_suffix = str(hash(name_suffix + str(value)) + sys.maxsize + 1)

        validated_pars.append(
            ValidatedParameter([par_type_value_expr], par.span, par.name.value, bound_value=value,
                               is_comptime=par.is_comptime))
        child_scope.add_var(
            Variable(validated_pars[-1].name, value, par.is_comptime, VariableType(validated_type_expr, True)))

    if len(name_suffix) > 0:
        lookup_name += f"_{name_suffix}"

    # check return type
    validated_return_type_expr, return_type_value_expr, error = validate_type_expr(child_scope,
                                                                                   parsed_function_definition.return_type)
    if error: return None, error

    # once bindings are passed, we monomorphization happened and function becomes complete
    all_comptime_pars_bound = bindings is not None
    is_incomplete = not parsed_function_definition.is_comptime and any(
        map(lambda par: par.is_comptime, parsed_function_definition.pars)) and not all_comptime_pars_bound
    child_scope.evaluation_allowed = not (parsed_function_definition.is_comptime or is_incomplete)

    validated_block, error = validate_block(child_scope, child_scope.evaluation_allowed,
                                            parsed_function_definition.body)
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

    if not return_type_value_expr.value.eq_or_safely_convertible_from(validated_return_stmt.expr().type):
        return None, ValidationError(
            f'Return type mismatch in function "{parsed_function_definition.name.value}": declared return type is {return_type_value_expr.value}, but returning expression of type {validated_return_stmt.expr().type}',
            parsed_function_definition.span)

    return ValidatedFunctionDefinition(
        [validated_name, return_type_value_expr,
         validated_block, *validated_pars], parsed_function_definition.span, lookup_name=lookup_name,
        is_declaration=False, is_incomplete=is_incomplete, is_extern=False,
        is_comptime=parsed_function_definition.is_comptime, validation_scope=scope,
        parsed_function_def=parsed_function_definition), None


def validate_name(parsed_name: ParsedName) -> (ValidatedName | None, ValidationError | None):
    return ValidatedName([], parsed_name.span, parsed_name.value), None


def validate_struct_field(scope: Scope, parsed_field: ParsedField) -> (ValidatedField | None, ValidationError | None):
    name, error = validate_name(parsed_field.name)
    if error: return None, error
    validated_type_expr, _, error = validate_type_expr(scope, parsed_field.type)
    if error: return None, error
    return ValidatedField([name, validated_type_expr], parsed_field.span), None


def validate_struct_pre(scope: Scope, parsed_struct: ParsedStructExpression) -> (
        ValidatedStructPre | None, ValidationError | None):
    name, error = validate_name(parsed_struct.name)
    if error: return None, error
    return ValidatedStructPre([name], parsed_struct.span, CompleteType(TypeSet([], all=True)), name.name), None


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

    return ValidatedStructExpr([*fields], parsed_struct.span, CompleteType(TypeSet([], all=True)), True,
                               ExpressionMode.Value, name), None


def validate_module(module: ParsedModule) -> (ValidatedModule | None, ValidationError | None):
    root_scope = create_root_scope()

    root_scope.add_type_info(TypeInfo)
    root_scope.add_type_info(Field)

    module_functions: list[
        Union[ValidatedFunctionDefinition, ValidatedStructExpr, ValidatedVariableDeclarationStmt]] = []
    module_other_statements: list[
        Union[ValidatedFunctionDefinition, ValidatedStructExpr, ValidatedVariableDeclarationStmt]] = []

    # 1. pure comptime functions
    # 2. mixed
    # 3. runtime

    for stmt in module.body:
        if not stmt.is_comptime:
            continue
        if isinstance(stmt, ParsedVariableDeclaration):
            validated_variable_decl, error = validate_variable_declaration(root_scope, True, stmt)
            if error: return None, error
            module_other_statements.append(validated_variable_decl)

    # comptime
    for stmt in module.body:
        if not stmt.is_comptime:
            continue
        if isinstance(stmt, ParsedFunctionDefinition):
            validated_function_def_pre, error = validate_function_declaration(root_scope, stmt)
            if error: return None, error
            root_scope.add_function(validated_function_def_pre)
        if isinstance(stmt, ParsedStructExpression):
            validated_struct_pre, error = validate_struct_pre(root_scope, stmt)
            if error: return None, error
            root_scope.add_type_info(Struct(validated_struct_pre.name, False, [], ''))

    for stmt in module.body:
        if not stmt.is_comptime:
            continue
        if isinstance(stmt, ParsedFunctionDefinition):
            validated_function_def_pre, error = validate_function_definition(root_scope, stmt)
            if error: return None, error
            root_scope.add_function(validated_function_def_pre)

    for stmt in module.body:
        if not stmt.is_comptime:
            continue
        if isinstance(stmt, ParsedStructExpression):
            validated_struct_expr, error = validate_expression(root_scope, None, True, stmt)
            if error: return None, error
            module_other_statements.append(validated_struct_expr)

    for stmt in module.body:
        if not stmt.is_comptime:
            continue
        if isinstance(stmt, ParsedVariableDeclaration):
            validated_variable_decl, error = validate_variable_declaration(root_scope, True, stmt)
            if error: return None, error
            module_other_statements.append(validated_variable_decl)

    # runtime
    for stmt in module.body:
        if stmt.is_comptime:
            continue
        if isinstance(stmt, ParsedVariableDeclaration):
            validated_variable_decl, error = validate_variable_declaration(root_scope, True, stmt)
            if error: return None, error
            module_other_statements.append(validated_variable_decl)

    for stmt in module.body:
        if stmt.is_comptime:
            continue
        if isinstance(stmt, ParsedFunctionDefinition):
            validated_function_def_pre, error = validate_function_declaration(root_scope, stmt)
            if error: return None, error
            root_scope.add_function(validated_function_def_pre)
        if isinstance(stmt, ParsedExternFunctionDeclaration):
            validated_function_def_pre, error = validate_function_declaration(root_scope, stmt)
            if error: return None, error
            root_scope.add_function(validated_function_def_pre)

    for stmt in module.body:
        if stmt.is_comptime:
            continue
        if isinstance(stmt, ParsedFunctionDefinition):
            validated_function_def_pre, error = validate_function_definition(root_scope, stmt)
            if error: return None, error
            root_scope.add_function(validated_function_def_pre)

    body = module_functions + module_other_statements

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
        if struct.is_comptime:
            continue

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
            init_value = do_evaluate_expr(stmt.initializer(), scope)
            scope.add_var(Variable(stmt.name, init_value, False, VariableType(stmt.type_expr(), True)))
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
            var, _ = scope.lookup_var(stmt.to().name)
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

    expr_type = expr.type

    if isinstance(expr, ValidatedNameExpr):
        if expr.name == 'true':
            return True

        if expr.name == 'false':
            return False

        if expr.name == 'type':
            return CompleteType(TypeSet([], True))

        lookup_result = scope.lookup_var(expr.name)
        if lookup_result:
            var, var_value = lookup_result
            if var_value is not None:
                return var_value
            if var and var.is_comptime:
                var_type = var.type.type(scope)
                if var.value is None and var_type.is_type():
                    return CompleteType(TypeSetValue(var.name, var_type.get()))
                return var.value

        if expr_type.is_type():
            return CompleteType(NamedType(expr.name))

        if expr_type.is_typeset():
            return CompleteType(NamedType(expr.name))

        func = scope.find_function(expr.name)
        if func and func.is_comptime:
            return func

        raise NotImplementedError(expr)

    if isinstance(expr, ValidatedUnaryOperationExpr):
        if expr_type.is_type() or expr_type.is_typeset():

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
        if expr_type.is_integer() or expr_type.is_floating_point() or expr_type.is_bool():
            if expr.op == Operator.Plus:
                return do_evaluate_expr(expr.lhs(), scope) + do_evaluate_expr(expr.rhs(), scope)
            if expr.op == Operator.Minus:
                return do_evaluate_expr(expr.lhs(), scope) - do_evaluate_expr(expr.rhs(), scope)
            if expr.op == Operator.Multiply:
                return do_evaluate_expr(expr.lhs(), scope) * do_evaluate_expr(expr.rhs(), scope)
            if expr.op == Operator.Divide:
                return do_evaluate_expr(expr.lhs(), scope) / do_evaluate_expr(expr.rhs(), scope)
            if expr.op == Operator.LessThan:
                return do_evaluate_expr(expr.lhs(), scope) < do_evaluate_expr(expr.rhs(), scope)
            if expr.op == Operator.Equals:
                return do_evaluate_expr(expr.lhs(), scope) == do_evaluate_expr(expr.rhs(), scope)
            if expr.op == Operator.And:
                return do_evaluate_expr(expr.lhs(), scope) and do_evaluate_expr(expr.rhs(), scope)
        if expr_type.is_typeset():
            if expr.op == Operator.SetUnion:
                lhs = do_evaluate_expr(expr.lhs(), scope)
                lhs_type = expr.lhs().type

                if lhs_type.is_type():
                    assert lhs_type.is_type()
                    lhs_typeset = TypeSet([lhs])
                else:
                    assert lhs_type.is_typeset()
                    lhs_typeset = lhs.get()

                rhs = do_evaluate_expr(expr.rhs(), scope)
                rhs_type = expr.rhs().type

                if rhs_type.is_type():
                    assert rhs_type.is_type()
                    rhs_typeset = TypeSet([rhs])
                else:
                    assert rhs_type.is_typeset()
                    rhs_typeset = rhs.get()

                added = []
                for r in rhs_typeset.types:
                    for l in lhs_typeset.types:
                        if r.eq(l):
                            continue
                    added.append(r)

                lhs_typeset.types += added
                return CompleteType(lhs_typeset)

        raise NotImplementedError(expr)

    if isinstance(expr, ValidatedCallExpr):

        function_definition = scope.find_function(expr.function_lookup_name)
        execution_scope = scope.get_root_scope().add_child_scope(f'call: {expr.function_lookup_name}')

        for par in function_definition.pars():
            if not par.is_comptime:
                raise "Unbound parameter"
            var = Variable(par.name, par.bound_value, True, VariableType(par.type_expr(), True))
            execution_scope.add_var(var)

        _, (return_value, _) = evaluate_block(function_definition.body(), execution_scope)
        return return_value

    if isinstance(expr, ValidatedTypeInfoCallExpr):
        value = do_evaluate_expr(expr.args()[0], scope)
        struct = scope.get_type_info(value.named_type().name)
        fields = list(map(lambda field: {"name": str_to_slicevalue(field.name), "type": field.type}, struct.fields))
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

        # TODO: set "is_comptime" to proper value
        struct = Struct(name, False, fields, '')
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

        return comptime_initialize_value(expr_type)

    if isinstance(expr, ValidatedIndexExpr):
        return get_comptime_value(scope, expr)

    if isinstance(expr, ValidatedLenCallExpr):
        val = do_evaluate_expr(expr.args()[0], scope)
        if isinstance(val, SliceValue):
            return val.end - val.start
        assert (isinstance(val, list))
        return len(val)

    if isinstance(expr, ValidatedSliceExpr):
        src_type = expr.src().type

        if src_type.is_array():
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

        if src_type.is_slice():
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

        raise NotImplementedError(expr)

    if isinstance(expr, ValidatedSliceTypeExpression):
        return CompleteType(Slice(), do_evaluate_expr(expr.rhs(), scope))

    if isinstance(expr, ValidatedPointerTypeExpression):
        return CompleteType(Pointer(), do_evaluate_expr(expr.rhs(), scope))

    if isinstance(expr, ValidatedArrayTypeExpression):
        length = do_evaluate_expr(expr.length(), scope)
        nxt = do_evaluate_expr(expr.rhs(), scope)
        return CompleteType(Array(length), nxt)

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
