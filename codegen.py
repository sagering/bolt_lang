from lexer import lex
from parser import TokenSource, parse_module, print_parser_error
from validator import validate_module, ArrayValue, StructValue, Array, ValidatedExpressionStmt
from collections import defaultdict
from validator import ValidatedModule, ValidatedFunctionDefinition, ValidatedReturnStmt, \
    ValidatedVariableDeclarationStmt, ValidatedWhileStmt, ValidatedBreakStmt, ValidatedIfStmt, \
    ValidatedExpression, ValidatedNameExpr, ValidatedComptimeValueExpr, ValidatedCallExpr, ValidatedBinaryOperationExpr, \
    ValidatedUnaryOperationExpr, ValidatedDotExpr, ValidatedIndexExpr, ValidatedInitializerExpr, \
    ValidatedStatement, CompleteType, ValidatedNode, visit_nodes, \
    ValidatedArrayExpr, ValidatedExternFunctionDeclaration, \
    ValidatedSliceExpr, SliceBoundaryPlaceholder, ValidatedAssignmentStmt, Struct, Value

string_table: dict[str, int] = dict()
data_table: list[int] = dict()


# TODO: It feels like this function is badly named, but I am not sure what to call it yet.
def c_typename_with_wrapped_pointers(complete_type: CompleteType) -> str:
    """
    The purpose of this function is to generate an alphabetic name (including underscores)
    for any type, e.g. a type [3][]*B will be named ARRAY_3_SLICE_PTR_B. This name will then
    be used pre-declare and finally define the respective struct in C, and whenever this type will
    be used in C, e.g. as part of a struct field or variable declaration.
    """

    if complete_type.is_pointer():
        return 'PTR_' + c_typename_with_wrapped_pointers(complete_type.next)

    if complete_type.is_slice():
        return 'SLICE_' + c_typename_with_wrapped_pointers(complete_type.next)

    if complete_type.is_array():
        return f'ARRAY_{complete_type.array().length}_' + c_typename_with_wrapped_pointers(complete_type.next)

    if complete_type.is_named_type():
        return complete_type.named_type().name

    raise NotImplementedError(complete_type)


def c_typename_with_ptrs(type: CompleteType) -> str:
    """
    Consider the following example:

    The type [3]*[2]B becomes

    struct ARRAY_2_B
    {
        B array[2];
    };

    struct ARRAY_3_PTR_ARRAY_2_B
    {
        ARRAY_2_B* array[3]; // <-- ARRAY_2_B* is generated by this function (as opposed to PTR_ARRAY_2_B).
    };
    """

    pointers = ''

    while type.is_pointer():
        pointers += '*'
        type = type.next

    return c_typename_with_wrapped_pointers(type) + pointers


def decay(complete_type: CompleteType) -> CompleteType:
    """Removes pointers and slices to expose the underlying type of complete type."""
    if complete_type.is_pointer() or complete_type.is_slice():
        return decay(complete_type.next)
    return complete_type


def codegen_prelude() -> str:
    out = '// PRELUDE\n'
    out += '#define u8  unsigned char\n'
    out += '#define u16 unsigned short\n'
    out += '#define u32 unsigned int\n'
    out += '#define u64 unsigned long\n'
    out += '#define i8  char\n'
    out += '#define i16 short\n'
    out += '#define i32 int\n'
    out += '#define i64 long\n'
    out += '#define f32 float\n'
    out += '#define f64 double\n'
    return out


def codegen_slices(type_dict: dict[str, CompleteType]) -> str:
    out = '// SLICES\n'
    for k, v in type_dict.items():
        if v.is_slice():
            out += f'struct {c_typename_with_wrapped_pointers(v)} {{\n'
            pointers = '*'
            t = v.next
            if v.next.is_pointer():
                while t.is_pointer():
                    pointers += '*'
                    t = t.next
            out += f'{c_typename_with_wrapped_pointers(t)}{pointers} to;\n'
            out += 'i32 length;\n'
            out += '};\n'
    return out


def codegen_struct_predeclarations(type_dict: dict[str, CompleteType]) -> str:
    out = '// STRUCT PRE-DECLARATIONS\n'
    for k, v in type_dict.items():
        if v.is_builtin() or v.is_function_ptr() or decay(v).is_builtin():
            continue
        out += f'struct {c_typename_with_wrapped_pointers(v)};\n'
    return out


def codegen_struct_definitions(type_dict: dict[str, CompleteType], type_infos: dict[str, Struct]) -> str:
    # Build a dependency graph between the structs.
    out_edges = defaultdict(list)
    in_edges = defaultdict(list)

    no_dependency = set()

    for key, type in type_dict.items():
        # downstream type can be pre-declared
        if type.is_pointer() or type.is_slice():
            continue

        if type.is_named_type() and not type.is_builtin() and not type.named_type().name == 'TypeInfo':
            struct = type_infos[type.named_type().name]
            for field in struct.fields:

                if field.type.is_pointer() or field.type.is_slice() or field.type.is_builtin():
                    continue

                out_edges[type.to_string()].append(field.type.to_string())
                in_edges[field.type.to_string()].append(type.to_string())

        if type.is_array():
            if not (type.next.is_pointer() or type.next.is_slice() or type.next.is_builtin()):
                out_edges[type.to_string()].append(type.next.to_string())
                in_edges[type.next.to_string()].append(type.to_string())

        if len(out_edges[key]) == 0:
            no_dependency.add(key)

    # print(out_edges)
    # print(in_edges)
    # print(no_dependency)

    # Kahn's algorithm for topological sorting.
    ordered = []
    while len(no_dependency) > 0:
        type_id = no_dependency.pop()

        for other in in_edges[type_id]:
            out_edges[other].remove(type_id)
            if len(out_edges[other]) == 0:
                no_dependency.add(other)

        ordered.append(type_id)
    #
    # print(out_edges)
    # print(in_edges)

    # Dependency graph has cycles there are any edges left, i.e. if any of node's out edge list is not empty.
    # This is not supposed to happen because the types should have been checked already in the validate step.
    for type_id, edges in out_edges.items():
        assert (len(edges) == 0)

    out = '// STRUCT DEFINITIONS\n'
    for key in ordered:
        type = type_dict[key]

        if type.is_builtin() or type.is_function_ptr() or type.is_named_type() and type.named_type().name == 'TypeInfo':
            continue

        if type.is_named_type():
            struct = type_infos[type.named_type().name]

            out += f'struct {c_typename_with_wrapped_pointers(type)} {{\n'
            for field in struct.fields:
                if field.type.is_pointer():
                    pointers = ''
                    t = field.type
                    while t.is_pointer():
                        pointers += '*'
                        t = t.next
                    out += f'{c_typename_with_wrapped_pointers(t)}{pointers} {field.name};\n'
                else:
                    out += f'{c_typename_with_wrapped_pointers(field.type)} {field.name};\n'
            out += '};\n'

        elif type.is_array():
            out += f'struct {c_typename_with_wrapped_pointers(type)} {{\n'
            if type.next.is_pointer():
                pointers = ''
                t = type.next
                while t.is_pointer():
                    pointers += '*'
                    t = t.next
                out += f'{c_typename_with_wrapped_pointers(t)}{pointers} array[{type.array().length}];\n'
            else:
                out += f'{c_typename_with_wrapped_pointers(type.next)} array[{type.array().length}];\n'
            out += '};\n'

    return out


def expr_is_string(expr: ValidatedExpression):
    return isinstance(expr,
                      ValidatedComptimeValueExpr) and expr.type.is_slice() and expr.type.next.is_named_type() and expr.type.next.named_type().name == 'u8'


def expr_is_slice(expr: ValidatedExpression):
    return isinstance(expr, ValidatedComptimeValueExpr) and expr.type.is_slice()


def codegen_value(value: Value) -> str:
    # bool check before int check, because bool is sublcass of int in python
    if isinstance(value, bool):
        return f'{"1" if value else "0"}'
    if isinstance(value, float) or isinstance(value, int):
        return f'{value}'
    if isinstance(value, list):
        return f'{{ .array = {{ {",".join(codegen_value(v) for v in value)} }} }}'
    if isinstance(value, dict):
        fields = ','.join([f".{k} = {codegen_value(v)}" for k, v in value.items()])
        return f'{{ {fields} }}'
    raise NotImplementedError(value)


def codegen_expr(expr: ValidatedExpression) -> str:
    if isinstance(expr, ValidatedComptimeValueExpr):
        if expr.type.is_array():
            return codegen_value(expr.value)
        if expr.type.is_named_type():
            return codegen_value(expr.value)
        if expr.type.is_slice():
            slice_type_name = c_typename_with_wrapped_pointers(expr.type)
            sliced_type_name = c_typename_with_wrapped_pointers(expr.type.next)
            offset = expr.value.byte_offset
            length = expr.value.end - expr.value.start
            return f'({slice_type_name}{{&__BUFFER[{offset} + {expr.value.start} * sizeof({sliced_type_name})], {length}}})'
        raise NotImplementedError(expr)
    elif isinstance(expr, ValidatedNameExpr):
        return f'{expr.name}'
    elif isinstance(expr, ValidatedUnaryOperationExpr):
        return f'({expr.op.literal()}{codegen_expr(expr.rhs())})'
    elif isinstance(expr, ValidatedBinaryOperationExpr):
        return f'({codegen_expr(expr.lhs())}{expr.op.literal()}{codegen_expr(expr.rhs())})'
    elif isinstance(expr, ValidatedCallExpr):
        if expr.name == 'len':
            return f'({codegen_expr(expr.args()[0])}.length)'
        return f'({expr.name}({",".join([codegen_expr(arg) for arg in expr.args()[expr.comptime_arg_count:]])}))'
    elif isinstance(expr, ValidatedDotExpr):
        deref = '*' if expr.auto_deref else ''
        return f'(({deref}{codegen_expr(expr.expr())}).{expr.name().name})'
    elif isinstance(expr, ValidatedInitializerExpr):
        return f'(({c_typename_with_ptrs(expr.type)})' + '{})'
    elif isinstance(expr, ValidatedIndexExpr):
        if expr.expr().type.is_slice():
            return f'({codegen_expr(expr.expr())}.to[{codegen_expr(expr.index())}])'
        return f'({codegen_expr(expr.expr())}.array[{codegen_expr(expr.index())}])'
    elif isinstance(expr, ValidatedArrayExpr):
        array_type_name = c_typename_with_wrapped_pointers(expr.type)
        return f'({array_type_name}{{{{ {",".join(codegen_expr(expr) for expr in expr.children)} }}}})'
    elif isinstance(expr, ValidatedSliceExpr):
        slice_type_name = c_typename_with_wrapped_pointers(expr.type)

        # FIXME: When slicing an unnamed slice, this code creates the temporary (unnamed) slice twice.
        if isinstance(expr.start(), SliceBoundaryPlaceholder):
            slice_start = '(0)'
        else:
            slice_start = codegen_expr(expr.start())

        if isinstance(expr.end(), SliceBoundaryPlaceholder):
            if expr.src().type.is_array():
                slice_end = f'{expr.src().type.array().length}'
            else:
                slice_end = '(' + codegen_expr(expr.src()) + '.length)'

        else:
            slice_end = codegen_expr(expr.end())

        if expr.src().type.is_array():
            return f'({slice_type_name}{{(&{codegen_expr(expr.src())}.array[0]) + {slice_start}, ({slice_end} - {slice_start})}})'

        return f'({slice_type_name}{{{codegen_expr(expr.src())}.to + {slice_start}, ({slice_end} - {slice_start})}})'

    else:
        raise NotImplementedError(expr)


def codegen_function_definition(validated_function_definition: ValidatedFunctionDefinition,
                                predeclaration: bool) -> str:
    out = 'extern "C" ' if validated_function_definition.is_extern else ''
    pars = ','.join([f'{c_typename_with_ptrs(par.type_expr().value)} {par.name}' for par in
                     filter(lambda par: not par.is_comptime, validated_function_definition.pars())])
    out += f'{c_typename_with_ptrs(validated_function_definition.return_type().value)} {validated_function_definition.name().name}({pars})'

    if predeclaration:
        out += ';\n'
    else:
        out += ' {\n'
        for substmt in validated_function_definition.body().statements():
            out += codegen_stmt(substmt)
        out += '}\n'

    return out


def codegen_extern_function_declaration(validated_extern_function_decl: ValidatedExternFunctionDeclaration) -> str:
    out = 'extern "C" '
    pars = ','.join(
        [f'{c_typename_with_ptrs(par.type_expr().value)} {par.name}' for par in validated_extern_function_decl.pars()])
    out += f'{c_typename_with_ptrs(validated_extern_function_decl.return_type().value)} {validated_extern_function_decl.name().name} ({pars})'
    out += ';\n'
    return out


def codegen_stmt(stmt: ValidatedStatement) -> str:
    out = ''
    if isinstance(stmt, ValidatedFunctionDefinition):
        if stmt.is_comptime: return ''
        out += codegen_function_definition(stmt, False)
    elif isinstance(stmt, ValidatedVariableDeclarationStmt):
        if stmt.is_comptime:
            pass
        else:
            out += f'{c_typename_with_ptrs(stmt.type_expr().value)} {stmt.name} = {codegen_expr(stmt.initializer())};\n'
    elif isinstance(stmt, ValidatedReturnStmt):
        out += f'return {codegen_expr(stmt.expr())};\n'
    elif isinstance(stmt, ValidatedBreakStmt):
        out += 'break;\n'
    elif isinstance(stmt, ValidatedWhileStmt):
        out += f'while ({codegen_expr(stmt.condition())}) {{\n'
        for substmt in stmt.block().statements():
            out += codegen_stmt(substmt)
        out += '}\n'
    elif isinstance(stmt, ValidatedIfStmt):
        if stmt.is_comptime and not stmt.condition().value:
            pass
        elif stmt.is_comptime and stmt.condition().value:
            for substmt in stmt.block().statements():
                out += codegen_stmt(substmt)
        else:
            out += f'if ({codegen_expr(stmt.condition())}) {{\n'
            for substmt in stmt.block().statements():
                out += codegen_stmt(substmt)
            out += '}\n'
    elif isinstance(stmt, ValidatedAssignmentStmt):
        if stmt.is_comptime:
            pass
        else:
            out += f'{codegen_expr(stmt.to())} = {codegen_expr(stmt.expr())};\n'
    elif isinstance(stmt, ValidatedExpressionStmt):
        out += codegen_expr(stmt.expr()) + ';\n'
    else:
        raise NotImplementedError(stmt)

    return out


def codegen_variable_definitions(validated_module: ValidatedModule) -> str:
    out = '// VARIABLE DEFINITIONS\n'
    for stmt in validated_module.body():
        if isinstance(stmt, ValidatedVariableDeclarationStmt):
            out += codegen_stmt(stmt)
    return out


def codegen_function_predeclarations(validated_module: ValidatedModule) -> str:
    out = '// FUNCTION PRE-DECLARATIONS\n'

    for function in validated_module.scope.functions:
        if isinstance(function, ValidatedFunctionDefinition):
            if function.is_incomplete and not function.is_extern or function.is_comptime: continue
            out += codegen_function_definition(function, predeclaration=True)

    return out


def codegen_function_definitions(validated_module: ValidatedModule) -> str:
    out = '// FUNCTION DEFINITIONS\n'

    for function in validated_module.scope.functions:
        if function.is_incomplete or function.is_comptime: continue
        out += codegen_function_definition(function, predeclaration=False)

    return out


def codegen_module(validated_module: ValidatedModule) -> str:
    type_dict = {}

    def collect_types(node: ValidatedNode) -> bool:
        if isinstance(node, ValidatedFunctionDefinition) and node.is_incomplete:
            return False

        if isinstance(node, ValidatedComptimeValueExpr) and node.type.is_type():
            type = node.value
            while type:
                type_dict[type.to_string()] = type
                type = type.next

        return True

    visit_nodes(validated_module, collect_types)

    for function in validated_module.scope.functions:
        visit_nodes(function, collect_types)

    # Collect all structs from all scopes
    type_infos = {}

    scopes = [validated_module.scope]
    while len(scopes) > 0:
        scope = scopes.pop()
        type_infos.update(scope.type_infos)
        for child in scope.children:
            scopes.append(child)

    for type_info in type_infos.values():
        for field in type_info.fields:
            type = field.type
            while type:
                type_dict[type.to_string()] = type
                type = type.next

    buffer = []
    id_to_buffer_byte_offset = dict()

    def dump_rec(val: Value, type: CompleteType):
        byte_offset = len(buffer)

        if type.is_array():
            if id(val) in id_to_buffer_byte_offset:
                return id_to_buffer_byte_offset[id(val)]
            assert (isinstance(val, list))
            assert (type.next is not None)
            for element in val:
                dump_rec(element, type.next)

            id_to_buffer_byte_offset[id(val)] = byte_offset
        elif type.is_u8():
            assert (isinstance(val, int))
            buffer.append(val)
        elif type.is_struct():
            assert (isinstance(val, dict))
            ti : Struct = type_infos[type.named_type().name]
            for field in ti.fields:
                assert(field.name in val)
                dump_rec(val[field.name], field.type)
        elif type.is_slice():
            val.byte_offset = dump_rec(val.ptr, CompleteType(Array(len(val.ptr)), type.next))
        else:
            raise NotImplementedError()

        return byte_offset

    def collect_data(node: ValidatedNode) -> bool:
        if isinstance(node, ValidatedComptimeValueExpr) and node.type.is_slice():
            node.value.byte_offset = dump_rec(node.value.ptr, CompleteType(Array(len(node.value.ptr)), node.type.next))

        if isinstance(node, ValidatedStatement) and node.is_comptime:
            return False

        return True

    visit_nodes(validated_module, collect_data)

    for function in validated_module.scope.functions:
        visit_nodes(function, collect_data)

    out = ''
    out += codegen_prelude()
    out += codegen_struct_predeclarations(type_dict)
    out += codegen_slices(type_dict)
    out += codegen_buffer(buffer)
    out += codegen_struct_definitions(type_dict, type_infos)
    out += codegen_variable_definitions(validated_module)
    out += codegen_function_predeclarations(validated_module)
    out += codegen_function_definitions(validated_module)

    return out


def codegen_buffer(buffer: list[int]) -> str:
    if len(buffer) == 0:
        return ''
    values = ','.join([f'{str(byte)}' for byte in buffer])
    return f"u8 __BUFFER[{len(buffer)}] = {{{values}}};\n"


def main():
    file_name = 'tests/example.bolt'

    with open(file_name, 'r') as f:
        text = f.read()

    (tokens, error) = lex(text)
    if error:
        print(error)
        return

    token_source = TokenSource(tokens, 0, text)
    (parsed_module, error) = parse_module(token_source)

    if error: return print_parser_error(error, text)

    validated_module, error = validate_module(parsed_module)
    if error: return print(error)

    code = codegen_module(validated_module)
    print(code)


if __name__ == '__main__':
    main()
