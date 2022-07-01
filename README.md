# Usage

## 0. Prerequisites

- python 3
- a C++ compiler (for example gcc or clang)

## 1. Transpile to C++
```commandline
mkdir build
python ./compiler.py ./tests/example.bolt ./build/example.cpp
```

## 2. Compile
```commandline
docker run -v ${PWD}:/home/ --rm gcc:10.2 g++ ./home/build/example.cpp -o ./home/build/example
```

## 3. Run
```commandline
docker run -v ${PWD}:/home/ --rm gcc:10.2 ./home/build/example
```


## Inspect generated C++ code

If you are interested to look into the generated C++ code, I'd suggest you run `clang-format` on it
for a better experience.

```commandline
clang-format -i ./example.cpp
```

# Language Overview

Below is an overview of some of the planned language features / ideas.

## Structs

```
struct MyType
{
    x : i32
    y : i32
}

let a : MyType = { .x = 1, .y = 2 }
```

## Pointers & Slices

Different kinds of pointers.
```
let a : i32      = 1
let b : *i32     = &a
let c : [4]i32   = { 0, 1, 3, 4 }
let d : []i32    = allocate(i32, 1999) 
let e : *[]i32   = &c 
let f : i32      = 3
let g : *mut i32 = 4

*g = 123
```

Dereferencing pointers.
```
let a : i32  = 1
let b : *i32 = &a
let c : i32  = *b

let d : MyType = { .x = 1, .y = 2 }
let e : *mut MyType = &d

e.x = 3
```

## Generics 
```
struct List(T)
{
    items: [10] *T
    length: u64
}
```

## Traits
```
trait hashable(T)
{
    fn hash( self : * T ) u64
}

trait comparable(T)
{
    fn equal( self : * T, other : * T ) bool
}

struct HashMap(Key : hashable & comparable & sized, Value : sized)
{
    allocator : *mut Allocator
    
    fn new( allocator : * mut Allocator )
    {
        return HashMap { .allocator = allocator }
    }
    
    fn insert(self : *mut Self, key : Key, val : Value ) void | error 
    {
        ...
    }
    
    fn contains_key( self : *Self, key : Key) bool
    {
        ...
    }
}

... 

string := try String.new(allocator, "abc")
mut hashmap = HashMap(i32, String).new(allocator)
try hashmap.insert( 16, string )
```

## Optionals
```

struct MyStruct
{
    a : ?i32
}

let b = MyStruct { .a = none }
```

## Defer
```

do_something_else() : void
{
    ...
}

do_something( a : i32 ) : i32
{
    defer do_something_else()
    
    ...
}
```

## Mutability
```
...
```

## Error handling
```

do_something_that_may_fail( a : i32 ) : !void
{
}

...

try de_something_that_may_fail(1)
or 
{
    // handle failure
}

...
try de_something_that_may_fail(2) // propage failure
...

```

## Safety features

### Bounds checks
### Integer overflow checks
### No implicit type conversions
