
# Todos

## Tests

Test code that should be rejected?


## Pointer to monomorphized functions

```

generic_func(@T : type, val : T) : T {
    return T{}
}

use() : void {

    // ok
    generic_func(i32, 42)
    
    // how to get a pointer to the function we just called?
    // maybe something like this?
    
    func_ptr := bind(generic_func, i32)
    func_ptr(42)
}

```

## Pointers

How to initialize null pointers?

```
    ptr := (*i32) {}
```

