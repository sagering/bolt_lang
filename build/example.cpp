// PRELUDE
#define i32 int
#define i64 long
// STRUCT PRE-DECLARATIONS
struct root_ABC;
struct ARRAY_6_root_B;
struct root_B;
struct PTR_root_ABC;
struct PTR_ARRAY_3_ARRAY_2_PTR_root_B;
struct ARRAY_3_ARRAY_2_PTR_root_B;
struct ARRAY_2_PTR_root_B;
struct PTR_root_B;
struct SLICE_root_B;
struct ARRAY_3_SLICE_root_B;
// SLICES
struct SLICE_root_B {
root_B* to;
i32 length;
};
// STRUCT DEFINITIONS
struct root_B {
i32 e;
root_ABC* k;
};
struct ARRAY_6_root_B {
root_B array[6];
};
struct ARRAY_2_PTR_root_B {
root_B* array[2];
};
struct ARRAY_3_SLICE_root_B {
SLICE_root_B array[3];
};
struct root_ABC {
i64 a;
i64 b;
i32 c;
ARRAY_6_root_B d;
ARRAY_3_ARRAY_2_PTR_root_B* e;
SLICE_root_B f;
ARRAY_3_SLICE_root_B g;
};
struct ARRAY_3_ARRAY_2_PTR_root_B {
ARRAY_2_PTR_root_B array[3];
};
// VARIABLE DEFINITIONS
i64 hans = (0);
i32 zimmer = (((1)+(3))-((1)/(3)));
// FUNCTION PRE-DECLARATIONS
i32* use_struct5 (root_ABC* input);
i32* use_struct6 (root_ABC* input);
i32 main ();
// FUNCTION DEFINITIONS
i32* use_struct5 (root_ABC* input){
return (&(((((*input).d).array[((1)+(1))])).e));
}
i32* use_struct6 (root_ABC* input){
return (use_struct6(input));
}
i32 main (){
root_ABC i = (root_ABC{});
return (*(use_struct5((&i))));
}
