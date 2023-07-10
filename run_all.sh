
all_tests=('array_slices' 'arrays' 'example' 'example_big' 'extern' 'safe_implicit_conversions' 'slice_expr' 'slices' 'type_inference')

for t in ${all_tests[@]}; do
  ./run.sh $t
done
