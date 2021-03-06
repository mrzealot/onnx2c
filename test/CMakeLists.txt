
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
set(ONNX_BACKEND_TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../onnx/onnx/backend/test/data/)
set(ONNX_NODE_TEST_DATA_DIR ${ONNX_BACKEND_TEST_DATA_DIR}/node/)

# testgen utility that reads the input from a
# onnx "standard" formatted test (see directory $ONNX_BACKEND_TEST_DATA_DIR)
# and generates input, the network-under-test, expected output and a main()
# to check output matches expected output.
add_executable( testgen
	onnx_backend_tests_generator.cc)
target_link_libraries(testgen onnx2c_lib ${Protobuf_LIBRARIES})
target_compile_options(testgen
	PRIVATE
		-I${CMAKE_CURRENT_SOURCE_DIR}/../aixlog/include
	)

function( compile_onnx onnx_file c_file )
	add_custom_command(
		OUTPUT
			${c_file}
		COMMAND
			onnx2c ${onnx_file} > ${c_file}
		DEPENDS 
			${onnx_file}
			onnx2c
	) 
endfunction()


function( ONNX_type_test_build node_name data_dir accuracy test_data_set)

	add_custom_command(
		OUTPUT
		${node_name}_${test_data_set}_generated.c
		COMMAND
		testgen ${data_dir} ${accuracy} ${test_data_set} > ${node_name}_${test_data_set}_generated.c
		DEPENDS
		#TODO also depends on test data -> don't depend, always run
		testgen
		)

	add_executable( ${node_name}_${test_data_set}_test
		${node_name}_${test_data_set}_generated.c
		)
	target_compile_options( ${node_name}_${test_data_set}_test
		PRIVATE
			-Wall -Werror
			#TODO: space for output tensor is generated, but not used.
			-Wno-unused-variable
		)
	target_link_libraries( ${node_name}_${test_data_set}_test m )

endfunction()

function( ONNX_type_test node_name data_dir test_ctest_name accuracy test_data_set)
	ONNX_type_test_build(${node_name} ${data_dir} ${accuracy} ${test_data_set})
	add_test( ${test_ctest_name}
		${node_name}_${test_data_set}_test
		)
endfunction()

function( ONNX_backend_node_test node_name)
	ONNX_type_test(
		${node_name}
		${ONNX_NODE_TEST_DATA_DIR}/test_${node_name}
		ONNX_backend_${node_name}
		0.00002
		0
	)
endfunction()

set(ONNX_LOCAL_NODE_TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/local_ops/)
function( ONNX_backend_node_local_test node_name)
	ONNX_type_test(
			${node_name}
			${ONNX_LOCAL_NODE_TEST_DATA_DIR}/test_${node_name}
			ONNX_backend_${node_name}
			0.00002
			0
	)
endfunction()

function( add_perftest node_name dir accuracy input_set )
	ONNX_type_test_build(${node_name} ${dir} ${accuracy} ${input_set})
	set_property(
		TARGET ${node_name}_${input_set}_test
		PROPERTY EXCLUDE_FROM_ALL TRUE
	)
	set( PERFTESTS "${PERFTESTS};${node_name}_${input_set}_test" PARENT_SCOPE)

	# run the perftest
	add_custom_target(
		${node_name}_${input_set}_test_run
		COMMAND time --format=%U ./${node_name}_${input_set}_test
		COMMENT "Running ${node_name} ${input_set}"
		DEPENDS perftest_build
	)
	set( PERFRUNS ${PERFRUNS};${node_name}_${input_set}_test_run PARENT_SCOPE)

endfunction()




ONNX_backend_node_test(add)
ONNX_backend_node_test(add_bcast)

ONNX_backend_node_test(averagepool_1d_default)
ONNX_backend_node_test(averagepool_2d_precomputed_pads)
ONNX_backend_node_test(averagepool_2d_same_upper)
ONNX_backend_node_test(averagepool_2d_ceil)
ONNX_backend_node_test(averagepool_2d_precomputed_pads_count_include_pad)
ONNX_backend_node_test(averagepool_2d_strides)
ONNX_backend_node_test(averagepool_2d_default)
ONNX_backend_node_test(averagepool_2d_precomputed_same_upper)
ONNX_backend_node_test(averagepool_3d_default)
ONNX_backend_node_test(averagepool_2d_pads)
ONNX_backend_node_test(averagepool_2d_precomputed_strides)
ONNX_backend_node_test(averagepool_2d_pads_count_include_pad)
ONNX_backend_node_test(averagepool_2d_same_lower)

ONNX_backend_node_test(batchnorm_epsilon)
ONNX_backend_node_test(batchnorm_example)

ONNX_backend_node_test(concat_1d_axis_0)
ONNX_backend_node_test(concat_1d_axis_negative_1)
ONNX_backend_node_test(concat_2d_axis_0)
ONNX_backend_node_test(concat_2d_axis_1)
ONNX_backend_node_test(concat_2d_axis_negative_1)
ONNX_backend_node_test(concat_2d_axis_negative_2)
ONNX_backend_node_test(concat_3d_axis_0)
ONNX_backend_node_test(concat_3d_axis_1)
ONNX_backend_node_test(concat_3d_axis_2)
ONNX_backend_node_test(concat_3d_axis_negative_1)
ONNX_backend_node_test(concat_3d_axis_negative_2)
ONNX_backend_node_test(concat_3d_axis_negative_3)
ONNX_type_test(operator_concat2 ${ONNX_BACKEND_TEST_DATA_DIR}/pytorch-operator/test_operator_concat2 ONNX_backend_pytorch_conv 0.00001 0)

ONNX_backend_node_test(conv_with_strides_no_padding)
ONNX_backend_node_test(conv_with_strides_padding)
ONNX_backend_node_test(conv_with_strides_and_asymmetric_padding)
ONNX_type_test(operator_conv ${ONNX_BACKEND_TEST_DATA_DIR}/pytorch-operator/test_operator_conv ONNX_backend_pytorch_conv 0.00001 0)

ONNX_backend_node_test(dropout_default)
ONNX_backend_node_test(dropout_default_mask_ratio)
ONNX_backend_node_test(dropout_default_ratio)
ONNX_backend_node_test(dropout_default_mask)
ONNX_backend_node_test(dropout_default_old)
ONNX_backend_node_test(dropout_random_old)

ONNX_backend_node_test(flatten_axis0)
ONNX_backend_node_test(flatten_axis1)
ONNX_backend_node_test(flatten_axis2)
ONNX_backend_node_test(flatten_axis3)
ONNX_backend_node_test(flatten_default_axis)
ONNX_backend_node_test(flatten_negative_axis1)
ONNX_backend_node_test(flatten_negative_axis2)
ONNX_backend_node_test(flatten_negative_axis3)
ONNX_backend_node_test(flatten_negative_axis4)

ONNX_backend_node_test(globalaveragepool)
ONNX_backend_node_test(globalaveragepool_precomputed)

ONNX_backend_node_test(lstm_defaults)
ONNX_backend_node_test(lstm_with_initial_bias)
ONNX_backend_node_test(lstm_with_peepholes)

ONNX_backend_node_test(matmul_2d)
# only 2d matmul implemented. _integer has multiple outputs
#ONNX_backend_node_test(matmul_3d)
#ONNX_backend_node_test(matmul_4d)


ONNX_backend_node_test(maxpool_1d_default)
ONNX_backend_node_test(maxpool_2d_precomputed_pads)
ONNX_backend_node_test(maxpool_2d_strides)
# This uses implicit padding on the axis ends which
# is agianst specifications - might be fragile
# see https://github.com/onnx/onnx/issues/2971
# and https://github.com/onnx/onnx/issues/2927
ONNX_backend_node_test(maxpool_2d_ceil)
ONNX_backend_node_test(maxpool_2d_precomputed_same_upper)
ONNX_backend_node_test(maxpool_2d_uint8)
ONNX_backend_node_test(maxpool_2d_default)
ONNX_backend_node_test(maxpool_2d_precomputed_strides)
ONNX_backend_node_test(maxpool_3d_default)
ONNX_backend_node_test(maxpool_2d_dilations)
ONNX_backend_node_test(maxpool_2d_same_lower)
ONNX_backend_node_test(maxpool_2d_pads)
ONNX_backend_node_test(maxpool_2d_same_upper)
ONNX_backend_node_test(maxpool_with_argmax_2d_precomputed_pads)
#Has column-major order
#ONNX_backend_node_test(maxpool_with_argmax_2d_precomputed_strides)


ONNX_backend_node_test(relu)

ONNX_backend_node_test(reshape_extended_dims)
ONNX_backend_node_test(reshape_negative_extended_dims)
ONNX_backend_node_test(reshape_reduced_dims)
ONNX_backend_node_test(reshape_reordered_last_dims)
ONNX_backend_node_test(reshape_zero_dim)
ONNX_backend_node_test(reshape_negative_dim)
ONNX_backend_node_test(reshape_one_dim)
ONNX_backend_node_test(reshape_reordered_all_dims)
ONNX_backend_node_test(reshape_zero_and_negative_dim)

ONNX_backend_node_test(sigmoid)
ONNX_backend_node_test(sigmoid_example)

ONNX_backend_node_test(squeeze)
ONNX_backend_node_test(squeeze_negative_axes)

ONNX_backend_node_test(softmax_axis_0)
ONNX_backend_node_test(softmax_axis_1)
ONNX_backend_node_test(softmax_axis_2)
ONNX_backend_node_test(softmax_default_axis)
ONNX_backend_node_test(softmax_example)
ONNX_backend_node_test(softmax_large_number)
ONNX_backend_node_test(softmax_negative_axis)

ONNX_backend_node_test(transpose_default)
ONNX_backend_node_test(transpose_all_permutations_0)
ONNX_backend_node_test(transpose_all_permutations_1)
ONNX_backend_node_test(transpose_all_permutations_2)
ONNX_backend_node_test(transpose_all_permutations_3)
ONNX_backend_node_test(transpose_all_permutations_4)
ONNX_backend_node_test(transpose_all_permutations_5)

ONNX_backend_node_test(unsqueeze_axis_0)
ONNX_backend_node_test(unsqueeze_axis_2)
ONNX_backend_node_test(unsqueeze_negative_axes)
ONNX_backend_node_test(unsqueeze_two_axes)
ONNX_backend_node_test(unsqueeze_axis_1)
ONNX_backend_node_test(unsqueeze_axis_3)
ONNX_backend_node_test(unsqueeze_three_axes)
ONNX_backend_node_test(unsqueeze_unsorted_axes)


# More end-to-end kind of tests here
add_subdirectory(tfl_helloworld)
add_subdirectory(mnist)
add_subdirectory(velardo)
add_subdirectory(simple_networks)
add_subdirectory(squeezenet-1.0)
add_subdirectory(squeezenet-1.1)

# local op tests 
ONNX_backend_node_local_test(matmul_precision)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/perftest.cmake)
