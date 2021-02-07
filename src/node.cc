
#include "error.h"
#include "graph.h"
#include "node.h"


using namespace toC;

bool Node::is_output_N_used(unsigned N)
{
	// ONNX spec:
	// "There are two ways to leave an optional input or output unspecified:
	// the first, available only for trailing inputs and outputs, is to simply
	// not provide that input; the second method is to use an empty string in
	// place of an input or output name."

	if( (int)N >= onnx_node->output_size() )
		return false;

	if( onnx_node->output(N) == "" )
		return false;

	return true;
}

bool Node::typeConstraint_highPrecisionNumeric(const Tensor *t) const
{
	return (
		   t->data_type == onnx::TensorProto_DataType_UINT32
		|| t->data_type == onnx::TensorProto_DataType_UINT64
		|| t->data_type == onnx::TensorProto_DataType_INT32
		|| t->data_type == onnx::TensorProto_DataType_INT64
		|| t->data_type == onnx::TensorProto_DataType_FLOAT16
		|| t->data_type == onnx::TensorProto_DataType_FLOAT
		|| t->data_type == onnx::TensorProto_DataType_DOUBLE
		|| t->data_type == onnx::TensorProto_DataType_BFLOAT16
	);
}
bool Node::typeConstraint_int64(const Tensor *t) const
{
	return (
		t->data_type == onnx::TensorProto_DataType_INT64
	);
}
bool Node::typeConstraint_plainFloatingPoints(const Tensor *t) const
{
	return (
		   t->data_type == onnx::TensorProto_DataType_FLOAT16
		|| t->data_type == onnx::TensorProto_DataType_FLOAT
		|| t->data_type == onnx::TensorProto_DataType_DOUBLE
	);
}
bool Node::typeConstraint_allFloatingPoints(const Tensor *t) const
{
	return (
		   typeConstraint_plainFloatingPoints(t)
		|| t->data_type == onnx::TensorProto_DataType_BFLOAT16
	);
}
bool Node::typeConstraint_8bit(const Tensor *t) const
{
	return (
		   t->data_type == onnx::TensorProto_DataType_INT8
		|| t->data_type == onnx::TensorProto_DataType_UINT8
	);
}

void Node::multidirectional_broadcast_size(
	const std::vector<int> A,
	const std::vector<int> B,
	std::vector<int> &result) const
{
		std::vector<int> dim_a = A;
		std::vector<int> dim_b = B;

		while( dim_a.size() < dim_b.size())
			dim_a.insert(dim_a.begin(), 1);
		while( dim_b.size() < dim_a.size())
			dim_b.insert(dim_b.begin(), 1);
		assert(dim_a.size() == dim_b.size());
		for( unsigned i=0; i<dim_a.size(); i++)
		{
			if( dim_a[i] == 1 || dim_b[i] == 1 )
				result.push_back( std::max(dim_a[i], dim_b[i]) );
			else if (dim_a[i] == dim_b[i])
				result.push_back( dim_a[i] );
			else
				ERROR("multidirectional_broadcast: bad tensor shapes for node " << onnx_name);
		}
}


int Node::attr_helper_int(const onnx::AttributeProto &a, const std::string &name) {
	if( a.type() != onnx::AttributeProto_AttributeType_INT )
		ERROR("Wrong attribute type for " + op_name + " attribute '" + name + "'");

	return a.i();
}

float Node::attr_helper_float(const onnx::AttributeProto &a, const std::string &name) {
	if( a.type() != onnx::AttributeProto_AttributeType_FLOAT )
		ERROR("Wrong attribute type for " + op_name + " attribute '" + name + "'");

	return a.f();
}

const std::string& Node::attr_helper_string(const onnx::AttributeProto &a, const std::string &name) {
	if( a.type() != onnx::AttributeProto_AttributeType_STRING )
		ERROR("Wrong attribute type for " + op_name + " attribute '" + name + "'");

	return a.s();
}
