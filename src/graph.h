
#include "onnx.pb.h"

#include "node.h"
#include "tensor.h"

namespace toC {

class Graph {
public:
	Graph(
		onnx::ModelProto &onnx_model,
		bool verbose_mode = false,
		std::vector<Tensor*> inputs={}
	);

	/* print the entire .h and .cc file contents */
	void print_header(std::ostream &destination);
	void print_source(std::ostream &destination);

	/* print individual parts of the file */
	void print_file_frontmatter(std::ostream &destination);
	void print_global_tensors(std::ostream &destination);
	void print_functions(std::ostream &destination);
	void print_includes(std::ostream &dst);
	void print_interface_function(std::ostream &dst);

	/* Add already resolved onnx::TensorProto. E.g. TensorProtos that
	 * are resolved already in the ONNX model (inputs and initialized ones)
	 */
	void addResolvedTensor(onnx::TensorProto &tensor);
	Tensor* getIoTensor(onnx::ValueInfoProto &vi);

	void tryResolveNode(onnx::NodeProto &node);
	bool hasUnresolvedNodes(void);
	bool nodeInputsResolved(const onnx::NodeProto &node, std::vector<const Tensor*> &inputs);
	Node* findNode(std::string opName);

private:
	// The top-level onnx object.
	onnx::ModelProto &model;
	// The default opset version.
	unsigned version;
	// The tensors of the network. Pointers are added to this
	// vector as walking the graph resolves node outputs.
	// Each node keeps pointers internally also to its own inputs&outputs
	std::vector<Tensor*> tensors;
	// The kernels/nodes/operators of the network.
	std::vector<Node*> nodes;
	// Should onnx2c print debug info while compiling
	bool verbose_mode;

	/* Add new tensor to set of known tensors.
	 * If the tensor is not already known (checked by name),
	 * the existing tensor is updated */
	void addTensor(Tensor *t);

	Tensor *findTensor(const std::string &name) const;

	static int anonymous_nodes;
};

}

