#include <cmath>

namespace toC {

class Upsample : public Node {
	public:

	// attributes
	std::string mode;

	// inputs
	const Tensor* x;
	const Tensor* scales;

	// output
	const Tensor* y;

	Upsample() : mode("nearest"), x(NULL), scales(NULL), y(NULL) {
		op_name = "Upsample";
	}

	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		x->print_tensor(dst, !decorate);
		dst << ", ";
		scales->print_tensor(dst, !decorate);
		dst << ", ";
		y->print_tensor(dst, !decorate);
	}


	virtual void parseAttributes( onnx::NodeProto &node ) override {

		for( const auto& a : node.attribute() ) {
			if( a.name() == "mode" )
				mode = attr_helper_string(a, "mode");
		}

	}




	virtual void print(std::ostream &dst) const override
	{
	  	dst << "\t/* PLACEHOLDER for Upsample */" << std::endl;
	} 





	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{

		x = inputs[0];
		scales = inputs[1];

		if( typeConstraint_plainFloatingPoints(scales) == false )
			ERROR("Incorrect scales input for node");

		if (x->data_dim.size() != (unsigned long)scales->data_num_elem()) {
			ERROR("Incorrect scales size");
		}

		if( scales->initialize == false ) {
			ERROR("Upsampling to a run-time defined shape is not supported");
		}

		if (mode != "nearest") {
			ERROR("Unimplemented Upsample mode: " + mode);
		}

		std::vector<int> new_dim;
		float *scales_data = (float*)(scales->data_buffer);
		for (unsigned long i = 0; i < x->data_dim.size(); ++i) {
			new_dim.push_back((int)floor(x->data_dim[i] * scales_data[i]));
		}

		// TODO possible other attr/dimension checks

		Tensor *rv = new Tensor;
		rv->data_dim = new_dim;
		rv->data_type = x->data_type;
		y=rv;
		outputs.push_back(rv);
	}

	
};

}
