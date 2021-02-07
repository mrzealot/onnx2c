namespace toC {

class Pad : public Node {
	public:

	// attributes
	std::string mode;

	// inputs
	const Tensor* data;
	const Tensor* pads;
	// optional input
	const Tensor* constant;

	// output
	const Tensor* output;

	Pad() : mode("constant"), data(NULL), pads(NULL), constant(NULL), output(NULL) {
		op_name = "Pad";
	}

	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		data->print_tensor(dst, !decorate);
		dst << ", ";
		pads->print_tensor(dst, !decorate);
		dst << ", ";
		if( constant ) {
			constant->print_tensor(dst, !decorate);
			dst << ", ";
		}
		output->print_tensor(dst, !decorate);
	}


	virtual void parseAttributes( onnx::NodeProto &node ) override {

		for( const auto& a : node.attribute() ) {
			if( a.name() == "mode" )
				mode = attr_helper_string(a, "mode");
		}

	}




	virtual void print(std::ostream &dst) const override
	{
		dst << "\t/* PLACEHOLDER for Pad */" << std::endl;
	}





	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{

		data = inputs[0];
		pads = inputs[1];
		if( inputs.size() == 3 )
			constant = inputs[2];
		else
			constant = NULL;

		if( typeConstraint_int64(pads) == false )
			ERROR("Incorrect pads input for node");

		if (data->data_dim.size() * 2 != (unsigned long)pads->data_num_elem()) {
			ERROR("Incorrect pads size");
		}

		if( pads->initialize == false ) {
			ERROR("Padding to a run-time defined shape is not supported");
		}

		if (mode != "constant") {
			ERROR("Unimplemented Pad mode: " + mode);
		}

		std::vector<int> new_dim;
		int half = data->data_dim.size();
		int64_t *pads_data = (int64_t*)(pads->data_buffer);
		for (int i = 0; i < half; ++i) {
			new_dim.push_back(data->data_dim[i] + pads_data[i] + pads_data[i + half]);
		}

		// TODO possible other attr/dimension checks

		Tensor *rv = new Tensor;
		
		rv->data_dim = new_dim;
		rv->data_type = data->data_type;
		output=rv;
		outputs.push_back(rv);
	}

	
};

}
