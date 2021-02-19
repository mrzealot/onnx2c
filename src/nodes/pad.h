namespace toC {

class Pad : public Node {
	public:

	// attributes
	std::string mode;
	std::vector<int> pads;
	float value;

	// inputs
	const Tensor* data;

	// output
	const Tensor* output;

	Pad() : mode("constant"), pads(), value(0.0), data(NULL), output(NULL) {
		op_name = "Pad";
	}

	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		data->print_tensor(dst, !decorate);
		dst << ", ";
		output->print_tensor(dst, !decorate);
	}


	virtual void parseAttributes( onnx::NodeProto &node ) override {

		for( const auto& a : node.attribute() ) {
			if( a.name() == "mode" )
				mode = attr_helper_string(a, "mode");
			else if( a.name() == "pads" )
				pads = attr_helper_intarr(a, "pads");
			else if( a.name() == "value" )
				value = attr_helper_float(a, "value");
		}

	}




	virtual void print(std::ostream &dst) const override
	{
		dst << "\t/* PLACEHOLDER for Pad */" << std::endl;

		// just copy stuff over, like in Flatten
		std::string type = data->data_type_str();

		dst << "\t" << type << " *data = (" << type << "*)" << data->cname() << ";" << std::endl;
		dst << "\t" << type << " *output = (" << type << "*)" << output->cname() << ";" << std::endl;

		dst << "\t" << "for( uint32_t i=0; i<" << data->data_num_elem() << "; i++ )" << std::endl;
		dst << "\t\toutput[i] = data[i];" << std::endl;
		dst << std::endl;
	}





	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{

		data = inputs[0];

		if (data->data_dim.size() * 2 != pads.size()) {
			ERROR("Incorrect pads size");
		}

		if (mode != "constant") {
			ERROR("Unimplemented Pad mode: " + mode);
		}

		// TEMPORARY: allow only syntactic padding, where everything is 0 and the shape doesn't actually change
		for (int p : pads) {
			if (p != 0) {
				ERROR("Temporaryly unimplemented: Non-zero pad within pads: " + std::to_string(p));
			}
		}

		std::vector<int> new_dim;
		int half = data->data_dim.size();
		for (int i = 0; i < half; ++i) {
			new_dim.push_back(data->data_dim[i] + pads[i] + pads[i + half]);
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
