namespace toC {

class Gemm : public Node {
	public:

	// attributes
	float alpha;
	float beta;
	int transA;
	int transB;

	// inputs
	const Tensor* a;
	const Tensor* b;
	// optional input
	const Tensor* c;

	// output
	const Tensor* y;

	Gemm() : alpha(1.0), beta(1.0), transA(0), transB(0), a(NULL), b(NULL), c(NULL), y(NULL) {
		op_name = "Gemm";
	}

	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		a->print_tensor(dst, !decorate);
		dst << ", ";
		b->print_tensor(dst, !decorate);
		dst << ", ";
		if( c ) {
			c->print_tensor(dst, !decorate);
			dst << ", ";
		}
		y->print_tensor(dst, !decorate);
	}


	virtual void parseAttributes( onnx::NodeProto &node ) override {

		for( const auto& a : node.attribute() ) {
			if( a.name() == "alpha" )
				alpha = attr_helper_float(a, "alpha");
			else if( a.name() == "beta" )
				beta = attr_helper_float(a, "beta");
			else if( a.name() == "transA" )
				transA = attr_helper_int(a, "transA");
			else if( a.name() == "transB" )
				transB = attr_helper_int(a, "transB");
		}

	}




	virtual void print(std::ostream &dst) const override
	{
		dst << "\t/* PLACEHOLDER for GEMM */" << std::endl;
	}





	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{

		a = inputs[0];
		b = inputs[1];
		if( inputs.size() == 3 )
			c = inputs[2];
		else
			b = NULL;

		if(  typeConstraint_highPrecisionNumeric(a) == false
		   ||typeConstraint_highPrecisionNumeric(b) == false)
			ERROR("Incorrect input for node");
		if( c && (typeConstraint_highPrecisionNumeric(c) == false) )
			ERROR("Incorrect input for node");


		// TODO possible other attr/dimension checks

		Tensor *rv = new Tensor;
		rv->data_dim.push_back(a->data_dim[transA ? 1 : 0]);
		rv->data_dim.push_back(b->data_dim[transB ? 0 : 1]);
		rv->data_type = a->data_type;
		y=rv;
		outputs.push_back(rv);
	}

	
};

}
