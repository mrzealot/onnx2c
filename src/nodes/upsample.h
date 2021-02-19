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




	virtual void print_helpers(std::ostream &dst) const override {

	  	dst << "\t/* Helpers for Upsample */" << std::endl;

		dst << "\tstatic inline float lerp(float s, float e, float t) { return s + (e - s) * t; }" << std::endl;
 
        dst << "\tstatic inline float blerp(float c00, float c10, float c01, float c11, float tx, float ty) { return lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty); }" << std::endl << std::endl;
	}

	virtual void print(std::ostream &dst) const override
	{
		// TEMP: bilinear upsampling is assumed for now
	  	dst << "\t/* Limited code for Upsample */" << std::endl;
	  	dst << "\t/* Naive bilinear algorithm borrowed from https://rosettacode.org/wiki/Bilinear_interpolation */" << std::endl;

		std::string type = x->data_type_str();
		// std::string scales_type = scales->data_type_str();
		// dst << "\t" << type << " *x = (" << type << "*)" << x->cname() << ";" << std::endl;
		// dst << "\t" << scales_type << " *scales = (" << scales_type << "*)" << scales->cname() << ";" << std::endl;
		// dst << "\t" << type << " *y = (" << type << "*)" << y->cname() << ";" << std::endl;

		dst << "\tuint32_t height = " << x->data_dim[2] << ";" << std::endl;
		dst << "\tuint32_t width = " << x->data_dim[3] << ";" << std::endl;
        dst << "\tuint32_t new_height = (uint32_t)(height * " << scales->cname() << "[2]);" << std::endl;
		dst << "\tuint32_t new_width = (uint32_t)(width * " << scales->cname() << "[3]);" << std::endl;

		dst << "\t/* Loop over batches */" << std::endl;
		dst << "\tfor( uint32_t b=0; b<" << x->data_dim[0] << "; b++) {" << std::endl << std::endl;
		dst << "\t\t/* Loop over channels */" << std::endl;
		dst << "\t\tfor( uint32_t c=0; c<" << x->data_dim[1] << "; c++) {" << std::endl << std::endl;
		dst << "\t\t\t/* Loop over rows */" << std::endl;
		dst << "\t\t\tfor( uint32_t h=0; h<new_height; h++) {" << std::endl << std::endl;
		dst << "\t\t\t\t/* Loop over columns */" << std::endl;
		dst << "\t\t\t\tfor( uint32_t w=0; w<new_width; w++) {" << std::endl << std::endl;
		

		dst << "\t\t\t\t\tfloat gw = ((float)w) / new_width * (width - 1);" << std::endl;
		dst << "\t\t\t\t\tfloat gh = ((float)h) / new_height * (height - 1);" << std::endl;
		dst << "\t\t\t\t\tint gwi = (int)gw;" << std::endl;
		dst << "\t\t\t\t\tint ghi = (int)gh;" << std::endl;
		dst << "\t\t\t\t\t" << type << " c00 = " << x->cname() << "[b][c][gwi][ghi];" << std::endl;
		dst << "\t\t\t\t\t" << type << " c10 = " << x->cname() << "[b][c][gwi + 1][ghi];" << std::endl;
		dst << "\t\t\t\t\t" << type << " c01 = " << x->cname() << "[b][c][gwi][ghi + 1];" << std::endl;
		dst << "\t\t\t\t\t" << type << " c11 = " << x->cname() << "[b][c][gwi + 1][ghi + 1];" << std::endl;
		dst << "\t\t\t\t\t" << y->cname() << "[b][c][h][w] = blerp(c00, c10, c01, c11, gw - gwi, gh - ghi);" << std::endl << std::endl;



		dst << "\t\t\t\t}"  << std::endl;
		dst << "\t\t\t}"  << std::endl;
		dst << "\t\t}"    << std::endl;
		dst << "\t}"      << std::endl;
	} 





	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{

		x = inputs[0];
		scales = inputs[1];

		if( typeConstraint_plainFloatingPoints(scales) == false )
			ERROR("Incorrect scales input for node");

		if (x->data_dim.size() != 4) {
			ERROR("Temporary restriction: 4D input is expected, got " + std::to_string(x->data_dim.size()));
		}

		if (x->data_dim.size() != (unsigned long)scales->data_num_elem()) {
			ERROR("Incorrect scales size");
		}

		if( scales->initialize == false ) {
			ERROR("Upsampling to a run-time defined shape is not supported");
		}

		// if (mode != "nearest") {
		// 	ERROR("Unimplemented Upsample mode: " + mode);
		// }

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
