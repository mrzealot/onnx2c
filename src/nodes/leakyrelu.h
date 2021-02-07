#include "error.h"
#include "nodes/relu.h"

namespace toC {

class LeakyRelu : public Relu {
	public:
	LeakyRelu() : Relu(), alpha(0.01) {
		op_name = "LeakyRelu";
	}
	float alpha;

	virtual void parseAttributes( onnx::NodeProto &node ) override {

		for( const auto& a : node.attribute() ) {
			if( a.name() == "alpha" )
				alpha = attr_helper_float(a, "alpha");
		}

	}

	virtual void print(std::ostream &dst) const override
	{
		std::string type = X->data_type_str();

		dst << "\t/*LeakyRelu*/" << std::endl;
		
		dst << "\t" << type << " *X = (" << type << "*)" << X->cname() << ";" << std::endl;
		dst << "\t" << type << " *Y = (" << type << "*)" << Y->cname() << ";" << std::endl;

		dst << "\t" << "for( uint32_t i=0; i<" << X->data_num_elem() << "; i++ )" << std::endl;
		dst << "\t\tY[i] = X[i] > 0 ? X[i] : " << alpha << " * X[i];" << std::endl;
		dst << std::endl;
	} 

};
}
