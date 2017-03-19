#include <sstream>
#include "matrix.h"

using namespace std;

namespace amunmt {
namespace FPGA {
namespace mblas {

Matrix::Matrix(const cl_context &context)
:context_(context)
,rows_(0)
,cols_(0)
{

}

Matrix::Matrix(const cl_context &context, size_t rows, size_t cols, float val)
:context_(context)
,rows_(rows)
,cols_(cols)
{

}

Matrix::Matrix(const cl_context &context, size_t rows, size_t cols, float *val)
:context_(context)
,rows_(rows)
,cols_(cols)
{
  mem_ = clCreateBuffer(context_,  CL_MEM_COPY_HOST_PTR,  sizeof(float) * rows * cols, val, NULL);
}


size_t Matrix::Rows() const
{
  return rows_;
}

size_t Matrix::Cols() const
{
  return cols_;
}

void Matrix::Resize(size_t rows, size_t cols, size_t beam, size_t batches)
{
  rows_ = rows;
  cols_ = cols;

  //clReleaseMemObject(mem_);
  mem_ = clCreateBuffer(context_,  CL_MEM_READ_WRITE,  sizeof(float) * rows * cols, NULL, NULL);

}

}
}
}