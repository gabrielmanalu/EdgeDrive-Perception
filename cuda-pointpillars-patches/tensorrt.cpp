#include "tensorrt.hpp"

#include <cuda_runtime.h>
#include <string.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "check.hpp"

namespace TensorRT {

static class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
      std::cerr << "[NVINFER LOG]: " << msg << std::endl;
    }
  }
} gLogger_;

static std::string format_shape(const nvinfer1::Dims &shape) {
  char buf[200] = {0};
  char *p = buf;
  for (int i = 0; i < shape.nbDims; ++i) {
    if (i + 1 < shape.nbDims)
      p += sprintf(p, "%ld x ", shape.d[i]);
    else
      p += sprintf(p, "%ld", shape.d[i]);
  }
  return buf;
}

static std::vector<uint8_t> load_file(const std::string &file) {
  std::ifstream in(file, std::ios::in | std::ios::binary);
  if (!in.is_open()) return {};
  in.seekg(0, std::ios::end);
  size_t length = in.tellg();
  std::vector<uint8_t> data;
  if (length > 0) {
    in.seekg(0, std::ios::beg);
    data.resize(length);
    in.read((char *)&data[0], length);
  }
  in.close();
  return data;
}

static const char *data_type_string(nvinfer1::DataType dt) {
  switch (dt) {
    case nvinfer1::DataType::kFLOAT:  return "Float32";
    case nvinfer1::DataType::kHALF:   return "Float16";
    case nvinfer1::DataType::kINT32:  return "Int32";
    case nvinfer1::DataType::kINT8:   return "Int8";
    case nvinfer1::DataType::kBOOL:   return "BOOL";
    default:                           return "Unknown";
  }
}

template <typename _T>
static void destroy_pointer(_T *ptr) {
  if (ptr) delete ptr;
}

class __native_engine_context {
 public:
  virtual ~__native_engine_context() { destroy(); }

  bool construct(const void *pdata, size_t size, const char *message_name) {
    destroy();
    if (pdata == nullptr || size == 0) {
      printf("Construct for empty data found.\n");
      return false;
    }

    runtime_ = std::shared_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(gLogger_),
        destroy_pointer<nvinfer1::IRuntime>);
    if (runtime_ == nullptr) {
      printf("Failed to create tensorRT runtime: %s.\n", message_name);
      return false;
    }

    // TRT 10.x: deserializeCudaEngine takes 2 args (no plugin factory)
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(pdata, size),
        destroy_pointer<nvinfer1::ICudaEngine>);
    if (engine_ == nullptr) {
      printf("Failed to deserialize engine: %s\n", message_name);
      return false;
    }

    context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
        engine_->createExecutionContext(),
        destroy_pointer<nvinfer1::IExecutionContext>);
    if (context_ == nullptr) {
      printf("Failed to create execution context: %s\n", message_name);
      return false;
    }
    return true;
  }

 private:
  void destroy() {
    context_.reset();
    engine_.reset();
    runtime_.reset();
  }

 public:
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  std::shared_ptr<nvinfer1::ICudaEngine>        engine_;
  std::shared_ptr<nvinfer1::IRuntime>           runtime_;
};

class EngineImplement : public Engine {
 public:
  std::shared_ptr<__native_engine_context> context_;
  // Map tensor name → index (0..N-1 across all IO tensors)
  std::unordered_map<std::string, int> binding_name_to_index_;

  virtual ~EngineImplement() = default;

  bool construct(const void *data, size_t size, const char *message_name) {
    context_ = std::make_shared<__native_engine_context>();
    if (!context_->construct(data, size, message_name)) return false;
    setup();
    return true;
  }

  bool load(const std::string &file) {
    auto data = load_file(file);
    if (data.empty()) {
      printf("Empty file: %s\n", file.c_str());
      return false;
    }
    return this->construct(data.data(), data.size(), file.c_str());
  }

  void setup() {
    auto engine = this->context_->engine_;
    // TRT 10.x: getNbIOTensors() + getIOTensorName()
    int nb = engine->getNbIOTensors();
    binding_name_to_index_.clear();
    for (int i = 0; i < nb; ++i) {
      const char *name = engine->getIOTensorName(i);
      binding_name_to_index_[name] = i;
    }
  }

  virtual int index(const std::string &name) override {
    auto iter = binding_name_to_index_.find(name);
    Assertf(iter != binding_name_to_index_.end(),
            "Cannot find binding name: %s", name.c_str());
    return iter->second;
  }

  // TRT 10.x: set tensor addresses then enqueueV3
  virtual bool forward(const std::vector<const void *> &bindings,
                       void *stream, void *input_consum_event) override {
    auto engine  = this->context_->engine_;
    auto context = this->context_->context_;
    int nb = engine->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
      const char *name = engine->getIOTensorName(i);
      context->setTensorAddress(name, const_cast<void *>(bindings[i]));
    }
    return context->enqueueV3((cudaStream_t)stream);
  }

  virtual std::vector<int> run_dims(const std::string &name) override {
    return run_dims(index(name));
  }

  virtual std::vector<int> run_dims(int ibinding) override {
    auto engine  = this->context_->engine_;
    auto context = this->context_->context_;
    const char *name = engine->getIOTensorName(ibinding);
    auto dim = context->getTensorShape(name);
    return std::vector<int>(dim.d, dim.d + dim.nbDims);
  }

  virtual std::vector<int> static_dims(const std::string &name) override {
    return static_dims(index(name));
  }

  virtual std::vector<int> static_dims(int ibinding) override {
    auto engine  = this->context_->engine_;
    const char *name = engine->getIOTensorName(ibinding);
    auto dim = engine->getTensorShape(name);
    return std::vector<int>(dim.d, dim.d + dim.nbDims);
  }

  virtual int num_bindings() override {
    return this->context_->engine_->getNbIOTensors();
  }

  virtual bool is_input(int ibinding) override {
    auto engine  = this->context_->engine_;
    const char *name = engine->getIOTensorName(ibinding);
    return engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
  }

  virtual bool set_run_dims(const std::string &name,
                            const std::vector<int> &dims) override {
    return this->set_run_dims(index(name), dims);
  }

  virtual bool set_run_dims(int ibinding,
                            const std::vector<int> &dims) override {
    auto engine  = this->context_->engine_;
    auto context = this->context_->context_;
    const char *name = engine->getIOTensorName(ibinding);
    nvinfer1::Dims d;
    memcpy(d.d, dims.data(), sizeof(int) * dims.size());
    d.nbDims = dims.size();
    return context->setInputShape(name, d);
  }

  virtual int numel(const std::string &name) override {
    return numel(index(name));
  }

  virtual int numel(int ibinding) override {
    auto engine  = this->context_->engine_;
    auto context = this->context_->context_;
    const char *name = engine->getIOTensorName(ibinding);
    auto dim = context->getTensorShape(name);
    return std::accumulate(dim.d, dim.d + dim.nbDims, 1,
                           std::multiplies<int>());
  }

  virtual DType dtype(const std::string &name) override {
    return dtype(index(name));
  }

  virtual DType dtype(int ibinding) override {
    auto engine = this->context_->engine_;
    const char *name = engine->getIOTensorName(ibinding);
    return (DType)engine->getTensorDataType(name);
  }

  virtual bool has_dynamic_dim() override {
    auto engine = this->context_->engine_;
    int nb = engine->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
      const char *name = engine->getIOTensorName(i);
      auto dims = engine->getTensorShape(name);
      for (int j = 0; j < dims.nbDims; ++j)
        if (dims.d[j] == -1) return true;
    }
    return false;
  }

  virtual void print(const char *name) override {
    printf("------------------------------------------------------\n");
    printf("%s is %s model\n", name,
           has_dynamic_dim() ? "Dynamic Shape" : "Static Shape");

    auto engine = this->context_->engine_;
    int nb = engine->getNbIOTensors();
    int num_input = 0, num_output = 0;
    for (int i = 0; i < nb; ++i) {
      const char *tname = engine->getIOTensorName(i);
      if (engine->getTensorIOMode(tname) == nvinfer1::TensorIOMode::kINPUT)
        num_input++;
      else
        num_output++;
    }

    printf("Inputs: %d\n", num_input);
    int idx = 0;
    for (int i = 0; i < nb; ++i) {
      const char *tname = engine->getIOTensorName(i);
      if (engine->getTensorIOMode(tname) != nvinfer1::TensorIOMode::kINPUT)
        continue;
      auto dim   = engine->getTensorShape(tname);
      auto dtype = engine->getTensorDataType(tname);
      printf("\t%d.%s : {%s} [%s]\n", idx++, tname,
             format_shape(dim).c_str(), data_type_string(dtype));
    }

    printf("Outputs: %d\n", num_output);
    idx = 0;
    for (int i = 0; i < nb; ++i) {
      const char *tname = engine->getIOTensorName(i);
      if (engine->getTensorIOMode(tname) != nvinfer1::TensorIOMode::kOUTPUT)
        continue;
      auto dim   = engine->getTensorShape(tname);
      auto dtype = engine->getTensorDataType(tname);
      printf("\t%d.%s : {%s} [%s]\n", idx++, tname,
             format_shape(dim).c_str(), data_type_string(dtype));
    }
    printf("------------------------------------------------------\n");
  }
};

std::shared_ptr<Engine> load(const std::string &file) {
  std::shared_ptr<EngineImplement> impl(new EngineImplement());
  if (!impl->load(file)) impl.reset();
  return impl;
}

};  // namespace TensorRT
