#include "memboundPluginQKV.h"

#include <iostream>

#include "checkMacrosPlugin.h"
#include "kernel.h"

using namespace nvinfer1;
using nvinfer1::plugin::memboundPluginQKV;
using nvinfer1::plugin::memboundPluginQKVCreator;

static const char *MEMBOUND_PLUGIN_QKV_VERSION{"1"};
static const char *MEMBOUND_PLUGIN_QKV_NAME{"MemboundPluginQKV"};
PluginFieldCollection memboundPluginQKVCreator::mFC{};
std::vector<PluginField> memboundPluginQKVCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(memboundPluginQKVCreator);

// Plugin

int memboundPluginQKV::getNbOutputs() const noexcept {
    return 3;
}

nvinfer1::DataType memboundPluginQKV::getOutputDataType(
    int32_t index,
    const nvinfer1::DataType *inputTypes,
    int32_t nbInputs) const noexcept {
    return inputTypes[0];
}

size_t memboundPluginQKV::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs,
    int32_t nbInputs,
    const nvinfer1::PluginTensorDesc *outputs,
    int32_t nbOutputs) const noexcept {
    return 0;
}

DimsExprs memboundPluginQKV::getOutputDimensions(
    int32_t outputIndex,
    const nvinfer1::DimsExprs *inputs,
    int32_t nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) noexcept {
    assert(nbInputs == 1);
    assert(inputs[0].nbDims == 4);
    DimsExprs output(inputs[0]);
    int num_heads, per_head_size, feat_len = inputs[0].d[1]->getConstantValue();
    switch (feat_len) {
    case 576:
        num_heads = 12;
        break;
    case 1152:
        num_heads = 24;
        break;
    }
    output.d[1] = exprBuilder.constant(num_heads);
    per_head_size = inputs[0].d[1]->getConstantValue() / num_heads / 3;
    int resolution =
        inputs[0].d[2]->getConstantValue() * inputs[0].d[3]->getConstantValue();
    output.d[2] =
        exprBuilder.constant(outputIndex == 1 ? per_head_size : resolution);
    output.d[3] = exprBuilder.constant(outputIndex == 0   ? per_head_size
                                       : outputIndex == 1 ? resolution
                                                          : per_head_size + 1);
    return output;
}

int32_t memboundPluginQKV::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                   const nvinfer1::PluginTensorDesc *outputDesc,
                                   const void *const *inputs,
                                   void *const *outputs,
                                   void *workspace,
                                   cudaStream_t stream) noexcept {
    membound(inputDesc->dims.d[0],
             inputDesc->dims.d[1],
             inputDesc->dims.d[2],
             (float *) inputs[0],
             (float *) outputs[0],
             (float *) outputs[1],
             (float *) outputs[2]);
    return 0;
}

void memboundPluginQKV::setPluginNamespace(
    const char *pluginNamespace) noexcept {
    this->mPluginNamespace = pluginNamespace;
}

const char *memboundPluginQKV::getPluginNamespace() const noexcept {
    return this->mPluginNamespace;
}

void memboundPluginQKV::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in,
    int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *out,
    int32_t nbOutputs) noexcept {
    assert(nbInputs == 1);
    assert(nbOutputs == 3);
}

nvinfer1::IPluginV2DynamicExt *memboundPluginQKV::clone() const noexcept {
    memboundPluginQKV *ret = new memboundPluginQKV;
    ret->setPluginNamespace(mPluginNamespace);
    return ret;
}

bool memboundPluginQKV::supportsFormatCombination(
    int32_t pos,
    const nvinfer1::PluginTensorDesc *inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept {
    return (inOut[pos].type == DataType::kFLOAT &&
            inOut[pos].format == TensorFormat::kLINEAR);
}

void memboundPluginQKV::serialize(void *buffer) const noexcept {
}

size_t memboundPluginQKV::getSerializationSize() const noexcept {
    return 0;
}

void memboundPluginQKV::destroy() noexcept {
}

const char *memboundPluginQKV::getPluginType() const noexcept {
    return MEMBOUND_PLUGIN_QKV_NAME;
}

const char *memboundPluginQKV::getPluginVersion() const noexcept {
    return MEMBOUND_PLUGIN_QKV_VERSION;
}

int memboundPluginQKV::initialize() noexcept {
    return 0;
}

void memboundPluginQKV::terminate() noexcept {
}

// Plugin Creator

memboundPluginQKVCreator::memboundPluginQKVCreator() {
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *memboundPluginQKVCreator::getPluginName() const noexcept {
    return MEMBOUND_PLUGIN_QKV_NAME;
}

const char *memboundPluginQKVCreator::getPluginVersion() const noexcept {
    return MEMBOUND_PLUGIN_QKV_VERSION;
}

const PluginFieldCollection *
memboundPluginQKVCreator::getFieldNames() noexcept {
    return &mFC;
}

IPluginV2DynamicExt *memboundPluginQKVCreator::createPlugin(
    const char *name, const PluginFieldCollection *fc) noexcept {
    memboundPluginQKV *ret = new memboundPluginQKV();
    ret->setPluginNamespace(mNamespace.c_str());
    return ret;
}

IPluginV2DynamicExt *memboundPluginQKVCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) noexcept {
    memboundPluginQKV *ret = new memboundPluginQKV();
    ret->setPluginNamespace(mNamespace.c_str());
    return ret;
}