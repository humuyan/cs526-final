#pragma once
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "NvInferPlugin.h"
#include "kernel.h"
#include "plugin.h"

void membound(
    int bs, int feat_len, int w, float *raw, float *q, float *k, float *v);

namespace nvinfer1 {
namespace plugin {

class memboundPluginQKV : public IPluginV2DynamicExt {
public:
    int32_t getNbOutputs() const noexcept override;

    DataType getOutputDataType(int32_t index,
                               const nvinfer1::DataType *inputTypes,
                               int32_t nbInputs) const noexcept override;

    using nvinfer1::IPluginV2::getWorkspaceSize;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                            int32_t nbInputs,
                            const nvinfer1::PluginTensorDesc *outputs,
                            int32_t nbOutputs) const noexcept override;

    // DynamicExt plugins returns DimsExprs class instead of Dims
    using nvinfer1::IPluginV2::getOutputDimensions;
    DimsExprs getOutputDimensions(
        int32_t outputIndex,
        const nvinfer1::DimsExprs *inputs,
        int32_t nbInputs,
        nvinfer1::IExprBuilder &exprBuilder) noexcept override;

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    using nvinfer1::IPluginV2::enqueue;
    int32_t enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                    const nvinfer1::PluginTensorDesc *outputDesc,
                    const void *const *inputs,
                    void *const *outputs,
                    void *workspace,
                    cudaStream_t stream) noexcept override;

    void setPluginNamespace(const char *pluginNamespace) noexcept override;

    const char *getPluginNamespace() const noexcept override;

    using nvinfer1::IPluginV2Ext::configurePlugin;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in,
                         int32_t nbInputs,
                         const nvinfer1::DynamicPluginTensorDesc *out,
                         int32_t nbOutputs) noexcept override;

    nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;

    // DynamicExt plugin supportsFormat update.
    bool supportsFormatCombination(int32_t pos,
                                   const nvinfer1::PluginTensorDesc *inOut,
                                   int32_t nbInputs,
                                   int32_t nbOutputs) noexcept override;

    void serialize(void *buffer) const noexcept override;

    size_t getSerializationSize() const noexcept override;

    void destroy() noexcept override;

    const char *getPluginType() const noexcept override;

    const char *getPluginVersion() const noexcept override;

private:
    const char *mPluginNamespace;
};

class memboundPluginQKVCreator : public BaseCreator {
public:
    memboundPluginQKVCreator();

    const char *getPluginName() const noexcept override;

    const char *getPluginVersion() const noexcept override;

    const PluginFieldCollection *getFieldNames() noexcept override;

    IPluginV2DynamicExt *createPlugin(
        const char *name, const PluginFieldCollection *fc) noexcept override;

    IPluginV2DynamicExt *deserializePlugin(
        const char *name,
        const void *serialData,
        size_t serialLength) noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

}  // namespace plugin
}  // namespace nvinfer1