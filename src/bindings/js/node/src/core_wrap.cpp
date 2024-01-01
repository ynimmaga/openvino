// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "core_wrap.hpp"

#include "addon.hpp"
#include "compiled_model.hpp"
#include "model_wrap.hpp"
#include "read_model_args.hpp"

CoreWrap::CoreWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<CoreWrap>(info), _core{} {}

Napi::Function CoreWrap::get_class_constructor(Napi::Env env) {
    return DefineClass(env,
                       "Core",
                       {
                           InstanceMethod("readModelSync", &CoreWrap::read_model_sync),
                           InstanceMethod("readModel", &CoreWrap::read_model_async),
                           InstanceMethod("compileModelSync", &CoreWrap::compile_model_sync_dispatch),
                           InstanceMethod("compileModel", &CoreWrap::compile_model_async),
                       });
}

Napi::Object CoreWrap::init(Napi::Env env, Napi::Object exports) {
    const auto& prototype = get_class_constructor(env);

    const auto ref = new Napi::FunctionReference();
    *ref = Napi::Persistent(prototype);
    const auto data = env.GetInstanceData<AddonData>();
    data->core_prototype = ref;

    exports.Set("Core", prototype);
    return exports;
}

Napi::Value CoreWrap::read_model_sync(const Napi::CallbackInfo& info) {
    try {
        ReadModelArgs* args;
        args = new ReadModelArgs(info);
        auto model = args->model_str.empty() ? _core.read_model(args->model_path, args->bin_path)
                                             : _core.read_model(args->model_str, args->weight_tensor);
        delete args;

        return ModelWrap::wrap(info.Env(), model);
    } catch (std::runtime_error& err) {
        reportError(info.Env(), err.what());

        return info.Env().Undefined();
    }
}

Napi::Value CoreWrap::read_model_async(const Napi::CallbackInfo& info) {
    try {
        ReadModelArgs* args = new ReadModelArgs(info);
        ReaderWorker* _readerWorker = new ReaderWorker(info.Env(), args);
        _readerWorker->Queue();

        return _readerWorker->GetPromise();
    } catch (std::runtime_error& err) {
        reportError(info.Env(), err.what());

        return info.Env().Undefined();
    }
}

Napi::Value CoreWrap::compile_model_sync(const Napi::CallbackInfo& info,
                                         const Napi::Object& model,
                                         const Napi::String& device) {
    const auto model_prototype = info.Env().GetInstanceData<AddonData>()->model_prototype;
    if (model_prototype && model.InstanceOf(model_prototype->Value().As<Napi::Function>())) {
        const auto m = Napi::ObjectWrap<ModelWrap>::Unwrap(model);
        const auto& compiled_model = _core.compile_model(m->get_model(), device);
        return CompiledModelWrap::wrap(info.Env(), compiled_model);
    } else {
        reportError(info.Env(), "Cannot create Model from Napi::Object.");
        return info.Env().Undefined();
    }
}

Napi::Value CoreWrap::compile_model_sync(const Napi::CallbackInfo& info,
                                         const Napi::String& model_path,
                                         const Napi::String& device) {
    const auto& compiled_model = _core.compile_model(model_path, device);
    return CompiledModelWrap::wrap(info.Env(), compiled_model);
}

Napi::Value CoreWrap::compile_model_sync(const Napi::CallbackInfo& info,
                                         const Napi::Object& model_obj,
                                         const Napi::String& device,
                                         const std::map<std::string, ov::Any>& config) {
    const auto& mw = Napi::ObjectWrap<ModelWrap>::Unwrap(model_obj);
    const auto& compiled_model = _core.compile_model(mw->get_model(), info[1].ToString(), config);
    return CompiledModelWrap::wrap(info.Env(), compiled_model);
}

Napi::Value CoreWrap::compile_model_sync(const Napi::CallbackInfo& info,
                                         const Napi::String& model_path,
                                         const Napi::String& device,
                                         const std::map<std::string, ov::Any>& config) {
    const auto& compiled_model = _core.compile_model(model_path, device, config);
    return CompiledModelWrap::wrap(info.Env(), compiled_model);
}

Napi::Value CoreWrap::compile_model_sync_dispatch(const Napi::CallbackInfo& info) {
    try {
        if (info.Length() == 2 && info[0].IsString() && info[1].IsString()) {
            return compile_model_sync(info, info[0].ToString(), info[1].ToString());
        } else if (info.Length() == 2 && info[0].IsObject() && info[1].IsString()) {
            return compile_model_sync(info, info[0].ToObject(), info[1].ToString());
        } else if (info.Length() == 3 && info[0].IsString() && info[1].IsString()) {
            const auto& config = js_to_cpp<std::map<std::string, ov::Any>>(info, 2, {napi_object});
            return compile_model_sync(info, info[0].ToString(), info[1].ToString(), config);
        } else if (info.Length() == 3 && info[0].IsObject() && info[1].IsString()) {
            const auto& config = js_to_cpp<std::map<std::string, ov::Any>>(info, 2, {napi_object});
            return compile_model_sync(info, info[0].ToObject(), info[1].ToString(), config);
        } else if (info.Length() < 2 || info.Length() > 3) {
            reportError(info.Env(), "Invalid number of arguments -> " + std::to_string(info.Length()));
            return info.Env().Undefined();
        } else {
            reportError(info.Env(), "Error while compiling model.");
            return info.Env().Undefined();
        }
    } catch (std::exception& e) {
        reportError(info.Env(), e.what());
        return info.Env().Undefined();
    }
}

void FinalizerCallbackModel(Napi::Env env, void* finalizeData, TsfnContextModel* context) {
    context->nativeThread.join();
    delete context;
};

void FinalizerCallbackPath(Napi::Env env, void* finalizeData, TsfnContextPath* context) {
    context->nativeThread.join();
    delete context;
};

void compileModelThreadModel(TsfnContextModel* context) {
    ov::Core core;
    context->_compiled_model = core.compile_model(context->_model, context->_device, context->_config);

    auto callback = [](Napi::Env env, Napi::Function, TsfnContextModel* context) {
        Napi::HandleScope scope(env);
        auto obj = CompiledModelWrap::get_class_constructor(env).New({});
        auto cm = Napi::ObjectWrap<CompiledModelWrap>::Unwrap(obj);
        cm->set_compiled_model(context->_compiled_model);

        context->deferred.Resolve(obj);
    };

    context->tsfn.BlockingCall(context, callback);
    context->tsfn.Release();
}

void compileModelThreadPath(TsfnContextPath* context) {
    ov::Core core;
    context->_compiled_model = core.compile_model(context->_model, context->_device, context->_config);

    auto callback = [](Napi::Env env, Napi::Function, TsfnContextPath* context) {
        Napi::HandleScope scope(env);
        auto obj = CompiledModelWrap::get_class_constructor(env).New({});
        auto cm = Napi::ObjectWrap<CompiledModelWrap>::Unwrap(obj);
        cm->set_compiled_model(context->_compiled_model);

        context->deferred.Resolve(obj);
    };

    context->tsfn.BlockingCall(context, callback);
    context->tsfn.Release();
}

Napi::Value CoreWrap::compile_model_async(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    if (info[0].IsObject() && info[1].IsString()) {
        auto context_data = new TsfnContextModel(env);
        auto m = Napi::ObjectWrap<ModelWrap>::Unwrap(info[0].ToObject());
        context_data->_model = m->get_model()->clone();
        context_data->_device = info[1].ToString();

        if (info.Length() == 3) {
            try {
                context_data->_config = js_to_cpp<std::map<std::string, ov::Any>>(info, 2, {napi_object});
            } catch (std::exception& e) {
                reportError(env, e.what());
            }
        }

        context_data->tsfn = Napi::ThreadSafeFunction::New(env,
                                                           Napi::Function(),
                                                           "TSFN",
                                                           0,
                                                           1,
                                                           context_data,
                                                           FinalizerCallbackModel,
                                                           (void*)nullptr);

        context_data->nativeThread = std::thread(compileModelThreadModel, context_data);
        return context_data->deferred.Promise();
    } else if (info[0].IsString() && info[1].IsString()) {
        auto context_data = new TsfnContextPath(env);
        context_data->_model = info[0].ToString();
        context_data->_device = info[1].ToString();

        if (info.Length() == 3) {
            try {
                context_data->_config = js_to_cpp<std::map<std::string, ov::Any>>(info, 2, {napi_object});
            } catch (std::exception& e) {
                reportError(env, e.what());
            }
        }

        context_data->tsfn = Napi::ThreadSafeFunction::New(env,
                                                           Napi::Function(),
                                                           "TSFN",
                                                           0,
                                                           1,
                                                           context_data,
                                                           FinalizerCallbackPath,
                                                           (void*)nullptr);

        context_data->nativeThread = std::thread(compileModelThreadPath, context_data);
        return context_data->deferred.Promise();
    } else if (info.Length() < 2 || info.Length() > 3) {
        reportError(info.Env(), "Invalid number of arguments -> " + std::to_string(info.Length()));
        return Napi::Value();
    } else {
        reportError(info.Env(), "Error while compiling model.");
        return Napi::Value();
    }
}
