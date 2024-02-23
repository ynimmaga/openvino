// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_plugin_cache.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <unordered_map>

#include "common_test_utils/file_utils.hpp"
#include "openvino/util/file_util.hpp"

namespace ov {
namespace test {
namespace utils {
namespace {
class TestListener : public testing::EmptyTestEventListener {
public:
    void OnTestEnd(const testing::TestInfo& testInfo) override {
        if (auto testResult = testInfo.result()) {
            if (testResult->Failed()) {
                PluginCache::get().reset();
            }
        }
    }
};
}  // namespace

PluginCache& PluginCache::get() {
    static PluginCache instance;
    return instance;
}

std::shared_ptr<ov::Core> PluginCache::core(const std::string& deviceToCheck) {
    if (std::getenv("DISABLE_PLUGIN_CACHE") != nullptr) {
#ifndef NDEBUG
        std::cout << "'DISABLE_PLUGIN_CACHE' environment variable is set. New Core object will be created!"
                  << std::endl;
#endif
        return std::make_shared<ov::Core>();
    }

    std::lock_guard<std::mutex> lock(g_mtx);
#ifndef NDEBUG
    std::cout << "Access PluginCache ov core. OV Core use count: " << ov_core.use_count() << std::endl;
#endif

    if (!ov_core) {
#ifndef NDEBUG
        std::cout << "Created ov core." << std::endl;
#endif
        ov_core = std::make_shared<ov::Core>();
        assert(0 != ov_core.use_count());

        // Register Template plugin as a reference provider
        const auto devices = ov_core->get_available_devices();
        if (std::find(devices.begin(), devices.end(), std::string(ov::test::utils::DEVICE_TEMPLATE)) == devices.end()) {
            auto plugin_path =
                ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                                   std::string(ov::test::utils::TEMPLATE_LIB) + OV_BUILD_POSTFIX);
            if (!ov::util::file_exists(plugin_path)) {
                throw std::runtime_error("Plugin: " + plugin_path + " does not exists!");
            }
            ov_core->register_plugin(plugin_path, ov::test::utils::DEVICE_TEMPLATE);
        }

        if (!deviceToCheck.empty()) {
            auto properties = ov_core->get_property(deviceToCheck, ov::supported_properties);

            if (std::find(properties.begin(), properties.end(), ov::available_devices) != properties.end()) {
                const auto availableDevices = ov_core->get_property(deviceToCheck, ov::available_devices);

                if (availableDevices.empty()) {
                    std::cerr << "No available devices for " << deviceToCheck << std::endl;
                    std::exit(EXIT_FAILURE);
                }
#ifndef NDEBUG
                std::cout << "Available devices for " << deviceToCheck << ":" << std::endl;

                for (const auto& device : availableDevices) {
                    std::cout << "    " << device << std::endl;
                }
#endif
            }
        }
    }

    return ov_core;
}

void PluginCache::reset() {
    std::lock_guard<std::mutex> lock(g_mtx);

#ifndef NDEBUG
    std::cout << "Reset PluginCache. OV Core use count: " << ov_core.use_count() << std::endl;
#endif

    ov_core.reset();
}

PluginCache::PluginCache() {
    auto& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new TestListener);
}
}  // namespace utils
}  // namespace test
}  // namespace ov
