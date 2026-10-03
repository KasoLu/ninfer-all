find_package(Python3 REQUIRED COMPONENTS Interpreter)

include(cmake/NinferTests.cmake)
include(cmake/CoreTests.cmake)
include(artifact/tests.cmake)
include(cmake/RuntimeTests.cmake)
include(cmake/ProductTests.cmake)
include(cmake/BenchSupportTests.cmake)
include(models/qwen3_5/tests.cmake)
include(models/qwen4_exp/tests.cmake)
include(ops/tests.cmake)

ninfer_finalize_bundle(ninfer_tests)

# Without a usable CUDA device -- no GPU, no driver, or only the toolkit's stub driver, as on a CI
# runner -- every test that needs one fails on its first CUDA call with one of these runtime
# messages. Report those as skipped instead, so such a box still runs and judges every host test.
# None of the messages can occur where a device is usable.
get_property(_ninfer_all_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_ninfer_all_tests} PROPERTIES SKIP_REGULAR_EXPRESSION
  "CUDA driver is a stub library;no CUDA-capable device is detected;CUDA driver version is insufficient for CUDA runtime version")

# Every test executable shares one per-config runtime directory; stage the FFmpeg/curl DLL closure
# there once so they resolve their imports (the Windows loader only searches the executable's own
# directory, the system directories and PATH).
ninfer_stage_test_runtime_dlls(ninfer_tests)
