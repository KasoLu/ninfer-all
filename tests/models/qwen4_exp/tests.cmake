ninfer_add_test(ninfer_qwen4_exp_ngram_hash_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_hash.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen4_exp_ngram_table_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_table.cpp"
  LIBRARIES ninfer_model_loading)
