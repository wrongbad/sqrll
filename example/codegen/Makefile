

.PHONY: model.js
model.js: artifacts/model.js

artifacts/model.js : cgen_math.h artifacts/model.cpp
	em++ -O3 artifacts/model.cpp -o artifacts/model.js -s MODULARIZE=1 -s EXPORT_NAME=load_model -s EXPORTED_FUNCTIONS=_model_step,_model_reset

.PHONY: model_cli
model_cli: artifacts/model_cli

artifacts/model_cli : cgen_math.h cgen_cli.cpp artifacts/model.cpp 
	c++ -std=c++17 -O3 -march=native -ffast-math artifacts/model.cpp cgen_cli.cpp -o artifacts/model_cli