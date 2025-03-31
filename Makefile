# === Config ===
NVCC = nvcc
CXX = g++
CXXFLAGS = -I./module
NVCCFLAGS = -I./module

# === Paths ===
MODULE_DIR = module
TEST_DIR = test
OBJ_DIR = build

# === Files ===
LINEAR_OBJ = $(OBJ_DIR)/Linear.o
TEST_OBJ = $(OBJ_DIR)/test_linear.o
XOR_OBJ = $(OBJ_DIR)/train_xor.o

EXEC_TEST = test_linear
EXEC_XOR = train_xor

# === Targets ===
all: $(EXEC_TEST) $(EXEC_XOR)

$(EXEC_TEST): $(LINEAR_OBJ) $(TEST_OBJ)
	$(NVCC) -o $@ $^

$(EXEC_XOR): $(LINEAR_OBJ) $(XOR_OBJ)
	$(NVCC) -o $@ $^

$(LINEAR_OBJ): $(MODULE_DIR)/Linear.cu
	mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TEST_OBJ): $(TEST_DIR)/test_linear.cpp
	mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(XOR_OBJ): $(TEST_DIR)/train_xor.cpp
	mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: $(EXEC_TEST)
	./$(EXEC_TEST)

xor: $(EXEC_XOR)
	./$(EXEC_XOR)

clean:
	rm -rf $(OBJ_DIR) $(EXEC_TEST) $(EXEC_XOR)
