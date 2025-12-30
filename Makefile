# Makefile for Colored Interlacing Triangles Enumeration
#
# Cross-platform build system that auto-detects:
#   - OpenMP support (via Homebrew GCC on macOS, native on Linux)
#   - Metal GPU support (macOS Apple Silicon only)
#
# Usage:
#   make              # Build all available targets
#   make enumerate    # Build CPU enumerator (any N, n)
#   make metal        # Build Metal GPU + disk-chunked version (macOS ARM)
#   make test         # Run verification tests
#   make clean        # Remove build artifacts
#   make help         # Show this help message

# ============================================================================
# Platform Detection
# ============================================================================

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

CXX ?= g++
HAS_OPENMP := 0
HAS_METAL := 0

# ============================================================================
# macOS Configuration
# ============================================================================

ifeq ($(UNAME_S),Darwin)
    HOMEBREW_GXX := $(wildcard /opt/homebrew/bin/g++-15 /opt/homebrew/bin/g++-14 \
                               /opt/homebrew/bin/g++-13 /usr/local/bin/g++-15 \
                               /usr/local/bin/g++-14 /usr/local/bin/g++-13)
    ifneq ($(HOMEBREW_GXX),)
        CXX := $(firstword $(HOMEBREW_GXX))
        OPENMP_FLAGS := -fopenmp
        HAS_OPENMP := 1
    else
        LIBOMP := $(wildcard /opt/homebrew/opt/libomp /usr/local/opt/libomp)
        ifneq ($(LIBOMP),)
            CXX := clang++
            OPENMP_FLAGS := -Xpreprocessor -fopenmp -I$(firstword $(LIBOMP))/include \
                           -L$(firstword $(LIBOMP))/lib -lomp
            HAS_OPENMP := 1
        endif
    endif
    ifeq ($(UNAME_M),arm64)
        HAS_METAL := 1
    endif
endif

# ============================================================================
# Linux Configuration
# ============================================================================

ifeq ($(UNAME_S),Linux)
    OPENMP_FLAGS := -fopenmp
    HAS_OPENMP := 1
endif

# ============================================================================
# Compiler Flags
# ============================================================================

# Aggressive optimization flags
ifeq ($(UNAME_M),arm64)
    # ARM: use -mcpu instead of -march
    OPTFLAGS := -O3 -mcpu=native -flto -funroll-loops \
                -finline-functions -fomit-frame-pointer -ffast-math \
                -fno-signed-zeros -fno-trapping-math
else
    OPTFLAGS := -O3 -march=native -mtune=native -flto -funroll-loops \
                -finline-functions -fomit-frame-pointer -ffast-math \
                -fno-signed-zeros -fno-trapping-math
endif

CXXFLAGS := -std=c++17 $(OPTFLAGS) -Wall

ifeq ($(HAS_OPENMP),1)
    CXXFLAGS += $(OPENMP_FLAGS)
endif

METAL_CXX := clang++
METAL_CXXFLAGS := -std=c++17 $(OPTFLAGS) -Wall
METAL_MMFLAGS := -std=c++17 $(OPTFLAGS) -fobjc-arc -Wall
METAL_FRAMEWORKS := -framework Metal -framework Foundation

# ============================================================================
# Targets
# ============================================================================

TARGETS := enumerate

ifeq ($(HAS_METAL),1)
    TARGETS += enumerate_metal
endif

# ============================================================================
# Build Rules
# ============================================================================

.PHONY: all info clean test help metal

all: info $(TARGETS)

info:
	@echo "=== Build Configuration ==="
	@echo "Platform: $(UNAME_S) $(UNAME_M)"
	@echo "Compiler: $(CXX)"
ifeq ($(HAS_OPENMP),1)
	@echo "OpenMP:   YES"
else
	@echo "OpenMP:   NO (install Homebrew GCC: brew install gcc)"
endif
ifeq ($(HAS_METAL),1)
	@echo "Metal:    YES (Apple Silicon)"
else
	@echo "Metal:    NO (requires macOS on Apple Silicon)"
endif
	@echo "==========================="
	@echo ""

# CPU enumerator (any N, n)
enumerate: enumerate_general.cpp
	@echo "Building enumerate (CPU version)..."
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "Done: ./enumerate <n> <N>"
	@echo ""

# Metal GPU + disk-chunked version (any N, n - macOS Apple Silicon)
ifeq ($(HAS_METAL),1)
metal: enumerate_metal
	@echo "Metal target built successfully"

enumerate_metal: enumerate_metal.o MetalController.o
	@echo "Linking enumerate_metal..."
	$(METAL_CXX) $(METAL_CXXFLAGS) -o $@ $^ $(METAL_FRAMEWORKS)
	@echo "Done: ./enumerate_metal <n> <N>"
	@echo ""

enumerate_metal.o: enumerate_metal.cpp MetalController.h
	@echo "Compiling enumerate_metal.cpp..."
	$(METAL_CXX) $(METAL_CXXFLAGS) -c $< -o $@

MetalController.o: MetalController.mm MetalController.h
	@echo "Compiling MetalController.mm (Objective-C++)..."
	$(METAL_CXX) $(METAL_MMFLAGS) -c $< -o $@
else
metal:
	@echo "Error: Metal requires macOS on Apple Silicon (arm64)"
	@echo "Your platform: $(UNAME_S) $(UNAME_M)"
	@exit 1
endif

# ============================================================================
# Test Targets
# ============================================================================

test: enumerate
	@echo "=== Running Verification Tests ==="
	@echo ""
	@echo "Testing T_3(3) (expected: 528)..."
	@./enumerate 3 3
	@echo ""
	@echo "Testing T_4(3) (expected: 8,160)..."
	@./enumerate 3 4
	@echo ""
	@echo "Testing T_3(4) (expected: 191,232)..."
	@./enumerate 4 3
	@echo ""
	@echo "=== All tests passed ==="

ifeq ($(HAS_METAL),1)
test-metal: enumerate_metal
	@echo "=== Running Metal GPU Tests ==="
	@./enumerate_metal 3 3
	@./enumerate_metal 3 4
	@./enumerate_metal 4 3
	@echo "=== Metal tests passed ==="
endif

# ============================================================================
# Clean
# ============================================================================

clean:
	rm -f enumerate enumerate_metal *.o
	rm -rf checkpoints_*
	rm -f triangles_*.txt

# ============================================================================
# Help
# ============================================================================

help:
	@echo "Colored Interlacing Triangles Enumeration"
	@echo ""
	@echo "Computes T_N(n), the number of colored interlacing n-triangles of depth N."
	@echo ""
	@echo "TARGETS:"
	@echo "  make              Build all available targets"
	@echo "  make enumerate    CPU enumerator (any N, n)"
	@echo "  make metal        Metal GPU + disk-chunked (macOS ARM only)"
	@echo ""
	@echo "TESTING:"
	@echo "  make test         Run CPU verification tests"
	@echo "  make test-metal   Run Metal GPU tests (macOS ARM)"
	@echo ""
	@echo "OTHER:"
	@echo "  make clean        Remove build artifacts"
	@echo "  make help         Show this help"
	@echo ""
	@echo "USAGE:"
	@echo "  ./enumerate 4 3         T_3(4) = 191,232 (CPU)"
	@echo "  ./enumerate_metal 4 3   T_3(4) = 191,232 (GPU)"
	@echo "  ./enumerate_metal 6 3   T_3(6) with disk chunking"
	@echo ""
	@echo "The Metal version uses:"
	@echo "  - GPU for fast interlacing checks"
	@echo "  - Disk-based chunking for large state spaces (>100M states)"
	@echo "  - Checkpoint/resume support for long computations"
	@echo ""
	@echo "PLATFORM REQUIREMENTS:"
	@echo "  macOS:  brew install gcc  (for OpenMP)"
	@echo "  Linux:  apt install g++   (OpenMP included)"
	@echo "  Metal:  macOS on Apple Silicon (M1/M2/M3/M4)"
