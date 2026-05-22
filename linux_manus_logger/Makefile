CXX := g++
CXXFLAGS := -std=c++17 -O2 -g -Wall -Wextra

MANUS_SDK_DIR ?= ./ManusSDK
INCLUDES := -I$(MANUS_SDK_DIR)/include
LDFLAGS := -L$(MANUS_SDK_DIR)/lib -lManusSDK_Integrated -lusb-1.0 -ludev -pthread -Wl,-rpath,$(MANUS_SDK_DIR)/lib

TARGET := manus_integrated_logger
SRC := manus_integrated_logger.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
