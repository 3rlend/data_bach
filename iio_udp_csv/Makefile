CXX     ?= g++
CXXFLAGS += -Wall -O2 -std=c++17
LDFLAGS  += -liio -lpthread

TARGET = ny_server
SRC    = ny_server.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
