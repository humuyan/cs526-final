#include "conv.h"
#include <iostream>

int main() {
	int n = 16, c = 128, h = 64, w = 64, f = 128, r = 3;
	std::cout << profile_conv(n, c, h, w, f, r, 1, 1, 1, 1, 0, 0) << std::endl;
	return 0;
}