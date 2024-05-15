#include <stdio.h>
#include <math.h>

int main() {
	float a = 0.0f, b = 0.0f;

	a = 0.00000000001f;
	b = a + 0.00000000001f;

	if (a == (b-0.00000000001f)) {
		printf("true\n");
	} else {
		printf("false\n");
	}
	printf("%.12f", a);

	return 0;
}
