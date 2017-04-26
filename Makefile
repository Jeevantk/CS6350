all:
	g++ -o stitch  stitching_detailed.cpp `pkg-config --cflags --libs opencv` 