all:
	pgc++ -acc -acc=host -Minfo=all -o cpu_host cpu_host.cpp -lboost_program_options
	pgc++ -acc -acc=multicore -Minfo=all -o cpu_multicore cpu_multicore.cpp -lboost_program_options
	#pgc++ -acc -acc=gpu -Minfo=all -o gpu gpu.cpp -lboost_program_options -I/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/include

clean:
	rm -f cpu_host
	rm -f cpu_multicore
	#rm -f gpu
