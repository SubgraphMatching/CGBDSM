#include "graph.cuh"
#include "gpu_error_check.cuh"
#include <cuda_profiler_api.h>
#include <../shared/timer.hpp>
#include "ligraUtils.h"
#include <unistd.h>
#define WEIGHT_CAP 5 

template <class E>
Graph<E>::Graph(string graphFilePath, bool isWeighted)
{
	this->graphFilePath = graphFilePath;
	this->isWeighted = isWeighted;
}

template <class E>
string Graph<E>::GetFileExtension(string fileName)
{
    if(fileName.find_last_of(".") != string::npos)
        return fileName.substr(fileName.find_last_of(".")+1);
    return "";
}

template <>
void Graph<OutEdgeWeighted>::AssignW8(uint w8, uint index)
{
    edgeList[index].w8 = w8;
}

template <>
void Graph<OutEdge>::AssignW8(uint w8, uint index)
{
    edgeList[index].end = edgeList[index].end; // do nothing
}

template <class E>
void Graph<E>::ReadGraph()
{

	cout << "Reading the input graph from the following file:\n>> " << graphFilePath << endl;
	
	this->graphFormat = GetFileExtension(graphFilePath);
	
	if(graphFormat == "bcsr" || graphFormat == "bwcsr")
	{
		ifstream infile (graphFilePath, ios::in | ios::binary);
	
		infile.read ((char*)&num_nodes, sizeof(uint));
		infile.read ((char*)&num_edges, sizeof(uint));
		
		nodePointer = new uint[num_nodes+1]();
		gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));
		
		infile.read ((char*)nodePointer, sizeof(uint)*num_nodes);
		infile.read ((char*)edgeList, sizeof(E)*num_edges);
		nodePointer[num_nodes] = num_edges;
	}
	else if(graphFormat == "el" || graphFormat == "wel")
	{
		ifstream infile;
		infile.open(graphFilePath);
		stringstream ss;
		uint max = 0;
		string line;
		uint edgeCounter = 0;
		if(isWeighted)
		{
			vector<EdgeWeighted> edges;
			EdgeWeighted newEdge;
			getline( infile, line ); // 注意注意注意！！！！！！
			while(getline( infile, line ))
			{
				ss.str("");
				ss.clear();
				ss << line;
				
				ss >> newEdge.source;
				ss >> newEdge.end;
				// ss >> newEdge.w8;
				newEdge.w8 = 1;

				edges.push_back(newEdge);
				edgeCounter++;
				
				if(max < newEdge.source)
					max = newEdge.source;
				if(max < newEdge.end)
					max = newEdge.end;				
			}
			infile.close();
			num_nodes = max + 1;
			num_edges = edgeCounter;
			nodePointer = new uint[num_nodes+1]();
			gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));
			uint *degree = new uint[num_nodes]();
			for(uint i=0; i<num_nodes; i++)
				degree[i] = 0;
			for(uint i=0; i<num_edges; i++)
				degree[edges[i].source]++;
			
			uint counter=0;
			for(uint i=0; i<num_nodes; i++)
			{
				nodePointer[i] = counter;
				counter = counter + degree[i];
			}
			nodePointer[num_nodes] = num_edges;
			uint *outDegreeCounter  = new uint[num_nodes]();
			uint location;  
			for(uint i=0; i<num_edges; i++)
			{
				location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
				edgeList[location].end = edges[i].end;
				if(isWeighted)
					AssignW8(edges[i].w8, location);
					//edgeList[location].w8 = edges[i].w8;
				outDegreeCounter[edges[i].source]++;  
			}
			edges.clear();
			delete[] degree;
			delete[] outDegreeCounter;
			
		}
		else
		{
			vector<Edge> edges;
			Edge newEdge;
			while(getline( infile, line ))
			{
				ss.str("");
				ss.clear();
				ss << line;
				
				ss >> newEdge.source;
				ss >> newEdge.end;
				
				edges.push_back(newEdge);
				edgeCounter++;
				
				if(max < newEdge.source)
					max = newEdge.source;
				if(max < newEdge.end)
					max = newEdge.end;				
			}
			infile.close();
			num_nodes = max + 1;
			num_edges = edgeCounter;
			nodePointer = new uint[num_nodes+1]();
			gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));
			uint *degree = new uint[num_nodes]();
			for(uint i=0; i<num_nodes; i++)
				degree[i] = 0;
			for(uint i=0; i<num_edges; i++)
				degree[edges[i].source]++;
			
			uint counter=0;
			for(uint i=0; i<num_nodes; i++)
			{
				nodePointer[i] = counter;
				counter = counter + degree[i];
			}
			nodePointer[num_nodes] = num_edges;
			uint *outDegreeCounter  = new uint[num_nodes]();
			uint location;  
			for(uint i=0; i<num_edges; i++)
			{
				location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
				edgeList[location].end = edges[i].end;
				//if(isWeighted)
				//	edgeList[location].w8 = edges[i].w8;
				outDegreeCounter[edges[i].source]++;  
			}
			edges.clear();
			delete[] degree;
			delete[] outDegreeCounter;						
		}
	}
	else
	{
		cout << "The graph format is not supported!\n";
		exit(-1);
	}
	
	outDegree  = new unsigned int[num_nodes]();
	
	for(uint i=1; i<num_nodes-1; i++)
		outDegree[i-1] = nodePointer[i] - nodePointer[i-1];
	outDegree[num_nodes-1] = num_edges - nodePointer[num_nodes-1];
	
	label1 = new bool[num_nodes]();
	label2 = new bool[num_nodes]();
	value  = new unsigned int[num_nodes]();
	
	gpuErrorcheck(cudaMalloc(&d_outDegree, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_value, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_label1, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label2, num_nodes * sizeof(bool)));
	
	cout << "Done reading.\n";
	cout << "Number of nodes = " << num_nodes << endl;
	cout << "Number of edges = " << num_edges << endl;


}

//--------------------------------------

template <class E>
GraphPR<E>::GraphPR(string graphFilePath, bool isWeighted)
{
	this->graphFilePath = graphFilePath;
	this->isWeighted = isWeighted;
}

template <class E>
string GraphPR<E>::GetFileExtension(string fileName)
{
    if(fileName.find_last_of(".") != string::npos)
        return fileName.substr(fileName.find_last_of(".")+1);
    return "";
}

template <>
void GraphPR<OutEdgeWeighted>::AssignW8(uint w8, uint index)
{
    edgeList[index].w8 = w8;
}

template <>
void GraphPR<OutEdge>::AssignW8(uint w8, uint index)
{
    edgeList[index].end = edgeList[index].end; // do nothing
}

template <class E>
void GraphPR<E>::ReadGraph()
{

	cout << "Reading the input graph from the following file:\n>> " << graphFilePath << endl;
	
	this->graphFormat = GetFileExtension(graphFilePath);
	
	if(graphFormat == "bcsr" || graphFormat == "bwcsr")
	{
		ifstream infile (graphFilePath, ios::in | ios::binary);
	
		infile.read ((char*)&num_nodes, sizeof(uint));
		infile.read ((char*)&num_edges, sizeof(uint));
		
		nodePointer = new uint[num_nodes+1]();
		gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));
		
		infile.read ((char*)nodePointer, sizeof(uint)*num_nodes);
		infile.read ((char*)edgeList, sizeof(E)*num_edges);
		nodePointer[num_nodes] = num_edges;
	}
	else if(graphFormat == "el" || graphFormat == "wel")
	{
		ifstream infile;
		infile.open(graphFilePath);
		stringstream ss;
		uint max = 0;
		string line;
		uint edgeCounter = 0;
		if(isWeighted)
		{
			vector<EdgeWeighted> edges;
			EdgeWeighted newEdge;
			while(getline( infile, line ))
			{
				ss.str("");
				ss.clear();
				ss << line;
				
				ss >> newEdge.source;
				ss >> newEdge.end;
				ss >> newEdge.w8;
				
				edges.push_back(newEdge);
				edgeCounter++;
				
				if(max < newEdge.source)
					max = newEdge.source;
				if(max < newEdge.end)
					max = newEdge.end;				
			}
			infile.close();
			num_nodes = max + 1;
			num_edges = edgeCounter;
			nodePointer = new uint[num_nodes+1]();
			gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));
			uint *degree = new uint[num_nodes]();
			for(uint i=0; i<num_nodes; i++)
				degree[i] = 0;
			for(uint i=0; i<num_edges; i++)
				degree[edges[i].source]++;
			
			uint counter=0;
			for(uint i=0; i<num_nodes; i++)
			{
				nodePointer[i] = counter;
				counter = counter + degree[i];
			}
			nodePointer[num_nodes] = num_edges;
			uint *outDegreeCounter  = new uint[num_nodes]();
			uint location;  
			for(uint i=0; i<num_edges; i++)
			{
				location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
				edgeList[location].end = edges[i].end;
				if(isWeighted)
					AssignW8(edges[i].w8, location);
					//edgeList[location].w8 = edges[i].w8;
				outDegreeCounter[edges[i].source]++;  
			}
			edges.clear();
			delete[] degree;
			delete[] outDegreeCounter;
			
		}
		else
		{
			vector<Edge> edges;
			Edge newEdge;
			while(getline( infile, line ))
			{
				ss.str("");
				ss.clear();
				ss << line;
				
				ss >> newEdge.source;
				ss >> newEdge.end;

				edges.push_back(newEdge);
				edgeCounter++;
				
				if(max < newEdge.source)
					max = newEdge.source;
				if(max < newEdge.end)
					max = newEdge.end;				
			}
			infile.close();
			num_nodes = max + 1;
			num_edges = edgeCounter;
			nodePointer = new uint[num_nodes+1]();
			gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));
			uint *degree = new uint[num_nodes]();
			for(uint i=0; i<num_nodes; i++)
				degree[i] = 0;
			for(uint i=0; i<num_edges; i++)
				degree[edges[i].source]++;
			
			uint counter=0;
			for(uint i=0; i<num_nodes; i++)
			{
				nodePointer[i] = counter;
				counter = counter + degree[i];
			}
			nodePointer[num_nodes] = num_edges;
			uint *outDegreeCounter  = new uint[num_nodes]();
			uint location;  
			for(uint i=0; i<num_edges; i++)
			{
				location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
				edgeList[location].end = edges[i].end;
				//if(isWeighted)
				//	edgeList[location].w8 = edges[i].w8;
				outDegreeCounter[edges[i].source]++;  
			}
			edges.clear();
			delete[] degree;
			delete[] outDegreeCounter;						
		}
	}
	else
	{
		cout << "The graph format is not supported!\n";
		exit(-1);
	}
	
	outDegree  = new unsigned int[num_nodes]();
	
	for(uint i=1; i<num_nodes-1; i++)
		outDegree[i-1] = nodePointer[i] - nodePointer[i-1];
	outDegree[num_nodes-1] = num_edges - nodePointer[num_nodes-1];
	

	value  = new float[num_nodes]();
	delta  = new float[num_nodes]();
	
	gpuErrorcheck(cudaMalloc(&d_outDegree, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_value, num_nodes * sizeof(float)));
	gpuErrorcheck(cudaMalloc(&d_delta, num_nodes * sizeof(float)));
	
	
	cout << "Done reading.\n";
	cout << "Number of nodes = " << num_nodes << endl;
	cout << "Number of edges = " << num_edges << endl;
	

}

template <class E>
GraphGPUMultiGPMA<E>::GraphGPUMultiGPMA(string graphFilePath, bool isWeighted)
{
	this->graphFilePath = graphFilePath;
	this->isWeighted = isWeighted;
	this->num_nodes = 0;
	this->num_edges = 0;
}

template <class E>
GraphGPUMultiGPMA<E>::GraphGPUMultiGPMA(string graphFilePath, string graphFileChangePath, bool isWeighted, uint batchSize)
{
	this->graphFilePath = graphFilePath;
	this->graphFileChangePath = graphFileChangePath;
	this->isWeighted = isWeighted;
	this->batchSize = batchSize;
	this->num_nodes = 0;
	this->num_edges = 0;
}

template <class E>
GraphGPUMultiGPMA<E>::GraphGPUMultiGPMA(string graphFilePath, string graphFileChangePath, bool isWeighted, uint batchSize, int node_per_block, int degree_upper)
{
	this->graphFilePath = graphFilePath;
	this->graphFileChangePath = graphFileChangePath;
	this->isWeighted = isWeighted;
	this->batchSize = batchSize;
	this->num_nodes = 0;
	this->num_edges = 0;
	this->node_per_block = node_per_block;
	this->degree_upper = degree_upper;
}

template <class E>
GraphGPUMultiGPMA<E>::~GraphGPUMultiGPMA()
{
}

template <class E>
float GraphGPUMultiGPMA<E>::ReadGraphChange(bool delta_calc)
{
	static ifstream infile;
	static FILE* file = nullptr;
	static FILE* file1 = nullptr;
	if(!file) {
		file = fopen("gpma_time_1.csv", "aw");
		file1 = fopen("gpma_mem_1.csv", "w");
		fprintf(file, "%s,%d\n", this->graphFilePath.c_str(), this->batchSize);
		fprintf(file, "update_time, level\n");
    fprintf(file1, "gpma_mem_usage(MB), gpu_mem_usage(MB)\n");
	}

	if(!infile.is_open()) {
		infile.open(graphFileChangePath);
	}
	stringstream ss;
	string line;
	int x, y;
	string op;

	if(delta_calc) {
		edge_addtions.clear();
		edge_addtions_values.clear();
		edge_deltions.clear();
		d_edge_addtions.clear();
		d_edge_addtions_values.clear();
		d_edge_deltions.clear();
	}

	KEY_TYPE *h_base_keys;
	VALUE_TYPE *h_base_values;
	SIZE_TYPE *batch_edge_num_tmp;
	h_base_keys = new KEY_TYPE[batchSize];
	h_base_values = new VALUE_TYPE[batchSize];
	batch_edge_num_tmp = new SIZE_TYPE[multi_gpma->num_blocks];
	parallel_cpu_for(uint i = 0; i < multi_gpma->num_blocks; i++) {
		batch_edge_num_tmp[i] = 0;
	}
	for(uint i = 0; i < batchSize; i++)
	{
		getline(infile, line);

		ss.str("");
		ss.clear();
		ss << line;
		
		ss >> op;
		ss >> x;
		ss >> y;
		int block_id = multi_gpma->h_node2block[x];

		if( op == "a" ) {
			outDegree[x]++;
			h_base_keys[i] = ((KEY_TYPE) ((KEY_TYPE)x) << 32) + y;
			h_base_values[i] = (VALUE_TYPE)((x + y) % WEIGHT_CAP + 1);
			batch_edge_num_tmp[block_id]++;
			if(delta_calc) {
				edge_addtions.push_back(h_base_keys[i]);
				edge_addtions_values.push_back(h_base_values[i]);
			}
			num_edges++;
		} else if( op == "d" ) {
			outDegree[x]--;
			h_base_keys[i] = ((KEY_TYPE) ((KEY_TYPE)x) << 32) + y;
			h_base_values[i] = VALUE_NONE;
			batch_edge_num_tmp[block_id]--;
			if(delta_calc) {
				edge_deltions.push_back(h_base_keys[i]);
			}
			num_edges--;
		}
	}

	d_edge_addtions = edge_addtions;
	d_edge_addtions_values = edge_addtions_values;
	d_edge_deltions = edge_deltions;

	parallel_cpu_for(uint i = 0; i < multi_gpma->num_blocks; i++) {
		multi_gpma->block_edge_num[i] += batch_edge_num_tmp[i];
	}

	multi_gpma->update_keys_size = batchSize;
	cErr(cudaMalloc(&multi_gpma->tmp_keys_array, sizeof(KEY_TYPE) * multi_gpma->update_keys_size));
	cErr(cudaMalloc(&multi_gpma->tmp_values_array, sizeof(VALUE_TYPE) * multi_gpma->update_keys_size));
	cErr(cudaMalloc(&multi_gpma->tmp_label_array, sizeof(KEY_TYPE) * multi_gpma->update_keys_size));
	cErr(cudaMalloc(&multi_gpma->tmp_exscan_array, sizeof(KEY_TYPE) * multi_gpma->update_keys_size));

	cErr(cudaMalloc(&multi_gpma->update_keys, sizeof(KEY_TYPE) * multi_gpma->update_keys_size));
	cErr(cudaMalloc(&multi_gpma->update_values, sizeof(VALUE_TYPE) * multi_gpma->update_keys_size));
	cErr(cudaMalloc(&multi_gpma->update_nodes, sizeof(KEY_TYPE) * multi_gpma->update_keys_size));
	cErr(cudaMalloc(&multi_gpma->unique_update_nodes, sizeof(KEY_TYPE) * multi_gpma->update_keys_size));
	cErr(cudaMalloc(&multi_gpma->update_offset, sizeof(SIZE_TYPE) * (multi_gpma->update_keys_size + 1)));

	cudaMemcpy(multi_gpma->update_keys, h_base_keys, batchSize * sizeof(KEY_TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(multi_gpma->update_values, h_base_values, batchSize * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);
	// static SIZE_TYPE batch_time = 0;
	// batch_time++;
	// if(batch_time == 8)
	// 	cudaProfilerStart();
	multi_gpma->single_rebalance_block_batch_time = 0;
	multi_gpma->single_rebalance_kernel_batch_time = 0;
	multi_gpma->single_block_update_size = 0;
	multi_gpma->single_kernel_update_size = 0;
	Timer timer1;
	timer1.Start();
	update_gpma_stage1(multi_gpma);
	resize_gpmas_batch(multi_gpma);
	update_gpma_stage2(multi_gpma);
	multi_gpma->level = 0;
	update_gpma_stage3(multi_gpma);
	update_gpma_stage4(multi_gpma);
	float updating_time = timer1.Finish();
	cudaProfilerStop();
	// if(batch_time == 42)
	// 	exit(0);
	cout << "gpma update edge time " << updating_time / 1000 << endl;
	fprintf(file, "%.8f, %d, %.8f, %.8f, %d, %d, %.8f, %.8f\n", updating_time / 1000, multi_gpma->level, multi_gpma->single_rebalance_block_batch_time, multi_gpma->single_rebalance_kernel_batch_time, multi_gpma->single_block_update_size, multi_gpma->single_kernel_update_size, multi_gpma->rebalance_block_batch_time, multi_gpma->rebalance_kernel_batch_time);
	fflush(file);
	cout << "Done reading.\n";
	cout << "Number of batch = " << batchSize << endl;
	size_t totalBytes, freeBytes;
	cudaError_t err = cudaMemGetInfo(&freeBytes, &totalBytes);
	fprintf(file1, "%.8f, %.8f\n", multi_gpma->mem_usage, (totalBytes - freeBytes) * 1.0 / 1024 / 1024);
	fflush(file1);
	cudaFree(multi_gpma->tmp_label_array);
	cudaFree(multi_gpma->tmp_exscan_array);
	cudaFree(multi_gpma->tmp_keys_array);
	cudaFree(multi_gpma->tmp_values_array);

	cudaFree(multi_gpma->update_keys);
	cudaFree(multi_gpma->update_values);
	cudaFree(multi_gpma->update_nodes);
	cudaFree(multi_gpma->unique_update_nodes);
	cudaFree(multi_gpma->update_offset);

	free(h_base_keys);
	free(h_base_values);
	free(batch_edge_num_tmp);

	gpuErrorcheck(cudaMemset(d_outDegree, 0, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMemset(d_value, 0, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMemset(d_label1, 0, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMemset(d_label2, 0, num_nodes * sizeof(bool)));

	return updating_time;
}

template <class E>
string GraphGPUMultiGPMA<E>::GetFileExtension(string fileName)
{
    if(fileName.find_last_of(".") != string::npos)
        return fileName.substr(fileName.find_last_of(".")+1);
    return "";
}

template <>
void GraphGPUMultiGPMA<OutEdgeWeighted>::AssignW8(uint w8, uint index)
{
}

template <>
void GraphGPUMultiGPMA<OutEdge>::AssignW8(uint w8, uint index)
{
}

template <class E>
void GraphGPUMultiGPMA<E>::ReadGraph()
{
	words W;
	_seq<char> S = readStringFromFile(const_cast<char *>(this->graphFilePath.c_str()));
	W = stringToWords(S.A, S.n);
	
	malloc_trim(0); // 将malloc内存池内存归还

	Timer start1;
	float readtimeing;
	start1.Start();

	Multi_GPMA *buffer;
	cErr(cudaMallocHost(&buffer, sizeof(Multi_GPMA)));

	num_nodes = atol(W.Strings[0]);
	num_edges = atol(W.Strings[1]);
	SIZE_TYPE num_blocks = (num_nodes + node_per_block - 1) / node_per_block;;
	multi_gpma = new (buffer)Multi_GPMA(num_nodes, num_blocks);
	printf("num_nodes: %u, num_edges: %u, num_blocks: %u\n", num_nodes, num_edges, num_blocks);

	outDegree  = new unsigned int[num_nodes]();
	label1 = new bool[num_nodes]();
	label2 = new bool[num_nodes]();
	value  = new unsigned int[num_nodes]();
	KEY_TYPE *h_base_keys;
	h_base_keys = new KEY_TYPE[num_edges];
	VALUE_TYPE *h_base_values;
	h_base_values = new VALUE_TYPE[num_edges];

	SIZE_TYPE *block_edge_num_tmp;
	block_edge_num_tmp = (SIZE_TYPE *)malloc(sizeof(SIZE_TYPE) * num_blocks);
	parallel_cpu_for(uint i = 0; i < num_blocks; i++) { // 可优化，边读边存
		block_edge_num_tmp[i] = 0;
	}
	parallel_cpu_for(uint i = 0; i < num_nodes; i++) {
		multi_gpma->h_node2block[i] = i / node_per_block; 
		outDegree[i] = 0;
		writeAdd(&(block_edge_num_tmp[multi_gpma->h_node2block[i]]), (uint)1);
	}

	parallel_cpu_for(uint i = 0; i < num_edges; i++) { // 可优化，边读边存
		uint from = atol(W.Strings[i * 2 + 2]);
		uint to   = atol(W.Strings[i * 2 + 3]);
		KEY_TYPE key = ((KEY_TYPE) ((KEY_TYPE)from) << 32) + to;
		writeAdd(&outDegree[from], (uint)1);
		h_base_keys[i] = key;
		h_base_values[i] = (VALUE_TYPE)((from + to) % WEIGHT_CAP + 1);
		uint block_id = multi_gpma->h_node2block[from];
		writeAdd(&(block_edge_num_tmp[block_id]), (uint)1);
	}

	cErr(cudaMemcpy(multi_gpma->d_node2block, multi_gpma->h_node2block, num_nodes * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));

	parallel_cpu_for(uint i = 0; i < num_blocks; i++) {
		multi_gpma->block_edge_num[i] += block_edge_num_tmp[i];
	}
	free(block_edge_num_tmp);

	printf("Graph file is loaded.\n");
	W.del();

	malloc_trim(0);

	readtimeing = start1.Finish();
	cout << "Graph change fscanf finished in " << readtimeing/1000 << " (s).\n";
	multi_gpma->update_keys_size = num_nodes + num_edges;
	cErr(cudaMalloc(&multi_gpma->tmp_keys_array, sizeof(KEY_TYPE) * multi_gpma->update_keys_size));
	cErr(cudaMalloc(&multi_gpma->tmp_values_array, sizeof(VALUE_TYPE) * multi_gpma->update_keys_size));
	cErr(cudaMalloc(&multi_gpma->tmp_label_array, sizeof(KEY_TYPE) * multi_gpma->update_keys_size));
	cErr(cudaMalloc(&multi_gpma->tmp_exscan_array, sizeof(KEY_TYPE) * multi_gpma->update_keys_size));

	cErr(cudaMalloc(&multi_gpma->update_keys, sizeof(KEY_TYPE) * multi_gpma->update_keys_size));
	cErr(cudaMalloc(&multi_gpma->update_values, sizeof(VALUE_TYPE) * multi_gpma->update_keys_size));
	cErr(cudaMalloc(&multi_gpma->update_nodes, sizeof(KEY_TYPE) * multi_gpma->update_keys_size));
	cErr(cudaMalloc(&multi_gpma->unique_update_nodes, sizeof(KEY_TYPE) * multi_gpma->update_keys_size));
	cErr(cudaMalloc(&multi_gpma->update_offset, sizeof(SIZE_TYPE) *(multi_gpma->update_keys_size + 1)));

	cudaMemcpy(multi_gpma->update_keys, h_base_keys, num_edges * sizeof(KEY_TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(multi_gpma->update_values, h_base_values, num_edges * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);
	SIZE_TYPE THREADS_NUM = 32;
	SIZE_TYPE BLOCKS_NUM;
	BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, num_nodes);
	init_row_wall<<<BLOCKS_NUM, THREADS_NUM>>>(multi_gpma->update_keys + num_edges, num_nodes);
	memset_kernel<VALUE_TYPE> <<<BLOCKS_NUM, THREADS_NUM>>>(multi_gpma->update_values + num_edges, 1, num_nodes);

	resize_gpmas_batch(multi_gpma);
	init_gpmas_keys_values_batch(multi_gpma);
	update_gpma_stage1(multi_gpma);
	update_gpma_stage2(multi_gpma);
	update_gpma_stage3(multi_gpma);
	update_gpma_stage4(multi_gpma);

	cudaFree(multi_gpma->tmp_label_array);
	cudaFree(multi_gpma->tmp_exscan_array);
	cudaFree(multi_gpma->tmp_keys_array);
	cudaFree(multi_gpma->tmp_values_array);

	cudaFree(multi_gpma->update_keys);
	cudaFree(multi_gpma->update_values);
	cudaFree(multi_gpma->update_nodes);
	cudaFree(multi_gpma->unique_update_nodes);
	cudaFree(multi_gpma->update_offset);
	free(h_base_keys);

	gpuErrorcheck(cudaMalloc(&d_outDegree, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_value, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_label1, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label2, num_nodes * sizeof(bool)));

	cout << "Done reading.\n";
	cout << "Number of nodes = " << num_nodes << endl;
	cout << "Number of edges = " << num_edges << endl;
}


template class Graph<OutEdge>;
template class Graph<OutEdgeWeighted>;

template class GraphGPUMultiGPMA<OutEdge>;
template class GraphGPUMultiGPMA<OutEdgeWeighted>;

template class GraphPR<OutEdge>;
template class GraphPR<OutEdgeWeighted>;
