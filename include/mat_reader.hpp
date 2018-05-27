#ifndef _MAT_READER_HPP_
#define _MAT_READER_HPP_
#include <mat.h>
#include <matrix.h>
#include <string>
#include <complex>
#include <iostream>

namespace localization
{
	void read_mat(std::string path, std::string var_name)
	{
		MATFile *pmatFile = NULL;
		mxArray *pMxArray = NULL;

		// 读取.mat文件（例：mat文件名为"initUrban.mat"，其中包含"initA"）  
		double *initA;

		pmatFile = matOpen(path.c_str(), "r");
		pMxArray = matGetVariable(pmatFile, var_name.c_str());
		initA = (double*)mxGetData(pMxArray);
		auto M = mxGetM(pMxArray);
		auto N = mxGetN(pMxArray);
		/*Matrix<double> A(M, N);
		for (int i = 0; i<M; i++)
			for (int j = 0; j<N; j++)
				A[i][j] = initA[M*j + i];*/

		matClose(pmatFile);
		mxFree(initA);
	}
}

#endif // !_MAT_READER_HPP_

