#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <immintrin.h>
#include <omp.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include "bits/stdc++.h"
#include <pthread.h>
#include <sched.h>
// std:: int32_t r=1;

void comp(float *img, float *result, const float kernel[3][3], const std::int32_t num_rows, const std::int32_t num_cols, int start, int end)
{

	int tred = 6;
	std::int32_t u = (num_cols - 2) / 8;
	u = u / tred;
	std::int32_t rem = u % tred;
	std::int32_t k;
	float sum = 0.0;
	int i, j;
	int ky, kx;
	int di, dj, ni, nj;
	__m256 sum1;
	__m256 pixels;
	__m256 filterVal;
	for (std::int32_t r = start; r < end; r++)
	{

		for (k = r * num_cols + 1; k < (((r + 1) * num_cols) - 1); k++)
		{
			sum = 0.0;
			i = k / num_cols, j = k % num_cols;
			if ((j <= num_cols - 9))
			{
				// float res[8];
				sum1 = _mm256_setzero_ps();
				for (ky = -1; ky <= 1; ++ky)
				{
					for (kx = -1; kx <= 1; ++kx)
					{
						pixels = _mm256_loadu_ps(img + (i + ky) * num_cols + (j + kx));
						filterVal = _mm256_set1_ps(kernel[(ky + 1)][(kx + 1)]);
						sum1 = _mm256_fmadd_ps(pixels, filterVal, sum1);
					}
				}
				_mm256_storeu_ps(result + i * num_cols + j, sum1);
				// sol_fs.write(reinterpret_cast<char*>(result+i*num_cols +j), sizeof(res));
				k += 7;
			}

			else
			{
				for (di = -1; di <= 1; di++)
				{
					for (dj = -1; dj <= 1; dj++)
					{
						ni = i + di, nj = j + dj;
						if (ni >= 0 and ni < num_rows and nj >= 0 and nj < num_cols)
							sum += kernel[di + 1][dj + 1] * img[ni * num_cols + nj];
					}
				}
				result[k] = sum;
			}
		}
	}
}

namespace solution
{
	std::string compute(const std::string &bitmap_path, const float kernel[3][3], const std::int32_t num_rows, const std::int32_t num_cols)
	{
		std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.bmp";
		int fd_input = open(bitmap_path.c_str(), O_RDONLY);
		int fd_output = open(sol_path.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
		if (fd_input == -1 || fd_output == -1)
		{
			perror("open");
			// Handle file open error
			return "";
		}

		off_t file_size = num_rows * num_cols * sizeof(float);
		if (ftruncate(fd_output, file_size) == -1)
		{
			perror("ftruncate");
			// Handle file truncation error
			close(fd_input);
			close(fd_output);
			return "";
		}

		float *img = reinterpret_cast<float *>(mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd_input, 0));
		float *result = reinterpret_cast<float *>(mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_output, 0));
		if (img == MAP_FAILED || result == MAP_FAILED)
		{
			perror("mmap");
			// Handle mmap error
			close(fd_input);
			close(fd_output);
			return "";
		}

		int arr[9];
		arr[0] = -1;
		arr[1] = 0;
		arr[2] = 1;
		arr[3] = -1;
		arr[4] = 0;
		arr[5] = 1;
		arr[6] = -1;
		arr[7] = 0;
		arr[8] = 1;

		// ker.push_back();

		int num_thread = 24;
#pragma omp parallel num_threads(24)
		{
			cpu_set_t cpuset;
			CPU_ZERO(&cpuset);
			CPU_SET(omp_get_thread_num() * 2, &cpuset);
			pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

#pragma omp single
			{
				int chunk_sz = (num_rows - 2) / num_thread;
				// std:: cout<<num_rows-1<<std:: endl;
				// int start = 1, end = (num_rows - 1) / num_threads;
				int o;
				std::int32_t k;
				float sum = 0.0;
				int i, j;
				int ky, kx;
				int di, dj, ni, nj;
				__m256 sum1;
				__m256 pixels;
				__m256 filterVal;
				std::int32_t start, end;
				int e;
				// #pragma omp parallel for private(o,k,i,j,ky,kx,di,dj,ni,nj,sum1,pixels,filterVal,start,end,e)  num_threads(num_thread)
				for (o = 0; o < (num_thread); o++)
				{
					{
#pragma omp task
						{
							start = (o * chunk_sz + 1), end = ((o + 1) * chunk_sz + 1);
							for (std::int32_t r = start; r < end; r++)
							{
								e = ((r + 1) * num_cols) - 8;
								// e=end*num_cols-1;
								// #pragma omp parallel for
								for (k = r * num_cols + 1; k < e; k++)
								{
									sum = 0.0;
									i = k / num_cols, j = k % num_cols;
									{
										// float res[8];
										sum1 = _mm256_setzero_ps();
										for (int u = 0; u < 9; u++)
										{
											ky = u / 3 - 1;
											kx = arr[u];
											pixels = _mm256_loadu_ps(img + (i + ky) * num_cols + (j + kx));
											filterVal = _mm256_set1_ps(kernel[(ky + 1)][(kx + 1)]);
											sum1 = _mm256_fmadd_ps(pixels, filterVal, sum1);
										}

										_mm256_storeu_ps(result + i * num_cols + j, sum1);
										// sol_fs.write(reinterpret_cast<char*>(result+i*num_cols +j), sizeof(res));
										k += 7;
									}
								}

								for (; k < ((r + 1) * num_cols - 1); k++)
								{
									sum = 0.0;
									i = k / num_cols, j = k % num_cols;
									for (di = -1; di <= 1; di++)
									{
										for (dj = -1; dj <= 1; dj++)
										{
											ni = i + di, nj = j + dj;
											if (ni >= 0 and ni < num_rows and nj >= 0)
												sum += kernel[di + 1][dj + 1] * img[ni * num_cols + nj];
										}
									}
									result[k] = sum;
								}
							}
						}

						// }
					}
					// start = end;
					// end += (num_rows - 1) / num_threads;
				}
#pragma omp taskwait
#pragma omp task
				{
					comp(img, result, kernel, num_rows, num_cols, num_thread * chunk_sz + 1, num_rows - 1);
				}
#pragma omp task
				{
					for (o = 0; o < num_rows; o++)
					{
						k = o * num_cols;
						sum = 0.0;
						i = k / num_cols, j = k % num_cols;
						for (di = -1; di <= 1; di++)
						{
							for (dj = -1; dj <= 1; dj++)
							{
								ni = i + di, nj = j + dj;
								if (ni >= 0 and ni < num_rows and nj >= 0)
									sum += kernel[di + 1][dj + 1] * img[ni * num_cols + nj];
							}
						}
						result[k] = sum;

						k = o * num_cols + num_cols - 1;
						sum = 0.0;
						i = k / num_cols, j = k % num_cols;
						for (di = -1; di <= 1; di++)
						{
							for (dj = -1; dj <= 1; dj++)
							{
								ni = i + di, nj = j + dj;
								if (ni >= 0 and ni < num_rows and nj < num_cols)
									sum += kernel[di + 1][dj + 1] * img[ni * num_cols + nj];
							}
						}
						result[k] = sum;
						// sol_fs.write(reinterpret_cast<char*>(&sum), sizeof(sum));
					}
				}

#pragma omp task
				{
					// #pragma omp parallel for
					for (std::int32_t k = 0; k < num_cols; k++)
					{
						sum = 0.0;
						i = k / num_cols, j = k % num_cols;
						for (di = -1; di <= 1; di++)
						{
							for (dj = -1; dj <= 1; dj++)
							{
								ni = i + di, nj = j + dj;
								if (ni >= 0 and nj >= 0 and nj < num_cols)
									sum += kernel[di + 1][dj + 1] * img[ni * num_cols + nj];
							}
						}
						result[k] = sum;
						// sol_fs.write(reinterpret_cast<char*>(&sum), sizeof(sum));
					}
				}

#pragma omp task
				{
					// #pragma omp parallel for
					for (std::int32_t k = (num_rows - 1) * num_cols; k < num_rows * num_cols; k++)
					{
						sum = 0.0;
						i = k / num_cols, j = k % num_cols;
						for (di = -1; di <= 1; di++)
							for (dj = -1; dj <= 1; dj++)
							{
								ni = i + di, nj = j + dj;
								if (ni < num_rows and nj >= 0 and nj < num_cols)
									sum += kernel[di + 1][dj + 1] * img[ni * num_cols + nj];
							}
						result[k] = sum;
						// sol_fs.write(reinterpret_cast<char*>(&sum), sizeof(sum));
					}
				} // std:: cout << start<<std:: endl;
				  // std:: cout << end<<std:: endl;
			}
		}
		// Unmap memory and close file descriptors

		munmap(img, file_size);
		munmap(result, file_size);
		// close(fd_input);
		// close(fd_output);
		return sol_path;
	}
};
