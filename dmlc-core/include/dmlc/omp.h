/*!
 *  Copyright (c) 2015 by Contributors
 * \file omp.h
 * \brief header to handle OpenMP compatibility issues
 */
#ifndef DMLC_OMP_H_
#define DMLC_OMP_H_
#if defined(_OPENMP)
#include <omp.h>
#else
#ifndef DISABLE_OPENMP
// use pragma message instead of warning
#pragma message("Warning: OpenMP is not available, "                    \
                "project will be compiled into single-thread code. "     \
                "Use OpenMP-enabled compiler to get benefit of multi-threading.")
#endif
//! \cond Doxygen_Suppress
#ifdef __cplusplus
extern "C" {
# define __GOMP_NOTHROW throw ()
#else
# define __GOMP_NOTHROW __attribute__((__nothrow__))
#endif
inline int omp_get_thread_num() __GOMP_NOTHROW { return 0; }
inline int omp_get_num_threads() __GOMP_NOTHROW { return 1; }
inline int omp_get_max_threads() __GOMP_NOTHROW { return 1; }
inline int omp_get_num_procs() __GOMP_NOTHROW { return 1; }
inline void omp_set_num_threads(int nthread) __GOMP_NOTHROW {}
#ifdef __cplusplus
}
#endif
#endif
// loop variable used in openmp
namespace dmlc {
#ifdef _MSC_VER
typedef int omp_uint;
typedef long omp_ulong;  // NOLINT(*)
#else
typedef unsigned omp_uint;
typedef unsigned long omp_ulong; // NOLINT(*)
#endif
//! \endcond
}  // namespace dmlc
#endif  // DMLC_OMP_H_
