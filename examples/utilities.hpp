#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <utility>
#include <algorithm>
#include <ext/hash_map>
#include <ext/hash_set>

/* Definitions for all the parameters and constants that we will ever use */
static const int ROOT                    = 0;
static const int SKETCH_LEFT             = 0;
static const int SKETCH_RIGHT            = 1;
static const int SKETCH_LEFT_RIGHT       = 2;

static const int USE_RANDOM_INDEX        = 0;
static const int RAND_SEED_INDEX         = 1;
static const int DEBUG_INDEX             = 2;
static const int M_INDEX                 = 3;
static const int N_INDEX                 = 4;
static const int N_RHS_INDEX             = 5;
static const int S_INDEX                 = 6;
static const int NUM_THREADS_INDEX       = 7;
static const int SKETCH_DIRECTION_INDEX  = 8;
static const int NUM_INT_PARAMETERS      = 9;
extern int int_params[NUM_INT_PARAMETERS];

static const int A_FILE_PATH_INDEX       = 0;
static const int B_FILE_PATH_INDEX       = 1;
static const int SA_FILE_PATH_INDEX      = 2;
static const int TRANSFORM_INDEX         = 3;
static const int NUM_CHR_PARAMETERS      = 4;
extern char* chr_params[NUM_CHR_PARAMETERS];

namespace std {
  /* Specialize std::equal_to for strings; used for parameter passing */
  template <> 
  struct equal_to<const char*> {
    bool operator()(const char* one, const char* two) const {
      return (0==strcmp(one,two));
    }
  };

  /* Specialize std::equal_to for pairs; used for parameter passing */
  template <typename T1, typename T2> 
  struct equal_to<std::pair<T1,T2> >{
    bool operator()(const std::pair<T1,T2>& one, 
                    const std::pair<T1,T2>& two) const {
      return (one.first==two.first && one.second==two.second);
    }
  };
}

#endif // UTILITIES_HPP
