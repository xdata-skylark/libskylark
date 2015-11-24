#ifndef PARSER_HPP
#define PARSER_HPP

/**
 * @author pkambadu
 */

#include <string>
#include <cstdlib>
#include <unordered_map>

#include "utilities.hpp"

static const char* description = "Example for SKYLARK";
typedef std::unordered_map<const char*, const char*> string_string_map_t;
typedef std::unordered_map<const char*, int> string_int_map_t;

static inline void print_help (const string_string_map_t& help_map,
                               const char* option=NULL,
                               const char* val=NULL) {
  if (NULL != option) printf ("%s: Invalid arguments \"%s\" is \"%s\"\n",
                                                description, option, val);
  else printf ("%s\n", description);
  string_string_map_t::const_iterator iter = help_map.begin();

  while (iter != help_map.end()) {
    printf ("%s: %s\n", (*iter).first, (*iter).second);
    ++iter;
  }

  exit (3);
}

static inline void parse_parameters (int argc, char** argv) {
  /** Set up the command line arguments */
  string_string_map_t help_map;
  help_map["help"] = "produce help messages";

  help_map["r"] = "should we use randomly generated matrices (default:1)";
  help_map["s"] = "seed for the random number generator (default:0)";
  help_map["debug"] = "print helpful messages out (default:0)";
  help_map["M"] = "number of rows in the distributed matrix (default:10)";
  help_map["N"] = "number of cols in the distributed matrix (default:5)";
  help_map["RHS"] = "number of cols in the RHS (default:1)";
  help_map["S"] = "reduced number of rows/cols after sketching (default:5)";
  help_map["num-threads"] = "number of threads to start (default:2)";
  help_map["sketch"] = "Direction of the sketch (left|right|both, def:left)";
  help_map["A-file"] = "path to matrix A";
  help_map["b-file"] = "path to matrix b";
  help_map["sA-file"] = "path to matrix sA";
  help_map["s-type"] = "transform to use (JLT|FJLT|CWT, default: JLT)";
  help_map["m-type"] = "matrix type (elem|cblas, default: elem)";

  string_int_map_t int_options_map;
  int_options_map["r"]                 = USE_RANDOM_INDEX;
  int_options_map["s"]                 = RAND_SEED_INDEX;
  int_options_map["debug"]             = DEBUG_INDEX;
  int_options_map["M"]                 = M_INDEX;
  int_options_map["N"]                 = N_INDEX;
  int_options_map["RHS"]               = N_RHS_INDEX;
  int_options_map["S"]                 = S_INDEX;
  int_options_map["num-threads"]       = NUM_THREADS_INDEX;
  int_options_map["sketch"]            = SKETCH_DIRECTION_INDEX;

  string_int_map_t chr_options_map;
  chr_options_map["A-file"]       = A_FILE_PATH_INDEX;
  chr_options_map["b-file"]       = B_FILE_PATH_INDEX;
  chr_options_map["sA-file"]      = SA_FILE_PATH_INDEX;
  chr_options_map["s-type"]       = TRANSFORM_INDEX;
  chr_options_map["m-type"]       = MATRIX_TYPE_INDEX;

  /* default initialize parameters */
  int_params[USE_RANDOM_INDEX]      = 1;
  int_params[RAND_SEED_INDEX]       = 0;
  int_params[DEBUG_INDEX]           = 0;
  int_params[M_INDEX]               = 10;
  int_params[N_INDEX]               = 5;
  int_params[N_RHS_INDEX]           = 1;
  int_params[S_INDEX]               = 5;
  int_params[NUM_THREADS_INDEX]     = 2;
  int_params[SKETCH_DIRECTION_INDEX]= SKETCH_LEFT;

  chr_params[A_FILE_PATH_INDEX]     = "";
  chr_params[B_FILE_PATH_INDEX]     = "";
  chr_params[SA_FILE_PATH_INDEX]    = "";
  chr_params[TRANSFORM_INDEX]       = "JLT";
  chr_params[MATRIX_TYPE_INDEX]     = "elem";

  /* parse the command line */
  if (!(argc&0x1)) print_help(help_map);

  for (int i=1; i<argc; i+=2) {
    char* option = argv[i];
    char* value = argv[i+1];

    if (0==strcmp(option,"help")) print_help (help_map);
    else if (int_options_map.end()!=int_options_map.find(option)) {
      int_params[int_options_map[option]] = atoi(value);
    } else if (chr_options_map.end()!=chr_options_map.find(option)) {
      chr_params[chr_options_map[option]] = value;
    } else {
      print_help (help_map,option);
    }
  }

  /* Make sure that all the parameters are given and are correct */
  if (0==int_params[USE_RANDOM_INDEX] &&
      0==strcmp("",chr_params[A_FILE_PATH_INDEX])) {
    print_help (help_map, "A-file");
  } else if (0==int_params[USE_RANDOM_INDEX] &&
             0==strcmp("",chr_params[B_FILE_PATH_INDEX])) {
    print_help (help_map, "A-file");
  } else if (0>int_params[M_INDEX]) {
    print_help (help_map, "M");
  } else if (0>int_params[N_INDEX]) {
    print_help (help_map, "N");
  } else if (0>int_params[N_RHS_INDEX]) {
    print_help (help_map, "RHS");
  } else if (0!=strcmp("JLT", chr_params[TRANSFORM_INDEX]) &&
             0!=strcmp("FJLT", chr_params[TRANSFORM_INDEX]) &&
             0!=strcmp("CWT", chr_params[TRANSFORM_INDEX])) {
    print_help (help_map, "type");
  }
}

#endif // PARSER_HPP
